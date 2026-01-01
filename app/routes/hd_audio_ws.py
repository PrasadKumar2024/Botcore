# app/routes/hd_audio_ws.py
import os
import io
import json
import time
import base64
import logging
import functools
import asyncio
import threading
import queue
import audioop
import wave
import re
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Google clients
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account

# local services
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------- Config ----------------
EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "6"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "3"))
STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "10.0"))
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "14.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "18.0"))

STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "0.35"))
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "10"))
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)

DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", os.getenv("DEFAULT_KB_CLIENT_ID", "default"))

BYTES_PER_SEC = STT_SAMPLE_RATE * 2
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

# Throttle queue thresholds (tunable)
THROTTLE_QSIZE_HIGH = int(os.getenv("HD_WS_THROTTLE_HIGH", "200"))
THROTTLE_QSIZE_LOW  = int(os.getenv("HD_WS_THROTTLE_LOW", "100"))

executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
global_tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)

# VAD & debounce tuning
SILENCE_TIMEOUT = float(os.getenv("HD_WS_SILENCE_TIMEOUT", "0.9"))
DEBOUNCE_SECONDS = float(os.getenv("HD_WS_DEBOUNCE_S", "0.4"))
VAD_THRESHOLD = int(os.getenv("HD_WS_VAD_THRESHOLD", "300"))
MIN_RESTART_INTERVAL = float(os.getenv("HD_WS_MIN_RESTART_INTERVAL", "2.0"))

# -------------- Google clients --------------
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = tts.TextToSpeechClient(credentials=_creds)

# -------------- LLM --------------
_gemini = GeminiService()

# -------------- Storage: optional Redis for session persistence --------------
REDIS_URL = os.getenv("REDIS_URL", "")
_redis = None
try:
    if REDIS_URL:
        import redis as _redis_pkg
        _redis = _redis_pkg.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis session store enabled")
except Exception:
    _redis = None
    logger.info("Redis not available — falling back to in-memory session store")

_memory_sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

def append_session_turn(session_id: str, role: str, text: str):
    entry = {"role": role, "text": text, "ts": time.time()}
    try:
        if _redis:
            key = f"session:{session_id}:history"
            _redis.rpush(key, json.dumps(entry))
            _redis.expire(key, 60 * 60 * 24 * 7)
        else:
            _memory_sessions[session_id].append(entry)
            if len(_memory_sessions[session_id]) > 500:
                _memory_sessions[session_id] = _memory_sessions[session_id][-500:]
    except Exception:
        logger.exception("append_session_turn failure")

def get_recent_turns(session_id: str, n: int = 12):
    try:
        if _redis:
            key = f"session:{session_id}:history"
            items = _redis.lrange(key, -n, -1)
            return [json.loads(i) for i in items]
        else:
            return _memory_sessions.get(session_id, [])[-n:]
    except Exception:
        logger.exception("get_recent_turns failure")
        return []

# -------------- Voice utils --------------
VOICE_MAP = {
    "en-IN": {"name": "en-IN-Neural2-C", "gender": "MALE"},
    "en-US": {"name": "en-US-Neural2-A", "gender": "FEMALE"},
    "hi-IN": {"name": "hi-IN-Neural2-A", "gender": "FEMALE"},
}
DEFAULT_VOICE = VOICE_MAP["en-IN"]

def get_best_voice(language_code: Optional[str]):
    if not language_code:
        return ("en-IN", DEFAULT_VOICE["name"], DEFAULT_VOICE.get("gender"))
    if language_code in VOICE_MAP:
        v = VOICE_MAP[language_code]
        return (language_code, v["name"], v.get("gender"))
    base = language_code.split("-")[0] if "-" in language_code else language_code
    fallback = {"en": "en-IN", "hi": "hi-IN"}
    if base in fallback:
        f = fallback[base]
        v = VOICE_MAP.get(f, DEFAULT_VOICE)
        return (f, v["name"], v.get("gender"))
    return ("en-IN", DEFAULT_VOICE["name"], DEFAULT_VOICE.get("gender"))

def ssml_for_text(text: str, sentiment: float = 0.0, prosody_rate: float = 0.95) -> str:
    s = max(-1.0, min(1.0, sentiment or 0.0))
    rate = prosody_rate
    pitch = "0st"
    volume = "medium"
    
    # Simple Prosody Mapping based on sentiment
    if s >= 0.5:
        rate = str(prosody_rate * 1.05)
        pitch = "+1st"
        volume = "loud"
    elif s >= 0.2:
        rate = str(prosody_rate * 1.02)
    elif s <= -0.4:
        rate = str(prosody_rate * 0.90)
        pitch = "-1st"
        volume = "soft"
    
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Micro-pauses for naturalness
    esc = esc.replace(", ", ", <break time='100ms'/> ")
    esc = esc.replace(". ", ". <break time='250ms'/> ")
    esc = esc.replace("? ", "? <break time='250ms'/> ")
    return f"<speak><prosody rate='{rate}' pitch='{pitch}' volume='{volume}'>{esc}</prosody></speak>"

def make_wav_from_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def is_silence(pcm16: bytes, threshold: int = VAD_THRESHOLD) -> bool:
    try:
        return audioop.rms(pcm16, 2) < threshold
    except Exception:
        return False

# -------------- Advanced Acoustic Heuristics --------------
def calculate_audio_energy(pcm16: bytes) -> str:
    """
    Returns a descriptive energy level to feed into the LLM context.
    """
    try:
        rms = audioop.rms(pcm16, 2)
        if rms < 400: return "whispering/quiet"
        if rms < 1500: return "normal volume"
        if rms < 3500: return "energetic/loud"
        return "shouting/very loud"
    except Exception:
        return "normal volume"

# -------------- Text Sentiment --------------
_POS_WORDS = {"good","great","happy","awesome","fantastic","love","thanks","thank","yes","sure","okay","ok"}
_NEG_WORDS = {"bad","sad","angry","upset","hate","problem","frustrat","frustration","not working","issue","no","don't","cannot"}

def quick_sentiment_score_text(text: str) -> float:
    if not text:
        return 0.0
    tl = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in tl)
    neg = sum(1 for w in _NEG_WORDS if w in tl)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1, pos + neg)
    return max(-1.0, min(1.0, score))

# ----------------- TTS blocking wrapper -----------------
def _sync_tts_linear16(ssml: str, language_code: str, voice_name: str, gender: Optional[str], sample_rate_hz: int = 24000):
    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_cfg_kwargs = {"audio_encoding": tts.AudioEncoding.LINEAR16, "sample_rate_hertz": sample_rate_hz}
    audio_config = tts.AudioConfig(**audio_cfg_kwargs)
    synthesis_input = tts.SynthesisInput(ssml=ssml)
    return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

async def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", sample_rate_hz: int = 24000, sentiment: float = 0.0) -> Optional[bytes]:
    ssml = ssml_for_text(text, sentiment=sentiment, prosody_rate=0.95)
    lang_code, voice_name, gender = get_best_voice(language_code)
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(global_tts_semaphore.acquire(), timeout=3.0)
    except Exception:
        logger.warning("TTS queue busy")
        return None
    try:
        fut = loop.run_in_executor(executor, functools.partial(_sync_tts_linear16, ssml, lang_code, voice_name, gender, sample_rate_hz))
        resp = await asyncio.wait_for(fut, timeout=TTS_TIMEOUT)
        return resp.audio_content
    except asyncio.TimeoutError:
        logger.warning("TTS timed out")
        return None
    except Exception as e:
        logger.exception("TTS error: %s", e)
        return None
    finally:
        try:
            global_tts_semaphore.release()
        except Exception:
            pass

# ----------------- Query normalization -----------------
_CONTRACTIONS = {
    r"\bwhat's\b": "what are",
    r"\bwhats\b": "what are",
    r"\bwhere's\b": "where is",
    r"\bwhen's\b": "when are",
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
}

def normalize_and_expand_query(transcript: str) -> str:
    if not transcript:
        return ""
    s = transcript.lower().strip()
    for patt, repl in _CONTRACTIONS.items():
        s = re.sub(patt, repl, s)
    toks = s.split()
    dedup = []
    prev = None
    for t in toks:
        if t != prev:
            dedup.append(t)
        prev = t
    mappings = {
        "timings": "business hours operating hours schedule",
        "timing": "business hours operating hours schedule",
        "phone number": "contact number telephone",
        "open": "business hours operating hours",
        "closed": "business hours operating hours",
        "appointment": "appointment booking consultation",
        "doctor": "doctor physician consultant",
    }
    out = []
    i = 0
    while i < len(dedup):
        two = " ".join(dedup[i:i+2]) if i+1 < len(dedup) else None
        if two and two in mappings:
            out.extend(mappings[two].split())
            i += 2
            continue
        w = dedup[i]
        out.append(w)
        if w in mappings:
            out.extend(mappings[w].split())
        i += 1
    return " ".join(out)

# ----------------- STT worker (robust) -----------------
def grpc_stt_worker(loop, audio_queue: queue.Queue, transcripts_queue: asyncio.Queue, stop_event: threading.Event, language_code: str):
    def gen_requests_with_config():
        cfg = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
        )
        streaming_cfg = speech.StreamingRecognitionConfig(config=cfg, interim_results=True, single_utterance=False)
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_cfg)
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error pulling audio chunk in STT worker: %s", e)
                if stop_event.is_set():
                    break

    def gen_requests_audio_only():
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error pulling audio chunk in STT worker (audio_only): %s", e)
                if stop_event.is_set():
                    break

    def call_streaming_recognize_with_fallback():
        try:
            return _speech_client.streaming_recognize(gen_requests_with_config())
        except TypeError as e:
            # Fallback for older google-cloud-speech versions
            cfg = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=STT_SAMPLE_RATE,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="default",
                use_enhanced=True,
            )
            streaming_cfg = speech.StreamingRecognitionConfig(config=cfg, interim_results=True, single_utterance=False)
            return _speech_client.streaming_recognize(streaming_cfg, gen_requests_audio_only())

    try:
        logger.info("Starting STT worker (thread) language=%s", language_code)
        responses = call_streaming_recognize_with_fallback()
        for response in responses:
            if stop_event.is_set():
                break
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(response), loop)
    except Exception as e:
        logger.exception("STT worker error: %s", e)
    finally:
        logger.info("STT worker exiting (language=%s)", language_code)

# ----------------- Metrics (in-memory) -----------------
METRICS = {
    "requests": 0,
    "intent_counts": defaultdict(int),
    "sentiments": [],
    "confidences": [],
    "avg_response_ms": [],
    "drops_oldest": 0,
    "drops_newest": 0,
}

def record_metric_intent(intent: str):
    if not intent: return
    METRICS["intent_counts"][intent] += 1

def record_metric_sentiment(s: float):
    METRICS["sentiments"].append(s)

def record_metric_confidence(c: float):
    METRICS["confidences"].append(c)

def record_latency(ms: float):
    METRICS["avg_response_ms"].append(ms)

# ----------------- FAST PATTERNS (Updated: REMOVED GREETINGS) -----------------
# We removed "Good Morning" etc. from here so the LLM can handle them naturally.
FAST_PATTERNS = [
    (re.compile(r"\b(hours|open|close|opening|closing|timings)\b", re.I), "business_hours"),
    (re.compile(r"\b(phone|contact|call|number)\b", re.I), "contact_number"),
    # Appointment kept as regex for speed, but you could remove this too if you want LLM to handle it.
    (re.compile(r"\b(appointment|book|booking|consultation)\b", re.I), "appointment"),
]

FAST_RESPONSES = {
    "business_hours": "We are open Monday to Friday, 10 AM to 6 PM. Shall I book an appointment?",
    "contact_number": "Our number is +1-800-555-0123. Shall I text it to you?",
    "appointment": "I can help with that. Do you need an in-person visit or a remote consultation?",
}

def fast_intent_match(text: str) -> Optional[Tuple[str, str]]:
    t = text.lower()
    for patt, label in FAST_PATTERNS:
        if patt.search(t):
            return (label, FAST_RESPONSES.get(label, ""))
    return None

def get_time_of_day_context() -> str:
    h = datetime.now().hour
    if 5 <= h < 12: return "Morning"
    if 12 <= h < 17: return "Afternoon"
    if 17 <= h < 22: return "Evening"
    return "Night"

# ----------------- Helper: robust JSON parsing from model text -----------------
def extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_blob = s[start:end+1]
    try:
        return json.loads(json_blob)
    except Exception:
        try:
            safe = json_blob.replace("\n", " ").replace("'", '"')
            return json.loads(safe)
        except Exception:
            return None

# ----------------- WebSocket handler -----------------
@router.websocket("/ws/hd-audio")
async def hd_audio_ws(ws: WebSocket):
    token = ws.query_params.get("token")
    if WEBSOCKET_API_TOKEN:
        if not token or token != WEBSOCKET_API_TOKEN:
            await ws.accept()
            await ws.send_text(json.dumps({"type": "error", "error": "unauthorized"}))
            await ws.close()
            logger.warning("WS rejected unauthenticated connection")
            return

    await ws.accept()
    logger.info("HD WS accepted connection")

    audio_queue = queue.Queue(maxsize=400)
    transcripts_queue = asyncio.Queue()
    stop_event = threading.Event()

    language = "en-IN"
    stt_thread = None

    conversation: deque = deque(maxlen=12)
    utterance_buffer: List[str] = []
    pending_debounce_task: Optional[asyncio.Task] = None

    last_voice_ts = time.time()
    restarting_lock = threading.Lock()
    last_restart_ts = 0.0

    is_bot_speaking = False
    current_tts_task: Optional[asyncio.Task] = None
    current_stream_stop_event: Optional[threading.Event] = None
    
    # Store the last known audio energy level
    last_audio_energy_desc = "normal volume"

    throttled = False 

    def push_audio_chunk(pcm: bytes):
        try:
            audio_queue.put_nowait(pcm)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except Exception: pass
            try:
                audio_queue.put_nowait(pcm)
                METRICS["drops_oldest"] += 1
            except Exception:
                METRICS["drops_newest"] += 1

    async def maybe_send_throttle():
        nonlocal throttled
        try:
            q = audio_queue.qsize()
            if q > THROTTLE_QSIZE_HIGH and not throttled:
                throttled = True
                await ws.send_text(json.dumps({"type":"control","action":"throttle"}))
            elif q < THROTTLE_QSIZE_LOW and throttled:
                throttled = False
                await ws.send_text(json.dumps({"type":"control","action":"resume"}))
        except Exception:
            pass

    async def _do_tts_and_send(ai_text: str, language_code: str, sentiment: float):
        nonlocal is_bot_speaking, current_tts_task
        try:
            is_bot_speaking = True
            pcm = await synthesize_text_to_pcm(ai_text, language_code=language_code, sample_rate_hz=24000, sentiment=sentiment)
            if current_tts_task and current_tts_task.cancelled():
                return
            if pcm:
                wav_bytes = make_wav_from_pcm16(pcm, sample_rate=24000)
                b64wav = base64.b64encode(wav_bytes).decode("ascii")
                try:
                    await ws.send_text(json.dumps({"type":"audio", "audio": b64wav}))
                except Exception: pass
            else:
                try:
                    await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
                except: pass
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("TTS/send failed: %s", e)
        finally:
            is_bot_speaking = False
            current_tts_task = None

    async def send_tts_and_audio(ai_text: str, language_code: str, sentiment: float = 0.0):
        nonlocal current_tts_task
        if current_tts_task and not current_tts_task.done():
            try: current_tts_task.cancel()
            except Exception: pass
        current_tts_task = asyncio.create_task(_do_tts_and_send(ai_text, language_code, sentiment))
        try: await current_tts_task
        except asyncio.CancelledError: return

    # helper to run streaming LLM for spoken output
    def _stream_writer_thread(loop_ref, prompt: str, system_msg: str, token_queue_async: asyncio.Queue, stop_evt: threading.Event):
        try:
            gen = _gemini.generate_stream(prompt=prompt, system_message=system_msg, temperature=0.7) # Slightly higher temp for naturalness
            for chunk in gen:
                if stop_evt.is_set(): break
                try: asyncio.run_coroutine_threadsafe(token_queue_async.put(chunk), loop_ref)
                except Exception: break
        except Exception: pass
        finally:
            try: asyncio.run_coroutine_threadsafe(token_queue_async.put(None), loop_ref)
            except Exception: pass

    async def run_streaming_llm_and_tts(convo_pref: str, user_text: str, language_code: str):
        loop = asyncio.get_running_loop()
        token_q: asyncio.Queue = asyncio.Queue()
        stop_evt = threading.Event()
        
        # Enhanced System Message for Fallback Streaming
        time_context = get_time_of_day_context()
        system_msg = (
            f"You are BrightCare, a warm, human-like phone assistant. It is currently {time_context}. "
            f"User is speaking with {last_audio_energy_desc}. "
            "Reply naturally. If it's a greeting, greet back warmly. "
            "Keep answers concise (1-2 sentences) and conversational."
        )
        user_prompt = f"{convo_pref}\n\nUser said: {user_text}\nReply:"

        writer_thread = threading.Thread(target=_stream_writer_thread, args=(loop, user_prompt, system_msg, token_q, stop_evt), daemon=True)
        writer_thread.start()

        full_text = ""
        sentence_buffer = ""
        first_sentence_sent = False
        sentiment_est = 0.0

        try:
            while True:
                token = await token_q.get()
                if token is None: break
                chunk = token if isinstance(token, str) else str(token)
                full_text += chunk
                sentence_buffer += chunk

                sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)
                if len(sentences) > 1:
                    complete_sentences = sentences[:-1]
                    sentence_buffer = sentences[-1]
                else:
                    complete_sentences = []

                for s in complete_sentences:
                    s = s.strip()
                    if not s: continue
                    if not first_sentence_sent:
                        first_sentence_sent = True
                        await send_tts_and_audio(s, language_code=language_code, sentiment=quick_sentiment_score_text(s))
                    else:
                        asyncio.create_task(send_tts_and_audio(s, language_code=language_code, sentiment=quick_sentiment_score_text(s)))
            remaining = sentence_buffer.strip()
            if remaining:
                await send_tts_and_audio(remaining, language_code=language_code, sentiment=quick_sentiment_score_text(remaining))
            sentiment_est = quick_sentiment_score_text(full_text)
            return full_text, sentiment_est
        finally:
            stop_evt.set()
            try: writer_thread.join(timeout=1.0)
            except Exception: pass

    # ---------------- Combined handler with persona ----------------
    async def handle_final_utterance(text: str, last_pcm: Optional[bytes] = None, session_id: str = "default"):
        nonlocal conversation, is_bot_speaking, current_stream_stop_event
        METRICS["requests"] += 1
        start_ms = time.time() * 1000

        user_text = text.strip()
        if not user_text: return

        ts = time.time()
        conversation.append(("user", user_text, None, None, 0.0, ts))
        append_session_turn(session_id, "user", user_text)

        # 1) Fast-path regex (ONLY for emergency/logic, NO GREETINGS)
        fast = fast_intent_match(user_text)
        if fast:
            intent_label, reply = fast
            record_metric_intent(intent_label)
            conversation.append(("assistant", reply, intent_label, {}, 0.0, time.time()))
            try: await ws.send_text(json.dumps({"type":"ai_text","text":reply,"metadata":{"intent":intent_label}}))
            except: pass
            await send_tts_and_audio(reply, language_code=language, sentiment=0.0)
            record_latency(time.time() * 1000 - start_ms)
            return

        # 2) RAG search
        norm_q = normalize_and_expand_query(user_text)
        try:
            rag_results = await pinecone_service.search_similar_chunks(
                client_id=DEFAULT_CLIENT_ID,
                query=norm_q or user_text,
                top_k=5,
                min_score=-1.0
            )
        except Exception as e:
            logger.exception("Pinecone search failed: %s", e)
            rag_results = None

        def rag_confidence(results):
            try:
                scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
                if not scores: return 0.0
                return max(0.0, min(1.0, (sum(scores) / len(scores)) + 0.4))
            except Exception: return 0.0

        # Build Context for LLM
        convo_lines = []
        for entry in list(conversation)[-8:]: # Increased history depth
            role = entry[0]; txt = entry[1]
            prefix = "User:" if role == "user" else "Assistant:"
            convo_lines.append(f"{prefix} {txt}")
        convo_pref = "\n".join(convo_lines)
        
        context_text = ""
        if rag_results:
            context_text = "\n\n".join([f"{r.get('source','')}: {r.get('chunk_text','')}" for r in rag_results[:3]])
        
        confidence_est = rag_confidence(rag_results) if rag_results else 0.0
        time_context = get_time_of_day_context()

        # PRODUCTION SYSTEM PROMPT (Fixed Naturalness)
        system_msg = (
            f"You are BrightCare, a warm, empathetic phone assistant. Current time: {time_context}. "
            f"User voice energy: {last_audio_energy_desc}. "
            "INSTRUCTIONS:\n"
            "1. If User says a greeting (hi, good morning), IGNORE CONTEXT and reply warmly ('Good morning! How are you?').\n"
            "2. If User asks a specific question, use the CONTEXT below to answer strictly factually.\n"
            "3. If Context is empty or irrelevant, politely apologize and offer to take a message.\n"
            "4. Output EXACT JSON: {\"intent\":\"...\",\"spoken\":\"...\",\"sentiment\":-1.0 to 1.0}"
        )
        
        user_prompt = f"{convo_pref}\n\nCONTEXT:\n{context_text}\n\nUser utterance: {user_text}\nReturn JSON."

        # SYNCHRONOUS safe JSON call
        loop = asyncio.get_running_loop()
        try:
            partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=system_msg, temperature=0.3, max_tokens=300)
            fut = loop.run_in_executor(executor, partial)
            resp_text = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            resp_text = (resp_text or "").strip()
        except Exception:
            resp_text = ""

        parsed = extract_json_from_text(resp_text)
        
        if parsed and parsed.get("spoken"):
            intent = parsed.get("intent","")
            spoken = parsed.get("spoken","")
            sentiment = float(parsed.get("sentiment", 0.0))
            
            # Send result
            record_metric_intent(intent or "rag_response")
            conversation.append(("assistant", spoken, intent, {}, sentiment, time.time()))
            append_session_turn("default", "assistant", spoken)
            try: await ws.send_text(json.dumps({"type":"ai_text", "text": spoken, "metadata": {"intent": intent}}))
            except: pass
            
            await send_tts_and_audio(spoken, language_code=language, sentiment=sentiment)
            record_latency(time.time()*1000 - start_ms)
            return
        
        # 3) Fallback if JSON failed or RAG failed completely (use Streaming Chat)
        if hasattr(_gemini, "generate_stream"):
            try:
                if current_stream_stop_event:
                    current_stream_stop_event.set()
                current_stream_stop_event = threading.Event()
                final_text, sentiment_est = await run_streaming_llm_and_tts(convo_pref, user_text, language_code=language)
                
                conversation.append(("assistant", final_text, "chat_fallback", {}, sentiment_est, time.time()))
                append_session_turn("default", "assistant", final_text)
                try: await ws.send_text(json.dumps({"type":"ai_text","text":final_text}))
                except: pass
                return
            except Exception: pass

        # Ultimate fallback
        ai_text = "I'm having a little trouble connecting. Could you say that again?"
        await send_tts_and_audio(ai_text, language_code=language, sentiment=0.0)

    # Debounce handler
    async def debounce_and_handle():
        nonlocal pending_debounce_task, utterance_buffer, last_voice_ts
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
            if (time.time() - last_voice_ts) < (DEBOUNCE_SECONDS - 0.05): return
            text = " ".join(utterance_buffer).strip()
            utterance_buffer.clear()
            if not text: return
            await handle_final_utterance(text, last_pcm=None, session_id="default")
        finally:
            pending_debounce_task = None

    # Transcript consumer
    async def process_transcripts_task():
        nonlocal language, utterance_buffer, pending_debounce_task, is_bot_speaking, current_tts_task, current_stream_stop_event
        while True:
            resp = await transcripts_queue.get()
            if resp is None: break
            for result in resp.results:
                if not result.alternatives: continue
                alt = result.alternatives[0]
                interim_text = alt.transcript.strip()
                is_final = getattr(result, "is_final", False)

                # Barge-in
                if interim_text and is_bot_speaking:
                    try: await ws.send_text(json.dumps({"type":"control","action":"stop_playback"}))
                    except: pass
                    if current_tts_task and not current_tts_task.done():
                        try: current_tts_task.cancel()
                        except: pass
                    if current_stream_stop_event:
                        try: current_stream_stop_event.set()
                        except: pass
                    is_bot_speaking = False

                if interim_text:
                    try: await ws.send_text(json.dumps({"type":"transcript","text":interim_text,"is_final":is_final}))
                    except: pass

                if is_final and interim_text:
                    utterance_buffer.append(interim_text)
                    language = getattr(result, "language_code", language) or language
                    if pending_debounce_task:
                        try: pending_debounce_task.cancel()
                        except: pass
                        pending_debounce_task = None
                    text_to_handle = " ".join(utterance_buffer).strip()
                    utterance_buffer.clear()
                    await handle_final_utterance(text_to_handle, last_pcm=None, session_id="default")

    transcript_consumer_task = None

    try:
        loop = asyncio.get_event_loop()
        stop_event.clear()
        stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
        stt_thread.start()

        transcript_consumer_task = asyncio.create_task(process_transcripts_task())
        await ws.send_text(json.dumps({"type":"ready"}))

        while True:
            msg = await ws.receive()
            if msg is None: continue
            if msg.get("type") == "websocket.disconnect": break

            if "text" in msg and msg["text"] is not None:
                try:
                    msgj = json.loads(msg["text"])
                except Exception: continue

                mtype = msgj.get("type")
                if mtype == "start":
                    meta = msgj.get("meta", {}) or {}
                    new_lang = meta.get("language")
                    if new_lang and new_lang != language:
                        language = new_lang
                        with restarting_lock:
                            stop_event.set()
                            try: audio_queue.put_nowait(None)
                            except: pass
                            if stt_thread: stt_thread.join(timeout=1.0)
                            stop_event = threading.Event()
                            stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
                            stt_thread.start()
                            last_restart_ts = time.time()
                    try: await ws.send_text(json.dumps({"type":"ack","message":"started"}))
                    except: pass

                elif mtype == "audio":
                    b64 = msgj.get("payload")
                    if b64:
                        try:
                            pcm = base64.b64decode(b64)
                            silent = is_silence(pcm)
                            if not silent:
                                last_voice_ts = time.time()
                                last_audio_energy_desc = calculate_audio_energy(pcm)
                            push_audio_chunk(pcm)
                            asyncio.create_task(maybe_send_throttle())
                        except: pass

                elif mtype == "stop":
                    try: await ws.send_text(json.dumps({"type":"bye"}))
                    except: pass
                    await ws.close()
                    return

            elif "bytes" in msg and msg["bytes"] is not None:
                pcm = msg["bytes"]
                try:
                    if not is_silence(pcm):
                        last_voice_ts = time.time()
                        last_audio_energy_desc = calculate_audio_energy(pcm)
                except: pass
                push_audio_chunk(pcm)
                asyncio.create_task(maybe_send_throttle())

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
    finally:
        stop_event.set()
        try: audio_queue.put_nowait(None)
        except: pass
        try: transcripts_queue.put_nowait(None)
        except: pass
        if current_tts_task: current_tts_task.cancel()
        if stt_thread: stt_thread.join(timeout=1.0)
        if transcript_consumer_task: transcript_consumer_task.cancel()
        try: await ws.close()
        except: pass
        logger.info("HD WS cleanup complete")
