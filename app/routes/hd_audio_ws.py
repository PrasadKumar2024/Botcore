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
        import redis as _redis_pkg  # optional
        _redis = _redis_pkg.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis session store enabled")
except Exception:
    _redis = None
    logger.info("Redis not available — falling back to in-memory session store")

# in-memory fallback store
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
            # trim to reasonable size
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
    if s >= 0.5:
        rate = str(prosody_rate * 1.06); pitch = "+2st"; volume = "loud"
    elif s >= 0.15:
        rate = str(prosody_rate * 1.03); pitch = "+1st"
    elif s <= -0.4:
        rate = str(prosody_rate * 0.88); pitch = "-2st"; volume = "soft"
    elif s <= -0.15:
        rate = str(prosody_rate * 0.93); pitch = "-1st"; volume = "soft"
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # add short breaks to make TTS more conversational
    esc = esc.replace(", ", ", <break time='120ms'/> ")
    esc = esc.replace(". ", ". <break time='200ms'/> ")
    esc = esc.replace("? ", "? <break time='200ms'/> ")
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

# -------------- Simple acoustic features (lightweight) --------------
def acoustic_sentiment_estimate(pcm16: bytes) -> float:
    """
    Lightweight acoustic heuristic: map RMS to [-1..1] with smoothing.
    This is not a full emotion model — replace with real acoustic model for production.
    """
    try:
        rms = audioop.rms(pcm16, 2)  # 0..32767
        # map typical human rms ranges into -1..+1
        # choose a heuristic scale: quiet <300 -> negative mild, loud >2000 -> positive energetic
        val = (rms - 800) / 2000.0
        return max(-1.0, min(1.0, val))
    except Exception:
        return 0.0

# -------------- improved text sentiment (still fast rules) --------------
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

def combined_sentiment(text: str, pcm16: Optional[bytes] = None) -> float:
    text_s = quick_sentiment_score_text(text or "")
    ac_s = acoustic_sentiment_estimate(pcm16) if pcm16 else 0.0
    # weighted ensemble (text heavier unless acoustic strong)
    return max(-1.0, min(1.0, 0.75 * text_s + 0.25 * ac_s))

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
            msg = str(e).lower()
            if "missing" in msg and "requests" in msg:
                logger.info("Detected SpeechHelpers signature requiring (config, requests). Using fallback call.")
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
            raise

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
    if not intent:
        return
    METRICS["intent_counts"][intent] += 1

def record_metric_sentiment(s: float):
    METRICS["sentiments"].append(s)

def record_metric_confidence(c: float):
    METRICS["confidences"].append(c)

def record_latency(ms: float):
    METRICS["avg_response_ms"].append(ms)

# ----------------- Regex fast-path -----------------
FAST_PATTERNS = [
    (re.compile(r"\b(hours|open|close|opening|closing|timings)\b", re.I), "business_hours"),
    (re.compile(r"\b(phone|contact|call|number)\b", re.I), "contact_number"),
    (re.compile(r"\b(appointment|book|booking|consultation)\b", re.I), "appointment"),
    (re.compile(r"\b(hello|hi|hey|good (morning|evening|afternoon))\b", re.I), "greeting"),
]

FAST_RESPONSES = {
    "business_hours": "Hi — thanks for calling BrightCare. We’re open Monday to Friday, 10 AM to 6 PM. Would you like to book an appointment or get directions?",
    "contact_number": "You can reach BrightCare at +1-800-555-0123. Would you like me to connect you or send this number by SMS?",
    "appointment": "I can help with that. Would you like to book an in-person appointment or a remote consultation?",
    "greeting": "Hello — welcome to BrightCare. How can I help you today?",
}

def fast_intent_match(text: str) -> Optional[Tuple[str, str]]:
    t = text.lower()
    for patt, label in FAST_PATTERNS:
        if patt.search(t):
            return (label, FAST_RESPONSES.get(label, ""))
    return None

# ----------------- Helper: robust JSON parsing from model text -----------------
def extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
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

    CANNED = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hey": "Hey — how can I help?",
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "good morning": "Good morning! How can I assist?",
        "good evening": "Good evening! How can I assist?",
    }

    throttled = False  # server-side flag for sending throttle/resume updates

    def push_audio_chunk(pcm: bytes):
        """Put chunk into audio_queue. If full, drop oldest to accept newest (prefer recent speech)."""
        try:
            audio_queue.put_nowait(pcm)
        except queue.Full:
            try:
                audio_queue.get_nowait()  # drop oldest
            except Exception:
                pass
            try:
                audio_queue.put_nowait(pcm)
                METRICS["drops_oldest"] += 1
                logger.warning("Audio queue full — dropped oldest chunk to accept new")
            except Exception:
                # last resort: drop newest
                METRICS["drops_newest"] += 1
                logger.warning("Audio queue full — dropping newest chunk")

    async def maybe_send_throttle():
        nonlocal throttled
        try:
            q = audio_queue.qsize()
            if q > THROTTLE_QSIZE_HIGH and not throttled:
                throttled = True
                await ws.send_text(json.dumps({"type":"control","action":"throttle"}))
                logger.info("Sent throttle to client (qsize=%d)", q)
            elif q < THROTTLE_QSIZE_LOW and throttled:
                throttled = False
                await ws.send_text(json.dumps({"type":"control","action":"resume"}))
                logger.info("Sent resume to client (qsize=%d)", q)
        except Exception:
            logger.exception("maybe_send_throttle failed")

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
                except Exception:
                    pass
            else:
                try:
                    await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
                except:
                    pass
        except asyncio.CancelledError:
            logger.info("TTS send task cancelled (barge-in)")
            return
        except Exception as e:
            logger.exception("TTS/send failed: %s", e)
            try:
                await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
            except:
                pass
        finally:
            is_bot_speaking = False
            current_tts_task = None

    async def send_tts_and_audio(ai_text: str, language_code: str, sentiment: float = 0.0):
        nonlocal current_tts_task
        if current_tts_task and not current_tts_task.done():
            try:
                current_tts_task.cancel()
            except Exception:
                pass
        current_tts_task = asyncio.create_task(_do_tts_and_send(ai_text, language_code, sentiment))
        try:
            await current_tts_task
        except asyncio.CancelledError:
            return

    # helper to run streaming LLM for spoken output
    def _stream_writer_thread(loop_ref, prompt: str, system_msg: str, token_queue_async: asyncio.Queue, stop_evt: threading.Event):
        try:
            gen = _gemini.generate_stream(prompt=prompt, system_message=system_msg, temperature=0.6)
            for chunk in gen:
                if stop_evt.is_set():
                    break
                try:
                    asyncio.run_coroutine_threadsafe(token_queue_async.put(chunk), loop_ref)
                except Exception:
                    break
        except Exception as e:
            logger.exception("stream_writer error: %s", e)
        finally:
            try:
                asyncio.run_coroutine_threadsafe(token_queue_async.put(None), loop_ref)
            except Exception:
                pass

    async def run_streaming_llm_and_tts(convo_pref: str, user_text: str, language_code: str):
        loop = asyncio.get_running_loop()
        token_q: asyncio.Queue = asyncio.Queue()
        stop_evt = threading.Event()

        system_msg = (
            "You are BrightCare, a warm phone assistant. Stream your textual output progressively. "
            "Keep replies natural, phone-friendly (2-3 short sentences), and offer next steps."
        )
        user_prompt = f"{convo_pref}\n\nUser said: {user_text}\nReply in a natural, helpful tone."

        writer_thread = threading.Thread(target=_stream_writer_thread, args=(loop, user_prompt, system_msg, token_q, stop_evt), daemon=True)
        writer_thread.start()

        full_text = ""
        sentence_buffer = ""
        first_sentence_sent = False
        sentiment_est = 0.0

        try:
            while True:
                token = await token_q.get()
                if token is None:
                    break
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
                    if not s:
                        continue
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
            try:
                writer_thread.join(timeout=1.0)
            except Exception:
                pass

    # ---------------- Combined handler with persona, RAG JSON (synchronous), streaming fallback ----------------
    async def handle_final_utterance(text: str, last_pcm: Optional[bytes] = None, session_id: str = "default"):
        nonlocal conversation, is_bot_speaking, current_stream_stop_event
        METRICS["requests"] += 1
        start_ms = time.time() * 1000

        user_text = text.strip()
        if not user_text:
            return

        ts = time.time()
        conversation.append(("user", user_text, None, None, 0.0, ts))
        append_session_turn(session_id, "user", user_text)

        # 1) fast-path regex
        fast = fast_intent_match(user_text)
        if fast:
            intent_label, reply = fast
            record_metric_intent(intent_label)
            conversation.append(("assistant", reply, intent_label, {}, 0.0, time.time()))
            try:
                await ws.send_text(json.dumps({"type":"ai_text","text":reply,"metadata":{"intent":intent_label}}))
            except:
                pass
            await send_tts_and_audio(reply, language_code=language, sentiment=0.0)
            record_latency(time.time() * 1000 - start_ms)
            return

        # 2) RAG search
        norm_q = normalize_and_expand_query(user_text)
        try:
            rag_results = await pinecone_service.search_similar_chunks(
                client_id=DEFAULT_CLIENT_ID,
                query=norm_q or user_text,
                top_k=6,
                min_score=-1.0
            )
        except Exception as e:
            logger.exception("Pinecone search failed: %s", e)
            rag_results = None

        def rag_confidence(results):
            try:
                scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
                if not scores:
                    return 0.0
                avg = sum(scores) / len(scores)
                conf = max(0.0, min(1.0, (avg + 0.5)))
                return conf
            except Exception:
                return 0.0

        if rag_results:
            convo_lines = []
            for entry in list(conversation)[-6:]:
                role = entry[0]; txt = entry[1]
                prefix = "User:" if role == "user" else "Assistant:"
                convo_lines.append(f"{prefix} {txt}")
            convo_pref = "\n".join(convo_lines)
            context_text = "\n\n".join([f"{r.get('source','')}: {r.get('chunk_text','')}" for r in rag_results[:4]])
            confidence_est = rag_confidence(rag_results)

            # IMPORTANT: structured JSON NLU should be requested in a synchronous safe call to avoid partial JSON parsing problems.
            system_msg = (
                "You are BrightCare, a warm, helpful phone assistant serving BrightCare Mini Health Service. "
                "Use only the CONTEXT to answer factual questions. Produce EXACT JSON only with keys: "
                "{\"intent\":\"\",\"entities\":{...},\"confidence\":<0.0-1.0> ,\"sentiment\":<num -1..1>, "
                "\"fact\":\"<short factual sentence>\", \"spoken\":\"<friendly spoken reply (2-3 short sentences)>\"}. "
                "spoken must use 'we' for BrightCare and offer a next step (book, directions, escalate) when appropriate."
            )
            user_prompt = f"{convo_pref}\n\nCONTEXT:\n{context_text}\n\nUser utterance: {user_text}\nReturn JSON."

            # SYNCHRONOUS safe JSON call (blocking in executor)
            loop = asyncio.get_running_loop()
            try:
                partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=system_msg, temperature=0.15, max_tokens=300)
                fut = loop.run_in_executor(executor, partial)
                resp_text = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
                resp_text = (resp_text or "").strip()
            except asyncio.TimeoutError:
                logger.warning("RAG NLU timed out")
                resp_text = ""
            except Exception as e:
                logger.exception("RAG LLM JSON call failed: %s", e)
                resp_text = ""

            parsed = extract_json_from_text(resp_text)
            if parsed:
                intent = parsed.get("intent","") or ""
                entities = parsed.get("entities",{}) or {}
                sentiment = float(parsed.get("sentiment",0.0) or 0.0)
                fact = (parsed.get("fact","") or "").strip()
                spoken = (parsed.get("spoken","") or "").strip()
                confidence = float(parsed.get("confidence", confidence_est) or confidence_est)
            else:
                # fallback: generate textual answer and wrap
                fact = ""
                try:
                    raw = await get_ai_text_response(user_text, language_code=language, conversation_history=[(r[0],r[1]) for r in list(conversation)])
                except Exception:
                    raw = "Sorry, I couldn't fetch that information right now."
                spoken = f"Hi — thanks for asking. {raw} Would you like me to help with anything else?"
                intent = ""
                entities = {}
                sentiment = combined_sentiment(spoken, last_pcm)
                confidence = confidence_est

            # metrics & storage
            record_metric_intent(intent or "rag_response")
            record_metric_sentiment(sentiment)
            record_metric_confidence(confidence)
            conversation.append(("assistant", spoken or fact or "Sorry, I couldn't find that.", intent, entities, sentiment, time.time()))
            append_session_turn(session_id="default", role="assistant", text=spoken or fact or "")

            try:
                await ws.send_text(json.dumps({
                    "type":"ai_text",
                    "text": spoken or fact,
                    "metadata": {"intent": intent, "entities": entities, "confidence": confidence, "sentiment": sentiment, "fact": fact}
                }))
            except:
                pass

            tts_text = spoken if spoken else fact
            await send_tts_and_audio(tts_text, language_code=language, sentiment=sentiment)
            record_latency(time.time()*1000 - start_ms)
            return

        # 3) No RAG hits -> streaming LLM fallback if supported
        convo_lines = []
        for entry in list(conversation)[-6:]:
            role = entry[0]; txt = entry[1]
            prefix = "User:" if role == "user" else "Assistant:"
            convo_lines.append(f"{prefix} {txt}")
        convo_pref = "\n".join(convo_lines)

        if hasattr(_gemini, "generate_stream"):
            try:
                if current_stream_stop_event:
                    current_stream_stop_event.set()
                current_stream_stop_event = threading.Event()
                final_text, sentiment_est = await run_streaming_llm_and_tts(convo_pref, user_text, language_code=language)
                intent_label = "chat_fallback"
                record_metric_intent(intent_label)
                record_metric_sentiment(sentiment_est)
                record_metric_confidence(0.0)
                conversation.append(("assistant", final_text, intent_label, {}, sentiment_est, time.time()))
                append_session_turn("default", "assistant", final_text)
                try:
                    await ws.send_text(json.dumps({"type":"ai_text","text":final_text,"metadata":{"intent":intent_label,"sentiment":sentiment_est}}))
                except:
                    pass
                record_latency(time.time()*1000 - start_ms)
                return
            except Exception as e:
                logger.exception("Streaming fallback error: %s", e)

        # synchronous fallback
        try:
            ai_text = await get_ai_text_response(user_text, language_code=language, conversation_history=[(r[0],r[1]) for r in list(conversation)])
        except Exception as e:
            logger.exception("Fallback LLM error: %s", e)
            ai_text = "Sorry, I'm having trouble answering right now."

        sentiment = combined_sentiment(ai_text, last_pcm)
        record_metric_sentiment(sentiment)
        record_metric_intent("chat_fallback_sync")
        record_metric_confidence(0.0)
        conversation.append(("assistant", ai_text, "chat_fallback_sync", {}, sentiment, time.time()))
        append_session_turn("default", "assistant", ai_text)
        try:
            await ws.send_text(json.dumps({"type":"ai_text","text":ai_text,"metadata":{"sentiment":sentiment}}))
        except:
            pass
        await send_tts_and_audio(ai_text, language_code=language, sentiment=sentiment)
        record_latency(time.time()*1000 - start_ms)

    # Debounce handler (kept for non-final cases)
    async def debounce_and_handle():
        nonlocal pending_debounce_task, utterance_buffer, last_voice_ts
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
            if (time.time() - last_voice_ts) < (DEBOUNCE_SECONDS - 0.05):
                return
            text = " ".join(utterance_buffer).strip()
            utterance_buffer.clear()
            if not text:
                return
            await handle_final_utterance(text, last_pcm=None, session_id="default")
        finally:
            pending_debounce_task = None

    # Transcript consumer
    async def process_transcripts_task():
        nonlocal language, utterance_buffer, pending_debounce_task, is_bot_speaking, current_tts_task, current_stream_stop_event
        while True:
            resp = await transcripts_queue.get()
            if resp is None:
                logger.info("Transcript consumer received sentinel; exiting")
                break
            for result in resp.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                interim_text = alt.transcript.strip()
                is_final = getattr(result, "is_final", False)

                # Barge-in: user speaks while assistant is speaking
                if interim_text and is_bot_speaking:
                    try:
                        await ws.send_text(json.dumps({"type":"control","action":"stop_playback"}))
                    except Exception:
                        pass
                    if current_tts_task and not current_tts_task.done():
                        try:
                            current_tts_task.cancel()
                        except Exception:
                            pass
                    if current_stream_stop_event:
                        try:
                            current_stream_stop_event.set()
                        except Exception:
                            pass
                    is_bot_speaking = False

                if interim_text:
                    try:
                        await ws.send_text(json.dumps({"type":"transcript","text":interim_text,"is_final":is_final}))
                    except Exception:
                        pass

                if is_final and interim_text:
                    # immediate action on is_final: handle now
                    utterance_buffer.append(interim_text)
                    language = getattr(result, "language_code", language) or language
                    if pending_debounce_task:
                        try:
                            pending_debounce_task.cancel()
                        except Exception:
                            pass
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
            # Use ws.receive() to accept both text and binary frames
            msg = await ws.receive()
            if msg is None:
                continue
            if msg.get("type") == "websocket.disconnect":
                logger.info("Client disconnected")
                break

            if "text" in msg and msg["text"] is not None:
                data_text = msg["text"]
                try:
                    msgj = json.loads(data_text)
                except Exception:
                    try:
                        await ws.send_text(json.dumps({"type":"error","error":"invalid_json"}))
                    except:
                        pass
                    continue

                mtype = msgj.get("type")
                if mtype == "start":
                    meta = msgj.get("meta", {}) or {}
                    new_lang = meta.get("language")
                    if new_lang and new_lang != language:
                        now_ts = time.time()
                        if now_ts - last_restart_ts < MIN_RESTART_INTERVAL:
                            logger.info("Language restart suppressed by backoff (last_restart_ts=%s)", last_restart_ts)
                        else:
                            logger.info("Language change requested: %s -> %s", language, new_lang)
                            language = new_lang
                            with restarting_lock:
                                try:
                                    stop_event.set()
                                    try:
                                        audio_queue.put_nowait(None)
                                    except Exception:
                                        pass
                                    if stt_thread and stt_thread.is_alive():
                                        stt_thread.join(timeout=2.0)
                                except Exception as e:
                                    logger.exception("Error stopping STT thread for language restart: %s", e)
                                stop_event = threading.Event()
                                stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
                                stt_thread.start()
                                last_restart_ts = time.time()
                                logger.info("Restarted STT worker with language=%s", language)
                    try:
                        await ws.send_text(json.dumps({"type":"ack","message":"started"}))
                    except:
                        pass

                elif mtype == "audio":
                    # backward-compatible base64 JSON path (support legacy clients)
                    b64 = msgj.get("payload")
                    if not b64:
                        continue
                    try:
                        pcm = base64.b64decode(b64)
                    except Exception:
                        try:
                            await ws.send_text(json.dumps({"type":"error","error":"bad_audio_b64"}))
                        except:
                            pass
                        continue

                    try:
                        silent = is_silence(pcm)
                    except Exception:
                        silent = False

                    if not silent:
                        last_voice_ts = time.time()

                    # push with drop-oldest policy
                    push_audio_chunk(pcm)
                    # inform client about throttle/resume if needed
                    asyncio.create_task(maybe_send_throttle())
                    continue

                elif mtype == "stop":
                    logger.info("Client stop received; flushing and closing")
                    try:
                        stop_event.set()
                        audio_queue.put_nowait(None)
                    except Exception:
                        pass
                    await transcripts_queue.put(None)
                    try:
                        await ws.send_text(json.dumps({"type":"bye"}))
                    except:
                        pass
                    await ws.close()
                    return

                elif mtype == "metrics":
                    try:
                        await ws.send_text(json.dumps({"type":"metrics","metrics": {
                            "requests": METRICS["requests"],
                            "intent_counts": dict(METRICS["intent_counts"]),
                            "avg_response_ms": (sum(METRICS["avg_response_ms"])/len(METRICS["avg_response_ms"])) if METRICS["avg_response_ms"] else None,
                            "sentiment_samples": len(METRICS["sentiments"]),
                            "confidence_samples": len(METRICS["confidences"]),
                            "drops_oldest": METRICS["drops_oldest"],
                            "drops_newest": METRICS["drops_newest"],
                        }}))
                    except Exception:
                        pass
                else:
                    try:
                        await ws.send_text(json.dumps({"type":"error","error":"unknown_type"}))
                    except:
                        pass

            elif "bytes" in msg and msg["bytes"] is not None:
                # binary frame received -> raw Int16 PCM from client
                pcm = msg["bytes"]
                try:
                    silent = is_silence(pcm)
                except Exception:
                    silent = False

                if not silent:
                    last_voice_ts = time.time()

                push_audio_chunk(pcm)
                asyncio.create_task(maybe_send_throttle())

            else:
                # unexpected receive type
                continue

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
    finally:
        try:
            stop_event.set()
            audio_queue.put_nowait(None)
        except Exception:
            pass
        try:
            transcripts_queue.put_nowait(None)
        except Exception:
            pass
        if current_tts_task and not current_tts_task.done():
            try:
                current_tts_task.cancel()
            except Exception:
                pass
        if current_stream_stop_event:
            try:
                current_stream_stop_event.set()
            except Exception:
                pass
        if stt_thread:
            stt_thread.join(timeout=2.0)
        if transcript_consumer_task:
            try:
                transcript_consumer_task.cancel()
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("HD WS cleanup complete")
