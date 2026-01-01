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
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Use explicit v1 clients to match common SDK installs
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions

# local services - keep using your modules
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
router = APIRouter()

# ================== Configuration (env-driven) ==================
EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "8"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "4"))
STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "10.0"))
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "12.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "20.0"))

STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))   # matches your frontend
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "0.32"))  # ~320ms
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "10"))
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)

DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", os.getenv("DEFAULT_KB_CLIENT_ID", "default"))
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BrightCare")
ASSISTANT_PERSONA = os.getenv("ASSISTANT_PERSONA", "friendly professional assistant")

BYTES_PER_SEC = STT_SAMPLE_RATE * 2
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

STREAM_SENTENCE_CHAR_LIMIT = int(os.getenv("HD_WS_SENTENCE_CHAR_LIMIT", "240"))
TTS_WORKER_IDLE_TIMEOUT = float(os.getenv("HD_WS_TTS_WORKER_IDLE", "60.0"))

executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
global_tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)

DEBOUNCE_SECONDS = float(os.getenv("HD_WS_DEBOUNCE_S", "0.5"))
VAD_THRESHOLD = int(os.getenv("HD_WS_VAD_THRESHOLD", "300"))
MIN_RESTART_INTERVAL = float(os.getenv("HD_WS_MIN_RESTART_INTERVAL", "2.0"))

# ================== Google clients (from JSON env) ==================
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = tts.TextToSpeechClient(credentials=_creds)

# ================== LLM Service ==================
_gemini = GeminiService()

# ================== Voice configuration & utils ==================
VOICE_MAP = {
    "en-IN": {"name": "en-IN-Neural2-C", "gender": "MALE"},
    "en-US": {"name": "en-US-Neural2-A", "gender": "FEMALE"},
    "hi-IN": {"name": "hi-IN-Neural2-A", "gender": "FEMALE"},
}
DEFAULT_VOICE = VOICE_MAP.get("en-IN")

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
    # dynamic values as strings for SSML
    if s >= 0.6:
        rate = f"{prosody_rate * 1.08:.2f}"
        pitch = "+3st"
        volume = "+6dB"
    elif s >= 0.3:
        rate = f"{prosody_rate * 1.04:.2f}"
        pitch = "+2st"
        volume = "+3dB"
    elif s >= 0.1:
        rate = f"{prosody_rate * 1.01:.2f}"
        pitch = "+1st"
        volume = "+1dB"
    elif s <= -0.5:
        rate = f"{prosody_rate * 0.85:.2f}"
        pitch = "-3st"
        volume = "-3dB"
    elif s <= -0.25:
        rate = f"{prosody_rate * 0.90:.2f}"
        pitch = "-2st"
        volume = "-2dB"
    elif s <= -0.1:
        rate = f"{prosody_rate * 0.93:.2f}"
        pitch = "-1st"
        volume = "-1dB"
    else:
        rate = f"{prosody_rate:.2f}"
        pitch = "0st"
        volume = "0dB"

    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc = esc.replace(", ", ", <break time='120ms'/> ")
    esc = esc.replace(". ", ". <break time='220ms'/> ")
    esc = esc.replace("? ", "? <break time='220ms'/> ")
    esc = esc.replace("! ", "! <break time='220ms'/> ")
    esc = esc.replace(": ", ": <break time='160ms'/> ")
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

# ================== TTS Synthesis (sync wrapper) ==================
def _sync_tts_linear16(ssml: str, language_code: str, voice_name: str, gender: Optional[str], sample_rate_hz: int = 24000):
    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz
    )
    synthesis_input = tts.SynthesisInput(ssml=ssml)
    return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

async def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", sample_rate_hz: int = 24000, sentiment: float = 0.0) -> Optional[bytes]:
    ssml = ssml_for_text(text, sentiment=sentiment, prosody_rate=0.95)
    lang_code, voice_name, gender = get_best_voice(language_code)
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(global_tts_semaphore.acquire(), timeout=5.0)
    except Exception:
        logger.warning("TTS queue busy")
        return None
    try:
        fut = loop.run_in_executor(executor, functools.partial(_sync_tts_linear16, ssml, lang_code, voice_name, gender, sample_rate_hz))
        resp = await asyncio.wait_for(fut, timeout=TTS_TIMEOUT)
        return getattr(resp, "audio_content", None)
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

# ================== Query normalization / tiny RAG helpers ==================
_CONTRACTIONS = {
    r"\bwhat's\b": "what is",
    r"\bwhats\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bwhen's\b": "when is",
    r"\bhow's\b": "how is",
    r"\bdon't\b": "do not",
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
        "phone number": "contact number telephone",
        "appointment": "appointment booking schedule",
        "doctor": "doctor physician specialist",
        "payment": "payment methods accepted cash card upi",
        "location": "address directions",
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

def calculate_rag_confidence(results: List[Dict]) -> float:
    if not results:
        return 0.0
    try:
        scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
        if not scores:
            return 0.0
        avg_score = sum(scores) / len(scores)
        confidence = (avg_score + 1.0) / 2.0
        confidence = max(0.0, min(1.0, confidence))
        if len(scores) >= 3 and avg_score > 0.3:
            confidence = min(1.0, confidence * 1.2)
        return confidence
    except Exception:
        return 0.0

# ================== Sentiment heuristics ==================
_POS_WORDS = {"good", "great", "happy", "thanks", "helpful", "amazing", "satisfied"}
_NEG_WORDS = {"bad", "sad", "angry", "problem", "issue", "disappointed", "hate"}
_INTENSITY_MODIFIERS = {"very": 1.5, "really": 1.5, "extremely": 2.0, "slightly": 0.5}

def advanced_sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    tl = text.lower()
    words = tl.split()
    pos_score = 0.0
    neg_score = 0.0
    for i, word in enumerate(words):
        intensity = 1.0
        if i > 0 and words[i-1] in _INTENSITY_MODIFIERS:
            intensity = _INTENSITY_MODIFIERS[words[i-1]]
        if any(pos in word for pos in _POS_WORDS):
            pos_score += intensity
        if any(neg in word for neg in _NEG_WORDS):
            neg_score += intensity
    negation_words = {"not", "no", "never", "dont", "don't", "cant", "can't"}
    for neg in negation_words:
        if neg in tl:
            pos_score, neg_score = neg_score * 0.8, pos_score * 0.8
            break
    if pos_score == 0 and neg_score == 0:
        return 0.0
    total = pos_score + neg_score
    sentiment = (pos_score - neg_score) / total
    return max(-1.0, min(1.0, sentiment))

# ================== STT worker (thread) - uses event loop for run_coroutine_threadsafe ==================
def grpc_stt_worker(loop, audio_queue: queue.Queue, transcripts_queue: asyncio.Queue, stop_event: threading.Event, language_code: str):
    """
    Streaming recognize thread.
    It yields an initial StreamingRecognizeRequest(streaming_config=...) then audio chunks.
    Puts protobuf responses into transcripts_queue via asyncio.run_coroutine_threadsafe.
    """
    def gen_requests_with_config():
        cfg = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="latest_long",
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
                logger.exception("Error in audio generator: %s", e)
                break

    try:
        logger.info("Starting STT worker (language=%s)", language_code)
        responses = _speech_client.streaming_recognize(requests=gen_requests_with_config())
        for response in responses:
            if stop_event.is_set():
                break
            # push the whole response object for the async consumer to parse (keeps original design)
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(response), loop)
    except Exception:
        logger.exception("STT worker error")
    finally:
        logger.info("STT worker exiting (language=%s)", language_code)

# ================== Contextual response generation (Gemini + RAG) ==================
async def classify_query_intent(text: str, rag_available: bool) -> Dict[str, Any]:
    if not text or len(text.strip()) < 2:
        return {"type": "conversational", "confidence": 0.5}
    text_lower = text.lower().strip()
    greeting_patterns = [r'\b(hi|hello|hey|good morning|good evening|good afternoon)\b', r'\b(how are you|whats up|what\'s up)\b']
    factual_patterns = [r'\b(what|when|where|which|who|how much|how many)\b', r'\b(tell me about|explain|describe|information about)\b', r'\b(hours|timing|price|cost|location|address|phone|contact)\b', r'\b(service|appointment|booking|consultation|doctor)\b']
    for p in greeting_patterns:
        if re.search(p, text_lower):
            return {"type": "greeting", "confidence": 0.9}
    for p in factual_patterns:
        if re.search(p, text_lower):
            return {"type": "factual", "confidence": 0.85}
    return {"type": "conversational", "confidence": 0.6}

async def generate_contextual_response(user_text: str, language_code: str, conversation_history: deque, rag_results: Optional[List[Dict]] = None) -> Tuple[str, float, Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    convo_context = []
    for entry in list(conversation_history)[-6:]:
        role = entry[0]
        txt = entry[1]
        prefix = "User:" if role == "user" else "Assistant:"
        convo_context.append(f"{prefix} {txt}")
    convo_prefix = "\n".join(convo_context) + "\n\n" if convo_context else ""
    intent_info = await classify_query_intent(user_text, rag_results is not None and len(rag_results) > 0)

    if intent_info["type"] == "greeting":
        system_msg = f"You are {BUSINESS_NAME}'s friendly voice assistant. Keep it warm and brief."
        prompt = f"{convo_prefix}User: {user_text}\n\nRespond warmly as {BUSINESS_NAME}'s assistant:"
        partial = functools.partial(_gemini.generate_response, prompt=prompt, system_message=system_msg, temperature=0.8, max_tokens=120)
        try:
            fut = loop.run_in_executor(executor, partial)
            response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            sentiment = advanced_sentiment_score(response)
            return (response.strip(), sentiment, {"intent": "greeting", "confidence": intent_info["confidence"], "response_type": "conversational"})
        except Exception:
            logger.exception("Greeting generation failed")
            return (f"Hello! Welcome to {BUSINESS_NAME}. How can I help you today?", 0.3, {"intent": "greeting", "error": True})

    if intent_info["type"] == "factual" and rag_results:
        context_text = "\n\n".join([f"Source: {r.get('source','doc')}\n{r.get('chunk_text','')}" for r in rag_results[:4]])
        confidence = calculate_rag_confidence(rag_results)
        system_msg = f"You are {BUSINESS_NAME}'s assistant. Use ONLY the provided context to answer."
        prompt = f"{convo_prefix}CONTEXT:\n{context_text}\n\nUser Question: {user_text}\n\nAnswer concisely:"
        partial = functools.partial(_gemini.generate_response, prompt=prompt, system_message=system_msg, temperature=0.3, max_tokens=250)
        try:
            fut = loop.run_in_executor(executor, partial)
            response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            sentiment = advanced_sentiment_score(response)
            metadata = {"intent": "factual", "confidence": confidence, "response_type": "rag", "sources_used": len(rag_results)}
            return (response.strip(), sentiment, metadata)
        except Exception:
            logger.exception("RAG response generation failed")
            return ("I apologize, I'm having trouble accessing that information right now. Can I help with something else?", -0.2, {"intent": "factual", "error": True})

    # default conversational
    system_msg = f"You are {BUSINESS_NAME}'s friendly voice assistant. Short helpful answers."
    prompt = f"{convo_prefix}User: {user_text}\n\nRespond naturally:"
    partial = functools.partial(_gemini.generate_response, prompt=prompt, system_message=system_msg, temperature=0.75, max_tokens=200)
    try:
        fut = loop.run_in_executor(executor, partial)
        response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
        sentiment = advanced_sentiment_score(response)
        return (response.strip(), sentiment, {"intent": "conversational", "confidence": intent_info["confidence"], "response_type": "conversational"})
    except Exception:
        logger.exception("Conversational response failed")
        return ("I'm happy to chat, but I'm having some technical difficulties. Could you try asking me again?", 0.0, {"intent": "conversational", "error": True})

# ================== Serial TTS worker (async) ==================
async def _tts_worker_loop(ws: WebSocket, queue: asyncio.Queue):
    current_task: Optional[asyncio.Task] = None
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=TTS_WORKER_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                break
            if item is None:
                break
            text = item.get("text", "")
            language = item.get("language", "en-IN")
            sentiment = item.get("sentiment", 0.0)

            async def _do_one():
                pcm = await synthesize_text_to_pcm(text, language_code=language, sample_rate_hz=24000, sentiment=sentiment)
                if not pcm:
                    try:
                        await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))
                    except Exception:
                        pass
                    return
                wav_bytes = make_wav_from_pcm16(pcm, sample_rate=24000)
                b64wav = base64.b64encode(wav_bytes).decode("ascii")
                try:
                    await ws.send_text(json.dumps({"type": "audio", "audio": b64wav, "metadata": {"sentiment": sentiment, "length": len(text)}}))
                except Exception:
                    pass

            current_task = asyncio.create_task(_do_one())
            try:
                await current_task
            except asyncio.CancelledError:
                if current_task and not current_task.done():
                    try:
                        current_task.cancel()
                    except Exception:
                        pass
                raise
            finally:
                current_task = None
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("TTS worker loop crashed")
    finally:
        # drain
        try:
            while not queue.empty():
                _ = queue.get_nowait()
        except Exception:
            pass

# ================== WebSocket handler (main) ==================
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

    # Queues/state
    audio_queue = queue.Queue(maxsize=400)
    transcripts_queue = asyncio.Queue()
    stop_event = threading.Event()

    language = "en-IN"
    stt_thread = None

    conversation: deque = deque(maxlen=20)
    utterance_buffer: List[str] = []
    pending_debounce_task: Optional[asyncio.Task] = None

    last_voice_ts = time.time()
    restarting_lock = threading.Lock()
    last_restart_ts = 0.0

    is_bot_speaking = False
    current_tts_task: Optional[asyncio.Task] = None
    current_stream_stop_event: Optional[threading.Event] = None

    session_start = time.time()
    session_id = f"session_{int(session_start)}"

    tts_queue: asyncio.Queue = asyncio.Queue()
    _tts_worker_task: Optional[asyncio.Task] = asyncio.create_task(_tts_worker_loop(ws, tts_queue))

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
                    await ws.send_text(json.dumps({"type": "audio", "audio": b64wav, "metadata": {"sentiment": sentiment, "length": len(ai_text)}}))
                except Exception:
                    pass
            else:
                try:
                    await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("TTS generation failed")
        finally:
            is_bot_speaking = False
            current_tts_task = None

    async def send_tts_and_audio(ai_text: str, language_code: str, sentiment: float = 0.0):
        nonlocal current_tts_task
        if current_tts_task and not current_tts_task.done():
            try:
                current_tts_task.cancel()
                await asyncio.sleep(0.05)
            except Exception:
                pass
        current_tts_task = asyncio.create_task(_do_tts_and_send(ai_text, language_code, sentiment))
        try:
            await current_tts_task
        except asyncio.CancelledError:
            logger.info("TTS task cancelled")

    async def handle_final_utterance(text: str):
        nonlocal conversation, is_bot_speaking, current_stream_stop_event
        METRICS = globals().setdefault("METRICS", {"requests": 0})
        METRICS["requests"] = METRICS.get("requests", 0) + 1
        start_ms = time.time() * 1000
        user_text = text.strip()
        if not user_text:
            return
        logger.info("Processing utterance: %s", user_text)
        ts = time.time()
        conversation.append(("user", user_text, None, None, 0.0, ts))
        try:
            # RAG
            norm_q = normalize_and_expand_query(user_text)
            rag_results = None
            try:
                rag_results = await pinecone_service.search_similar_chunks(client_id=DEFAULT_CLIENT_ID, query=norm_q or user_text, top_k=6, min_score=-1.0)
            except Exception:
                logger.exception("Pinecone search failed")
                rag_results = None

            intent_info = await classify_query_intent(user_text, rag_results is not None and len(rag_results) > 0)
            response_text, sentiment, metadata = await generate_contextual_response(user_text, language, conversation, rag_results)
            # metrics
            try:
                # store last values if available
                pass
            except Exception:
                pass
            conversation.append(("assistant", response_text, metadata.get("intent"), metadata.get("entities", {}), sentiment, time.time()))
            try:
                await ws.send_text(json.dumps({"type": "ai_text", "text": response_text, "metadata": {"intent": metadata.get("intent"), "confidence": metadata.get("confidence", 0.0), "sentiment": sentiment, "response_type": metadata.get("response_type"), "session_id": session_id}}))
            except Exception:
                pass
            # produce TTS via queue
            await tts_queue.put({"text": response_text, "language": language, "sentiment": sentiment})
            latency_ms = time.time() * 1000 - start_ms
            logger.info("Total response time: %.0fms", latency_ms)
        except Exception:
            logger.exception("Error in handle_final_utterance")
            try:
                err_msg = "I apologize, but I encountered an issue processing your request. Could you please try again?"
                await ws.send_text(json.dumps({"type": "ai_text", "text": err_msg, "metadata": {"error": True}}))
                await tts_queue.put({"text": err_msg, "language": language, "sentiment": 0.0})
            except Exception:
                pass

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
            await handle_final_utterance(text)
        finally:
            pending_debounce_task = None

    async def process_transcripts_task():
        nonlocal language, utterance_buffer, pending_debounce_task, is_bot_speaking, current_tts_task, current_stream_stop_event, _tts_worker_task, tts_queue
        while True:
            resp = await transcripts_queue.get()
            if resp is None:
                logger.info("Transcript consumer received sentinel; exiting")
                break
            # resp is a protobuf StreamingRecognizeResponse
            for result in resp.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                interim_text = alt.transcript.strip()
                is_final = getattr(result, "is_final", False)
                if interim_text and is_bot_speaking:
                    try:
                        await ws.send_text(json.dumps({"type": "control", "action": "stop_playback"}))
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
                    # flush tts queue
                    try:
                        while not tts_queue.empty():
                            try:
                                _ = tts_queue.get_nowait()
                            except Exception:
                                break
                    except Exception:
                        pass
                    if _tts_worker_task and not _tts_worker_task.done():
                        try:
                            _tts_worker_task.cancel()
                        except Exception:
                            pass
                    _tts_worker_task = asyncio.create_task(_tts_worker_loop(ws, tts_queue))
                    is_bot_speaking = False

                # send interim transcript for UI
                if interim_text:
                    try:
                        await ws.send_text(json.dumps({"type": "transcript", "text": interim_text, "is_final": is_final}))
                    except Exception:
                        pass

                if is_final and interim_text:
                    utterance_buffer.append(interim_text)
                    detected_lang = getattr(result, "language_code", None)
                    if detected_lang:
                        language = detected_lang
                    if not pending_debounce_task or pending_debounce_task.done():
                        pending_debounce_task = asyncio.create_task(debounce_and_handle())

    # START STT thread and consumer
    try:
        loop = asyncio.get_event_loop()
        stop_event.clear()
        stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
        stt_thread.start()
        transcript_consumer_task = asyncio.create_task(process_transcripts_task())

        await ws.send_text(json.dumps({"type": "ready", "session_id": session_id, "language": language}))
        logger.info("Session %s ready", session_id)

        # main message loop
        while True:
            msg = await ws.receive()
            if msg is None:
                continue
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                logger.info("Client websocket disconnected")
                break

            # text frame JSON
            if "text" in msg and msg["text"] is not None:
                data_text = msg["text"]
                try:
                    ctrl = json.loads(data_text)
                except Exception:
                    try:
                        await ws.send_text(json.dumps({"type": "error", "error": "invalid_json"}))
                    except Exception:
                        pass
                    continue
                mtype = ctrl.get("type")
                if mtype == "start":
                    meta = ctrl.get("meta", {}) or {}
                    new_lang = meta.get("language")
                    if new_lang and new_lang != language:
                        now_ts = time.time()
                        if now_ts - last_restart_ts < MIN_RESTART_INTERVAL:
                            logger.info("Language restart suppressed by backoff")
                        else:
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
                                except Exception:
                                    logger.exception("Error stopping STT thread")
                                stop_event = threading.Event()
                                stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
                                stt_thread.start()
                                last_restart_ts = time.time()
                                logger.info("Restarted STT worker with language=%s", language)
                    try:
                        await ws.send_text(json.dumps({"type": "ack", "message": "started"}))
                    except Exception:
                        pass

                elif mtype == "audio":
                    b64 = ctrl.get("payload")
                    if not b64:
                        continue
                    try:
                        pcm = base64.b64decode(b64)
                    except Exception:
                        try:
                            await ws.send_text(json.dumps({"type": "error", "error": "bad_audio_b64"}))
                        except Exception:
                            pass
                        continue
                    try:
                        silent = is_silence(pcm)
                    except Exception:
                        silent = False
                    if not silent:
                        last_voice_ts = time.time()
                    if audio_queue.qsize() > 350:
                        logger.warning("Audio queue large (%s), dropping input", audio_queue.qsize())
                        continue
                    try:
                        audio_queue.put_nowait(pcm)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")
                        continue

                elif mtype == "stop":
                    logger.info("Client requested stop")
                    try:
                        stop_event.set()
                        audio_queue.put_nowait(None)
                    except Exception:
                        pass
                    await transcripts_queue.put(None)
                    try:
                        await ws.send_text(json.dumps({"type": "bye"}))
                    except Exception:
                        pass
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return

                elif mtype == "metrics":
                    try:
                        await ws.send_text(json.dumps({"type": "metrics", "session_id": session_id, "metrics": {}}))
                    except Exception:
                        pass
                else:
                    try:
                        await ws.send_text(json.dumps({"type": "error", "error": "unknown_type"}))
                    except Exception:
                        pass

            # binary audio frames
            elif "bytes" in msg and msg["bytes"] is not None:
                pcm = msg["bytes"]
                try:
                    silent = is_silence(pcm)
                except Exception:
                    silent = False
                if not silent:
                    last_voice_ts = time.time()
                if audio_queue.qsize() > 350:
                    logger.warning("Audio queue large (%s), dropping input", audio_queue.qsize())
                    continue
                try:
                    audio_queue.put_nowait(pcm)
                except queue.Full:
                    logger.warning("Audio queue full, dropping chunk")
                    continue

            else:
                logger.debug("Received unexpected websocket message: %s", msg)
                continue

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception:
        logger.exception("WS loop error")
    finally:
        logger.info("Starting WS cleanup")
        try:
            stop_event.set()
            audio_queue.put_nowait(None)
        except Exception:
            pass
        try:
            transcripts_queue.put_nowait(None)
        except Exception:
            pass
        try:
            await tts_queue.put(None)
        except Exception:
            pass
        try:
            if _tts_worker_task and not _tts_worker_task.done():
                _tts_worker_task.cancel()
        except Exception:
            pass
        try:
            if current_tts_task and not current_tts_task.done():
                current_tts_task.cancel()
        except Exception:
            pass
        try:
            if current_stream_stop_event:
                current_stream_stop_event.set()
        except Exception:
            pass
        try:
            if stt_thread:
                stt_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if transcript_consumer_task:
                transcript_consumer_task.cancel()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("HD WS cleanup complete. Session duration: %.1fs", time.time() - session_start)
