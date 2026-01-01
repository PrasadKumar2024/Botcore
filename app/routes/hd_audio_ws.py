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
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions

# local services
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)  # Use module name for logging
router = APIRouter()

# ================== Configuration ==================

EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "8"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "4"))
STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "10.0"))
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "12.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "15.0"))

STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "0.35"))
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "10"))
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)

DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", os.getenv("DEFAULT_KB_CLIENT_ID", "default"))
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BrightCare")

BYTES_PER_SEC = STT_SAMPLE_RATE * 2
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

# TTS/streaming tuning
STREAM_SENTENCE_CHAR_LIMIT = int(os.getenv("HD_WS_SENTENCE_CHAR_LIMIT", "240"))
TTS_WORKER_IDLE_TIMEOUT = float(os.getenv("HD_WS_TTS_WORKER_IDLE", "60.0"))

executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
global_tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)

# VAD & debounce tuning
DEBOUNCE_SECONDS = float(os.getenv("HD_WS_DEBOUNCE_S", "0.5"))
VAD_THRESHOLD = int(os.getenv("HD_WS_VAD_THRESHOLD", "300"))
MIN_RESTART_INTERVAL = float(os.getenv("HD_WS_MIN_RESTART_INTERVAL", "2.0"))

# ================== Google Clients ==================

GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = tts.TextToSpeechClient(credentials=_creds)

# ================== LLM Service ==================

_gemini = GeminiService()

# ================== Voice Configuration ==================

# Only English voices for now (Indian and US English)
VOICE_MAP = {
    "en-IN": {"name": "en-IN-Neural2-C", "gender": "MALE"},
    "en-US": {"name": "en-US-Neural2-A", "gender": "FEMALE"},
}
DEFAULT_VOICE = VOICE_MAP["en-IN"]

def get_best_voice(language_code: Optional[str]):
    """Select best voice name given a language code (English variants only)."""
    if not language_code:
        return ("en-IN", DEFAULT_VOICE["name"], DEFAULT_VOICE.get("gender"))
    if language_code in VOICE_MAP:
        v = VOICE_MAP[language_code]
        return (language_code, v["name"], v.get("gender"))
    base = language_code.split("-")[0]
    # Only support English locales
    if base == "en":
        v = VOICE_MAP.get("en-IN", DEFAULT_VOICE)
        return ("en-IN", v["name"], v.get("gender"))
    # Default to Indian English if unknown
    return ("en-IN", DEFAULT_VOICE["name"], DEFAULT_VOICE.get("gender"))

def ssml_for_text(text: str, sentiment: float = 0.0, prosody_rate: float = 0.95) -> str:
    """Enhanced SSML with emotional prosody. Adjust pitch/rate based on sentiment."""
    s = max(-1.0, min(1.0, sentiment or 0.0))
    # Set prosody parameters by sentiment
    if s >= 0.6:
        rate = str(prosody_rate * 1.08); pitch = "+3st"; volume = "loud"
    elif s >= 0.3:
        rate = str(prosody_rate * 1.04); pitch = "+2st"; volume = "medium"
    elif s >= 0.1:
        rate = str(prosody_rate * 1.01); pitch = "+1st"; volume = "medium"
    elif s <= -0.5:
        rate = str(prosody_rate * 0.85); pitch = "-3st"; volume = "soft"
    elif s <= -0.25:
        rate = str(prosody_rate * 0.90); pitch = "-2st"; volume = "soft"
    elif s <= -0.1:
        rate = str(prosody_rate * 0.93); pitch = "-1st"; volume = "soft"
    else:
        rate = str(prosody_rate); pitch = "0st"; volume = "medium"

    # Escape special SSML characters
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Add pauses after punctuation for natural speech
    esc = esc.replace(", ", ", <break time='140ms'/> ")
    esc = esc.replace(". ", ". <break time='250ms'/> ")
    esc = esc.replace("? ", "? <break time='250ms'/> ")
    esc = esc.replace("! ", "! <break time='250ms'/> ")
    esc = esc.replace(": ", ": <break time='180ms'/> ")

    return f"<speak><prosody rate='{rate}' pitch='{pitch}' volume='{volume}'>{esc}</prosody></speak>"

def make_wav_from_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap PCM bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def is_silence(pcm16: bytes, threshold: int = VAD_THRESHOLD) -> bool:
    """Detect if an audio chunk is (mostly) silence via RMS volume."""
    try:
        return audioop.rms(pcm16, 2) < threshold
    except Exception:
        return False

# ================== TTS Synthesis ==================

def _sync_tts_linear16(ssml: str, language_code: str, voice_name: str, gender: Optional[str], sample_rate_hz: int = 24000):
    """Blocking call to synthesize SSML text to raw PCM16 audio."""
    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz
    )
    synthesis_input = tts.SynthesisInput(ssml=ssml)
    return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

async def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", sample_rate_hz: int = 24000, sentiment: float = 0.0) -> Optional[bytes]:
    """Async wrapper: uses a thread to call Google TTS synchronously."""
    ssml = ssml_for_text(text, sentiment=sentiment, prosody_rate=0.95)
    lang_code, voice_name, gender = get_best_voice(language_code)
    loop = asyncio.get_running_loop()
    try:
        # Limit concurrent TTS calls to control rate
        await asyncio.wait_for(global_tts_semaphore.acquire(), timeout=3.0)
    except Exception:
        logger.warning("TTS queue busy")
        return None

    try:
        # Run blocking TTS in executor
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

# ================== Query Normalization ==================

_CONTRACTIONS = {
    r"\bwhat's\b": "what is", r"\bwhats\b": "what is", r"\bwhere's\b": "where is",
    r"\bwhen's\b": "when is", r"\bhow's\b": "how is", r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
}
def normalize_and_expand_query(transcript: str) -> str:
    """Lowercase, expand contractions, dedupe words, and add semantic keywords."""
    if not transcript:
        return ""
    s = transcript.lower().strip()
    # Expand contractions
    for patt, repl in _CONTRACTIONS.items():
        s = re.sub(patt, repl, s)
    toks = s.split()
    # Remove duplicate consecutive tokens
    dedup = []; prev = None
    for t in toks:
        if t != prev:
            dedup.append(t)
        prev = t

    # Semantic expansion mapping (domain-specific keywords)
    mappings = {
        "timings": "business hours operating hours schedule",
        "timing": "business hours operating hours schedule",
        "phone number": "contact number telephone",
        "open": "business hours operating hours",
        "close": "business hours operating hours",
        "closed": "business hours operating hours",
        "appointment": "appointment booking consultation schedule",
        "doctor": "doctor physician consultant specialist",
        "payment": "payment methods accepted cash card upi",
        "payments": "payment methods accepted cash card upi",
        "location": "address location directions",
        "address": "address location directions",
        "service": "service offering treatment",
        "services": "services offerings treatments",
    }
    out = []
    i = 0
    while i < len(dedup):
        # Check two-word phrases first
        two = " ".join(dedup[i:i+2]) if i+1 < len(dedup) else None
        if two and two in mappings:
            out.extend(mappings[two].split())
            i += 2
            continue
        # Single word
        w = dedup[i]
        out.append(w)
        if w in mappings:
            out.extend(mappings[w].split())
        i += 1
    return " ".join(out)

# ================== STT Worker ==================

def grpc_stt_worker(loop, audio_queue: queue.Queue, transcripts_queue: asyncio.Queue, stop_event: threading.Event, language_code: str):
    """
    Threaded worker that reads PCM16 audio chunks from audio_queue,
    streams them to Google STT, and puts RecognitionResponse objects onto transcripts_queue.
    """
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
        # Initial request with config
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_cfg)
        # Then stream audio chunks
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error pulling audio chunk: %s", e)
                if stop_event.is_set():
                    break

    try:
        logger.info("Starting STT worker thread (language=%s)", language_code)
        responses = _speech_client.streaming_recognize(gen_requests_with_config())
        for response in responses:
            if stop_event.is_set():
                break
            # Put each SpeechRecognitionResponse into async queue for processing
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(response), loop)
    except TypeError as e:
        # Fallback for older client signature
        logger.info("Legacy STT client, using alternate call method")
        cfg = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
        )
        streaming_cfg = speech.StreamingRecognitionConfig(config=cfg, interim_results=True, single_utterance=False)
        responses = _speech_client.streaming_recognize(streaming_cfg, gen_requests_with_config())
        for response in responses:
            if stop_event.is_set():
                break
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(response), loop)
    except Exception as e:
        logger.exception("STT worker error: %s", e)
    finally:
        logger.info("STT worker exiting (language=%s)", language_code)

# ================== Helper Functions ==================

def advanced_sentiment_score(text: str) -> float:
    """Simple sentiment analysis: weight positive/negative keywords with modifiers."""
    if not text:
        return 0.0
    tl = text.lower()
    words = tl.split()
    pos_score = 0.0
    neg_score = 0.0
    # Scoring lists
    _POS_WORDS = {"good","great","happy","helpful","amazing","perfect","excellent","love","thanks","wonderful"}
    _NEG_WORDS = {"bad","sad","angry","hate","terrible","horrible","awful","problem","frustrat","annoyed","worst"}
    _INTENSITY = {"very":1.5, "really":1.5, "extremely":2.0, "super":1.8, "somewhat":0.6, "slightly":0.5}
    for i, word in enumerate(words):
        intensity = 1.0
        if i > 0 and words[i-1] in _INTENSITY:
            intensity = _INTENSITY[words[i-1]]
        if any(pos in word for pos in _POS_WORDS):
            pos_score += intensity
        if any(neg in word for neg in _NEG_WORDS):
            neg_score += intensity
    # Check for negation flipping
    negation_words = {"not","no","never","neither","dont","can't","cant"}
    if any(n in tl for n in negation_words):
        pos_score, neg_score = neg_score * 0.8, pos_score * 0.8
    if pos_score == 0 and neg_score == 0:
        return 0.0
    total = pos_score + neg_score
    sentiment = (pos_score - neg_score) / total
    return max(-1.0, min(1.0, sentiment))

def extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """Try to parse a JSON object from a text blob (robust to minor format issues)."""
    if not s:
        return None
    start = s.find("{"); end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_blob = s[start:end+1]
    try:
        return json.loads(json_blob)
    except Exception:
        # Attempt common fixes
        safe = json_blob.replace("\n", " ").replace("'", '"')
        safe = re.sub(r',\s*}', '}', safe)
        safe = re.sub(r',\s*]', ']', safe)
        try:
            return json.loads(safe)
        except Exception:
            return None

def calculate_rag_confidence(results: List[Dict]) -> float:
    """Compute a confidence score from Pinecone search results."""
    if not results:
        return 0.0
    try:
        scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
        if not scores:
            return 0.0
        avg_score = sum(scores) / len(scores)
        # Normalize cosine similarity (-1..1) to 0..1 confidence
        confidence = (avg_score + 1.0) / 2.0
        confidence = max(0.0, min(1.0, confidence))
        # Boost if many good matches
        if len(scores) >= 3 and avg_score > 0.3:
            confidence = min(1.0, confidence * 1.2)
        return confidence
    except Exception:
        return 0.0

# ================== Intent Classification ==================

async def classify_query_intent(text: str, rag_available: bool) -> Dict[str, Any]:
    """
    Determine intent type: greeting, factual (lookup), or conversational.
    Returns type and a confidence score.
    """
    if not text or len(text.strip()) < 2:
        return {"type": "conversational", "confidence": 0.5}
    text_lower = text.lower().strip()
    greeting_patterns = [
        r'\b(hi|hello|hey|good morning|good evening|good afternoon)\b',
        r'\b(how are you|what\'s up)\b'
    ]
    factual_patterns = [
        r'\b(what|when|where|which|who|how much|how many)\b',
        r'\b(tell me about|explain|describe)\b',
        r'\b(hours|timing|price|cost|location|address|phone|contact)\b',
        r'\b(service|appointment|booking|consultation|doctor)\b'
    ]
    for pattern in greeting_patterns:
        if re.search(pattern, text_lower):
            return {"type": "greeting", "confidence": 0.9}
    for pattern in factual_patterns:
        if re.search(pattern, text_lower):
            return {"type": "factual", "confidence": 0.85}
    return {"type": "conversational", "confidence": 0.6}

# ================== Response Generation ==================

async def generate_contextual_response(
    user_text: str,
    language_code: str,
    conversation_history: deque,
    rag_results: Optional[List[Dict]] = None
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Generate assistant response. Chooses strategy based on intent:
    - greeting: simple warm reply
    - factual: use RAG context
    - conversation: general chat
    Returns (response_text, sentiment_score, metadata).
    """
    loop = asyncio.get_running_loop()
    # Build conversation context for prompt
    convo_context = []
    for entry in list(conversation_history)[-6:]:
        role, txt, *_ = entry
        prefix = "User:" if role == "user" else "Assistant:"
        convo_context.append(f"{prefix} {txt}")
    convo_prefix = "\n".join(convo_context) + "\n\n" if convo_context else ""

    intent_info = await classify_query_intent(user_text, bool(rag_results))
    metadata = {"intent": intent_info["type"], "confidence": intent_info["confidence"]}
    try:
        if intent_info["type"] == "greeting":
            system_msg = (
                f"You are {BUSINESS_NAME}'s friendly voice assistant. "
                "Respond to this greeting naturally and warmly in 1-2 sentences. "
                "Offer to help without being pushy."
            )
            prompt = f"{convo_prefix}User: {user_text}\n\nRespond naturally as {BUSINESS_NAME}'s assistant:"
            partial = functools.partial(_gemini.generate_response,
                                        prompt=prompt, system_message=system_msg,
                                        temperature=0.8, max_tokens=100)
        elif intent_info["type"] == "factual" and rag_results:
            # Prepare context from RAG results
            context_text = "\n\n".join(
                f"Source: {r.get('source','Doc')}\n{r.get('chunk_text','')}" 
                for r in rag_results[:4]
            )
            confidence = calculate_rag_confidence(rag_results)
            metadata.update({"response_type": "rag", "sources_used": len(rag_results)})
            system_msg = (
                f"You are {BUSINESS_NAME}'s voice assistant. Use ONLY the provided context to answer. "
                "Answer naturally in 2-3 sentences. Use 'we' when referring to the business. "
                "If context is insufficient, say so clearly and offer to connect them with someone who can help."
            )
            prompt = (
                f"{convo_prefix}CONTEXT:\n{context_text}\n\n"
                f"User Question: {user_text}\n\nYour helpful answer:"
            )
            partial = functools.partial(_gemini.generate_response,
                                        prompt=prompt, system_message=system_msg,
                                        temperature=0.3, max_tokens=250)
        else:
            # Conversational fallback
            system_msg = (
                f"You are {BUSINESS_NAME}'s friendly voice assistant. "
                "Respond naturally and helpfully to the user. Keep responses brief (2-3 sentences). "
                "Be warm, professional, and offer assistance when appropriate."
            )
            prompt = f"{convo_prefix}User: {user_text}\n\nRespond naturally:"
            partial = functools.partial(_gemini.generate_response,
                                        prompt=prompt, system_message=system_msg,
                                        temperature=0.75, max_tokens=200)

        # Call Gemini LLM
        fut = loop.run_in_executor(executor, partial)
        response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
        sentiment = advanced_sentiment_score(response)
        metadata["response_type"] = intent_info["type"]
        return (response.strip(), sentiment, metadata)

    except Exception as e:
        logger.exception("Response generation failed: %s", e)
        # Fallback generic reply
        fallback = "I'm sorry, something went wrong. Can I help with anything else?"
        return (fallback, 0.0, {"intent": intent_info["type"], "error": True})

# ================== Streaming Response ==================

def _stream_writer_thread(loop_ref, prompt: str, system_msg: str, token_queue: asyncio.Queue, stop_evt: threading.Event):
    """
    Thread to stream tokens from Gemini and put them in token_queue.
    """
    try:
        logger.info("Starting streaming generation")
        gen = _gemini.generate_stream(prompt=prompt, system_message=system_msg, temperature=0.7)
        for chunk in gen:
            if stop_evt.is_set():
                logger.info("Stream stopped by event")
                break
            if chunk:
                try:
                    asyncio.run_coroutine_threadsafe(token_queue.put(chunk), loop_ref)
                except Exception as e:
                    logger.error(f"Failed to queue chunk: {e}")
                    break
        logger.info("Streaming generation complete")
    except Exception as e:
        logger.exception("Stream writer error: %s", e)
    finally:
        # Signal end of stream
        try:
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop_ref)
        except Exception:
            pass

async def run_streaming_response(user_text: str, language_code: str, conversation_history: deque, ws: WebSocket, tts_queue: Optional[asyncio.Queue] = None) -> Tuple[str, float]:
    """
    Stream Gemini response token-by-token, queueing complete sentences for TTS as they form.
    Returns full response text and overall sentiment.
    """
    loop = asyncio.get_running_loop()
    token_q = asyncio.Queue()
    stop_evt = threading.Event()

    # Build conversation context
    convo_context = []
    for entry in list(conversation_history)[-6:]:
        role, txt, *_ = entry
        prefix = "User:" if role == "user" else "Assistant:"
        convo_context.append(f"{prefix} {txt}")
    convo_prefix = "\n".join(convo_context) + "\n\n" if convo_context else ""
    system_msg = (
        f"You are {BUSINESS_NAME}'s warm, helpful voice assistant. "
        "Respond naturally and conversationally. Keep responses concise (2-3 sentences). "
        "Be friendly, professional, and helpful."
    )
    prompt = f"{convo_prefix}User: {user_text}\n\nYour natural response:"

    # Start the streaming generator in a separate thread
    writer_thread = threading.Thread(
        target=_stream_writer_thread,
        args=(loop, prompt, system_msg, token_q, stop_evt),
        daemon=True
    )
    writer_thread.start()

    full_text = ""
    sentence_buffer = ""
    first_sentence_sent = False

    try:
        while True:
            token = await token_q.get()
            if token is None:
                break
            chunk = str(token)
            full_text += chunk
            sentence_buffer += chunk
            # Detect end of sentences
            sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)
            if len(sentences) > 1:
                complete_sentences = sentences[:-1]
                sentence_buffer = sentences[-1]
            else:
                complete_sentences = []
                if len(sentence_buffer) >= STREAM_SENTENCE_CHAR_LIMIT or "\n" in sentence_buffer:
                    split_at = sentence_buffer.rfind(" ", 0, STREAM_SENTENCE_CHAR_LIMIT)
                    if split_at <= 0:
                        split_at = STREAM_SENTENCE_CHAR_LIMIT
                    complete_sentences = [sentence_buffer[:split_at].strip()]
                    sentence_buffer = sentence_buffer[split_at:].strip()

            # Queue complete sentences for TTS
            for sentence in complete_sentences:
                if not sentence:
                    continue
                sent_score = advanced_sentiment_score(sentence)
                if tts_queue:
                    await tts_queue.put({"text": sentence, "language": language_code, "sentiment": sent_score})
                else:
                    # Fallback direct send
                    await send_tts_and_audio(sentence, language_code=language_code, sentiment=sent_score)

        # Send any remaining text
        remaining = sentence_buffer.strip()
        if remaining:
            sent_score = advanced_sentiment_score(remaining)
            if tts_queue:
                await tts_queue.put({"text": remaining, "language": language_code, "sentiment": sent_score})
            else:
                await send_tts_and_audio(remaining, language_code=language_code, sentiment=sent_score)

        overall_sentiment = advanced_sentiment_score(full_text)
        logger.info(f"Streaming complete. Total text length: {len(full_text)} chars")
        return (full_text, overall_sentiment)
    except Exception as e:
        logger.exception("Streaming response error: %s", e)
        return ("I'm sorry, I encountered an issue generating that response.", 0.0)
    finally:
        stop_evt.set()
        try:
            writer_thread.join(timeout=1.0)
        except Exception:
            pass

# ================== Serial TTS Worker (per-connection) ==================

async def _tts_worker_loop(ws: WebSocket, queue: asyncio.Queue):
    """
    Consume TTS jobs sequentially. Each job is a dict: {"text": str, "language": str, "sentiment": float}.
    Synthesizes and sends audio for each sentence.
    """
    current_task: Optional[asyncio.Task] = None
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=TTS_WORKER_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                # Idle timeout: exit worker
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
                    await ws.send_text(json.dumps({
                        "type": "audio",
                        "audio": b64wav,
                        "metadata": {"sentiment": sentiment, "length": len(text)}
                    }))
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
        # Worker cancelled externally (e.g. new barge-in)
        pass
    except Exception:
        logger.exception("TTS worker loop crashed")
    finally:
        # Drain any remaining queue items
        try:
            while not queue.empty():
                _ = queue.get_nowait()
        except Exception:
            pass

# ================== WebSocket Handler ==================

@router.websocket("/ws/hd-audio")
async def hd_audio_ws(ws: WebSocket):
    """Main WebSocket handler for real-time voice conversation."""
    # --- Authentication ---
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

    # Initialize queues and state
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

    # Create per-connection TTS queue and start worker
    tts_queue: asyncio.Queue = asyncio.Queue()
    _tts_worker_task: Optional[asyncio.Task] = asyncio.create_task(_tts_worker_loop(ws, tts_queue))

    async def _do_tts_and_send(ai_text: str, language_code: str, sentiment: float):
        """Generate TTS audio (one utterance) and send."""
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
                    await ws.send_text(json.dumps({
                        "type": "audio",
                        "audio": b64wav,
                        "metadata": {"sentiment": sentiment, "length": len(ai_text)}
                    }))
                    logger.debug(f"Sent TTS audio: {len(ai_text)} chars")
                except Exception as e:
                    logger.error(f"Failed to send audio: {e}")
            else:
                try:
                    await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))
                except Exception:
                    pass
        except asyncio.CancelledError:
            logger.info("TTS task cancelled")
            raise
        except Exception as e:
            logger.exception("TTS generation failed: %s", e)
            try:
                await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))
            except:
                pass
        finally:
            is_bot_speaking = False
            current_tts_task = None

    async def send_tts_and_audio(ai_text: str, language_code: str, sentiment: float = 0.0):
        """Cancel any in-progress TTS and start a new one for ai_text."""
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
            logger.info("TTS task cancelled successfully")

    async def handle_final_utterance(text: str):
        """Process a complete user utterance."""
        nonlocal conversation, is_bot_speaking, current_stream_stop_event
        METRICS["requests"] += 1
        start_ms = time.time() * 1000

        user_text = text.strip()
        if not user_text:
            return
        logger.info(f"Processing utterance: {user_text}")

        # Add to conversation history
        ts = time.time()
        conversation.append(("user", user_text, None, None, 0.0, ts))

        try:
            # 1) RAG search
            norm_q = normalize_and_expand_query(user_text)
            logger.info(f"Normalized query: {norm_q}")
            rag_results = None
            try:
                rag_results = await pinecone_service.search_similar_chunks(
                    client_id=DEFAULT_CLIENT_ID,
                    query=norm_q or user_text,
                    top_k=6,
                    min_score=-1.0
                )
                if rag_results:
                    METRICS["rag_hits"] += 1
                    logger.info(f"RAG found {len(rag_results)} results")
                else:
                    logger.info("No RAG results found")
            except Exception as e:
                logger.exception("Pinecone search failed: %s", e)
                record_error("rag_search_failed")
                rag_results = None

            # 2) Generate response
            response_text, sentiment, metadata = await generate_contextual_response(
                user_text=user_text,
                language_code=language,
                conversation_history=conversation,
                rag_results=rag_results
            )
            record_metric_intent(metadata.get("intent", "unknown"))
            record_metric_sentiment(sentiment)
            if "confidence" in metadata:
                record_metric_confidence(metadata["confidence"])

            # Add assistant response to history
            conversation.append(("assistant", response_text, metadata.get("intent"), metadata.get("entities", {}),
                                 sentiment, time.time()))

            # Send text to client
            try:
                await ws.send_text(json.dumps({
                    "type": "ai_text",
                    "text": response_text,
                    "metadata": {
                        "intent": metadata.get("intent"),
                        "confidence": metadata.get("confidence", 0.0),
                        "sentiment": sentiment,
                        "response_type": metadata.get("response_type"),
                        "session_id": session_id
                    }
                }))
            except Exception as e:
                logger.error(f"Failed to send response text: {e}")

            # Queue TTS for the response
            await tts_queue.put({"text": response_text, "language": language, "sentiment": sentiment})

            # Record latency
            latency_ms = time.time() * 1000 - start_ms
            record_latency(latency_ms)
            logger.info(f"Total response time: {latency_ms:.0f}ms")

        except Exception as e:
            logger.exception("Error processing utterance: %s", e)
            record_error("utterance_processing_failed")
            # Send error message
            error_msg = "I apologize, but I encountered an issue processing your request. Could you please try again?"
            try:
                await ws.send_text(json.dumps({"type": "ai_text", "text": error_msg, "metadata": {"error": True}}))
                await tts_queue.put({"text": error_msg, "language": language, "sentiment": 0.0})
            except:
                pass

    async def debounce_and_handle():
        """Debounce delay after voice input ends, then process utterance."""
        nonlocal pending_debounce_task, utterance_buffer, last_voice_ts
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
            # If new voice occurred, skip
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
        """Consumes STT streaming responses and buffers final transcripts."""
        nonlocal language, utterance_buffer, pending_debounce_task, is_bot_speaking
        nonlocal current_tts_task, current_stream_stop_event, _tts_worker_task, tts_queue

        while True:
            resp = await transcripts_queue.get()
            if resp is None:
                logger.info("Transcript consumer exiting")
                break
            for result in resp.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                interim_text = alt.transcript.strip()
                is_final = getattr(result, "is_final", False)

                # Barge-in: user speaks while assistant is outputting
                if interim_text and is_bot_speaking:
                    logger.info("Barge-in detected: user interrupted the assistant")
                    try:
                        await ws.send_text(json.dumps({"type": "control", "action": "stop_playback"}))
                    except Exception:
                        pass
                    # Cancel TTS
                    if current_tts_task and not current_tts_task.done():
                        try:
                            current_tts_task.cancel()
                        except Exception:
                            pass
                    # Stop streaming response
                    if current_stream_stop_event:
                        try:
                            current_stream_stop_event.set()
                        except Exception:
                            pass
                    # Clear queued audio
                    try:
                        while not tts_queue.empty():
                            _ = tts_queue.get_nowait()
                        logger.debug("Cleared pending TTS queue")
                    except Exception:
                        pass
                    # Restart TTS worker
                    if _tts_worker_task and not _tts_worker_task.done():
                        try:
                            _tts_worker_task.cancel()
                        except Exception:
                            pass
                    _tts_worker_task = asyncio.create_task(_tts_worker_loop(ws, tts_queue))
                    is_bot_speaking = False

                # Send interim transcripts for client UI
                if interim_text:
                    try:
                        await ws.send_text(json.dumps({
                            "type": "transcript",
                            "text": interim_text,
                            "is_final": is_final
                        }))
                    except Exception:
                        pass

                # On final transcript, buffer it
                if is_final and interim_text:
                    logger.info(f"Final transcript: {interim_text}")
                    utterance_buffer.append(interim_text)
                    # Check for language code (if auto-detect)
                    detected_lang = getattr(result, "language_code", None)
                    if detected_lang:
                        language = detected_lang
                    if not pending_debounce_task or pending_debounce_task.done():
                        pending_debounce_task = asyncio.create_task(debounce_and_handle())

    # Start STT thread and transcript consumer
    try:
        loop = asyncio.get_event_loop()
        stop_event.clear()
        stt_thread = threading.Thread(
            target=grpc_stt_worker,
            args=(loop, audio_queue, transcripts_queue, stop_event, language),
            daemon=True
        )
        stt_thread.start()
        transcript_consumer_task = asyncio.create_task(process_transcripts_task())

        # Notify client ready
        await ws.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
            "language": language
        }))
        logger.info(f"Session {session_id} ready (language={language})")

        # Main loop: handle incoming WS messages
        while True:
            msg = await ws.receive()
            if msg is None:
                continue
            msg_type = msg.get("type")

            # Handle disconnect
            if msg_type == "websocket.disconnect":
                logger.info("Client websocket disconnected")
                break

            # Text frame (control messages)
            if "text" in msg and msg["text"] is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except Exception:
                    await ws.send_text(json.dumps({"type": "error", "error": "invalid_json"}))
                    continue
                mtype = ctrl.get("type")
                if mtype == "start":
                    # Client starts sending audio
                    meta = ctrl.get("meta", {}) or {}
                    new_lang = meta.get("language")
                    if new_lang and new_lang != language:
                        now_ts = time.time()
                        if now_ts - last_restart_ts >= MIN_RESTART_INTERVAL:
                            logger.info(f"Language change requested: {language} -> {new_lang}")
                            language = new_lang
                            with restarting_lock:
                                try:
                                    stop_event.set()
                                    audio_queue.put_nowait(None)
                                    if stt_thread and stt_thread.is_alive():
                                        stt_thread.join(timeout=2.0)
                                except Exception as e:
                                    logger.exception("Error stopping STT thread: %s", e)
                                # Restart STT thread with new language
                                stop_event = threading.Event()
                                stt_thread = threading.Thread(
                                    target=grpc_stt_worker,
                                    args=(loop, audio_queue, transcripts_queue, stop_event, language),
                                    daemon=True
                                )
                                stt_thread.start()
                                last_restart_ts = time.time()
                                logger.info(f"Restarted STT worker with language={language}")
                        else:
                            logger.info("Language restart suppressed by backoff")
                    await ws.send_text(json.dumps({"type": "ack", "message": "started"}))
                elif mtype == "audio":
                    # Legacy path: base64 audio payload
                    b64 = ctrl.get("payload")
                    if not b64:
                        continue
                    try:
                        pcm = base64.b64decode(b64)
                    except Exception:
                        await ws.send_text(json.dumps({"type": "error", "error": "bad_audio_b64"}))
                        continue
                    silent = is_silence(pcm)
                    if not silent:
                        last_voice_ts = time.time()
                    if audio_queue.qsize() > 350:
                        logger.warning(f"Audio queue large ({audio_queue.qsize()}), dropping input")
                        continue
                    try:
                        audio_queue.put_nowait(pcm)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")
                        continue
                elif mtype == "stop":
                    logger.info("Client requested stop")
                    stop_event.set()
                    try:
                        audio_queue.put_nowait(None)
                    except Exception:
                        pass
                    await transcripts_queue.put(None)
                    try:
                        await ws.send_text(json.dumps({"type": "bye"}))
                    except:
                        pass
                    try:
                        await ws.close()
                    except:
                        pass
                    return
                elif mtype == "metrics":
                    # Send session metrics
                    session_duration = time.time() - session_start
                    avg_response_time = (sum(METRICS["response_times_ms"]) / len(METRICS["response_times_ms"])
                                         if METRICS["response_times_ms"] else None)
                    avg_sentiment = (sum(METRICS["sentiments"]) / len(METRICS["sentiments"])
                                     if METRICS["sentiments"] else None)
                    try:
                        await ws.send_text(json.dumps({
                            "type": "metrics",
                            "session_id": session_id,
                            "metrics": {
                                "requests": METRICS["requests"],
                                "intent_counts": dict(METRICS["intent_counts"]),
                                "avg_response_ms": avg_response_time,
                                "sentiment_samples": len(METRICS["sentiments"]),
                                "avg_sentiment": avg_sentiment,
                                "confidence_samples": len(METRICS["confidences"]),
                                "rag_hits": METRICS["rag_hits"],
                                "streaming_uses": METRICS["streaming_uses"],
                                "errors": dict(METRICS["errors"]),
                                "session_duration_s": session_duration
                            }
                        }))
                    except Exception as e:
                        logger.error(f"Failed to send metrics: {e}")
                else:
                    await ws.send_text(json.dumps({"type": "error", "error": "unknown_type"}))
            # Binary frame (raw audio bytes)
            elif "bytes" in msg and msg["bytes"] is not None:
                pcm = msg["bytes"]
                silent = is_silence(pcm)
                if not silent:
                    last_voice_ts = time.time()
                if audio_queue.qsize() > 350:
                    logger.warning(f"Audio queue large ({audio_queue.qsize()}), dropping input")
                    continue
                try:
                    audio_queue.put_nowait(pcm)
                except queue.Full:
                    logger.warning("Audio queue full, dropping chunk")
                    continue
            else:
                logger.debug(f"Received unexpected message: {msg}")
                continue

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
        record_error("websocket_error")
    finally:
        logger.info("Cleaning up session")
        try:
            stop_event.set()
            audio_queue.put_nowait(None)
        except Exception:
            pass
        try:
            transcripts_queue.put_nowait(None)
        except Exception:
            pass
        # Stop TTS worker
        try:
            await tts_queue.put(None)
            if _tts_worker_task and not _tts_worker_task.done():
                try:
                    _tts_worker_task.cancel()
                except Exception:
                    pass
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
        session_duration = time.time() - session_start
        logger.info(f"Session {session_id} cleanup complete (duration {session_duration:.1f}s).")
