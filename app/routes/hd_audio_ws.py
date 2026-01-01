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

logger = logging.getLogger(__name__)
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
    """Enhanced SSML with emotional intelligence"""
    s = max(-1.0, min(1.0, sentiment or 0.0))

    # Dynamic prosody based on sentiment
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

    # Enhanced text escaping and pausing
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc = esc.replace(", ", ", <break time='140ms'/> ")
    esc = esc.replace(". ", ". <break time='250ms'/> ")
    esc = esc.replace("? ", "? <break time='250ms'/> ")
    esc = esc.replace("! ", "! <break time='250ms'/> ")
    esc = esc.replace(": ", ": <break time='180ms'/> ")

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


# ================== TTS Synthesis ==================
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


# ================== Query Normalization ==================
_CONTRACTIONS = {
    r"\bwhat's\b": "what is",
    r"\bwhats\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bwhen's\b": "when is",
    r"\bhow's\b": "how is",
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
}


def normalize_and_expand_query(transcript: str) -> str:
    if not transcript:
        return ""

    s = transcript.lower().strip()

    # Expand contractions
    for patt, repl in _CONTRACTIONS.items():
        s = re.sub(patt, repl, s)

    toks = s.split()

    # Dedupe consecutive tokens
    dedup = []
    prev = None
    for t in toks:
        if t != prev:
            dedup.append(t)
        prev = t

    # Semantic expansion mappings
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
                logger.exception("Error pulling audio chunk: %s", e)
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
                logger.exception("Error pulling audio chunk (audio_only): %s", e)
                if stop_event.is_set():
                    break

    def call_streaming_recognize_with_fallback():
        try:
            return _speech_client.streaming_recognize(gen_requests_with_config())
        except TypeError as e:
            msg = str(e).lower()
            if "missing" in msg and "requests" in msg:
                logger.info("Detected legacy signature, using fallback call")
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


# ================== Advanced Metrics ==================
METRICS = {
    "requests": 0,
    "intent_counts": defaultdict(int),
    "sentiments": [],
    "confidences": [],
    "response_times_ms": [],
    "rag_hits": 0,
    "streaming_uses": 0,
    "errors": defaultdict(int),
}


def record_metric_intent(intent: str):
    if intent:
        METRICS["intent_counts"][intent] += 1


def record_metric_sentiment(s: float):
    METRICS["sentiments"].append(s)


def record_metric_confidence(c: float):
    METRICS["confidences"].append(c)


def record_latency(ms: float):
    METRICS["response_times_ms"].append(ms)


def record_error(error_type: str):
    METRICS["errors"][error_type] += 1


# ================== Advanced Sentiment Analysis ==================
_POS_WORDS = {
    "good", "great", "happy", "awesome", "fantastic", "wonderful", "excellent",
    "love", "thanks", "thank", "appreciate", "helpful", "perfect", "amazing",
    "brilliant", "pleased", "satisfied", "delighted"
}

_NEG_WORDS = {
    "bad", "sad", "angry", "upset", "hate", "terrible", "horrible", "awful",
    "problem", "frustrat", "frustration", "annoyed", "disappointed", "issue",
    "not working", "broken", "fail", "failed", "useless", "worse", "worst"
}

_INTENSITY_MODIFIERS = {
    "very": 1.5, "really": 1.5, "extremely": 2.0, "super": 1.8,
    "somewhat": 0.6, "slightly": 0.5, "kind of": 0.6, "a bit": 0.5
}


def advanced_sentiment_score(text: str) -> float:
    """Enhanced sentiment analysis with intensity modifiers and context"""
    if not text:
        return 0.0

    tl = text.lower()
    words = tl.split()

    pos_score = 0.0
    neg_score = 0.0

    for i, word in enumerate(words):
        # Check for intensity modifiers
        intensity = 1.0
        if i > 0 and words[i-1] in _INTENSITY_MODIFIERS:
            intensity = _INTENSITY_MODIFIERS[words[i-1]]

        # Count sentiment words with intensity
        if any(pos in word for pos in _POS_WORDS):
            pos_score += intensity
        if any(neg in word for neg in _NEG_WORDS):
            neg_score += intensity

    # Check for negations that flip sentiment
    negation_words = {"not", "no", "never", "neither", "dont", "don't", "cant", "can't"}
    for neg in negation_words:
        if neg in tl:
            # Swap scores if negation present
            pos_score, neg_score = neg_score * 0.8, pos_score * 0.8
            break

    if pos_score == 0 and neg_score == 0:
        return 0.0

    # Calculate normalized sentiment
    total = pos_score + neg_score
    sentiment = (pos_score - neg_score) / total

    return max(-1.0, min(1.0, sentiment))


# ================== Helper Functions ==================
def extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction from LLM response"""
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
            # Try cleaning common issues
            safe = json_blob.replace("\n", " ").replace("'", '"')
            # Remove trailing commas
            safe = re.sub(r',\s*}', '}', safe)
            safe = re.sub(r',\s*]', ']', safe)
            return json.loads(safe)
        except Exception:
            return None


def calculate_rag_confidence(results: List[Dict]) -> float:
    """Calculate confidence from RAG search results"""
    if not results:
        return 0.0

    try:
        scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)

        # Normalize Pinecone cosine similarity (-1 to 1) to confidence (0 to 1)
        # Higher scores indicate better matches
        confidence = (avg_score + 1.0) / 2.0
        confidence = max(0.0, min(1.0, confidence))

        # Boost confidence if multiple high-quality matches
        if len(scores) >= 3 and avg_score > 0.3:
            confidence = min(1.0, confidence * 1.2)

        return confidence
    except Exception:
        return 0.0


# ================== Intelligent Intent Classification ==================
async def classify_query_intent(text: str, rag_available: bool) -> Dict[str, Any]:
    """
    Classify query intent to determine optimal response strategy:
    - greeting: Simple conversational greeting
    - factual: Requires knowledge base lookup
    - conversational: General chat, no specific facts needed
    """
    if not text or len(text.strip()) < 2:
        return {"type": "conversational", "confidence": 0.5}

    text_lower = text.lower().strip()

    # Pattern-based classification for speed
    greeting_patterns = [
        r'\b(hi|hello|hey|good morning|good evening|good afternoon)\b',
        r'\b(how are you|whats up|what\'s up)\b'
    ]

    factual_patterns = [
        r'\b(what|when|where|which|who|how much|how many)\b',
        r'\b(tell me about|explain|describe|information about)\b',
        r'\b(hours|timing|price|cost|location|address|phone|contact)\b',
        r'\b(service|appointment|booking|consultation|doctor)\b'
    ]

    # Check greeting patterns
    for pattern in greeting_patterns:
        if re.search(pattern, text_lower):
            return {"type": "greeting", "confidence": 0.9}

    # Check factual patterns
    for pattern in factual_patterns:
        if re.search(pattern, text_lower):
            return {"type": "factual", "confidence": 0.85}

    # Default to conversational
    return {"type": "conversational", "confidence": 0.6}


# ================== Context-Aware Response Generation ==================
async def generate_contextual_response(
    user_text: str,
    language_code: str,
    conversation_history: deque,
    rag_results: Optional[List[Dict]] = None
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Generate intelligent response based on context and available information
    Returns: (response_text, sentiment, metadata)
    """
    loop = asyncio.get_running_loop()

    # Build conversation context
    convo_context = []
    for entry in list(conversation_history)[-6:]:
        role = entry[0]
        txt = entry[1]
        prefix = "User:" if role == "user" else "Assistant:"
        convo_context.append(f"{prefix} {txt}")

    convo_prefix = "\n".join(convo_context) + "\n\n" if convo_context else ""

    # Classify intent
    intent_info = await classify_query_intent(user_text, rag_results is not None and len(rag_results) > 0)

    # Handle based on intent type
    if intent_info["type"] == "greeting":
        # Use Gemini for natural greeting responses
        system_msg = (
            f"You are {BUSINESS_NAME}'s friendly voice assistant. "
            "Respond to this greeting naturally and warmly in 1-2 sentences. "
            "Offer to help without being pushy."
        )

        prompt = f"{convo_prefix}User: {user_text}\n\nRespond naturally as {BUSINESS_NAME}'s assistant:"

        partial = functools.partial(
            _gemini.generate_response,
            prompt=prompt,
            system_message=system_msg,
            temperature=0.8,
            max_tokens=100
        )

        try:
            fut = loop.run_in_executor(executor, partial)
            response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            sentiment = advanced_sentiment_score(response)

            metadata = {
                "intent": "greeting",
                "confidence": intent_info["confidence"],
                "response_type": "conversational"
            }

            return (response.strip(), sentiment, metadata)
        except Exception as e:
            logger.exception("Greeting generation failed: %s", e)
            return (f"Hello! Welcome to {BUSINESS_NAME}. How can I help you today?", 0.3, {"intent": "greeting", "error": True})

    elif intent_info["type"] == "factual" and rag_results:
        # Use RAG + Gemini for factual queries
        context_text = "\n\n".join([
            f"Source: {r.get('source', 'Document')}\n{r.get('chunk_text', '')}"
            for r in rag_results[:4]
        ])

        confidence = calculate_rag_confidence(rag_results)

        system_msg = (
            f"You are {BUSINESS_NAME}'s voice assistant. Use ONLY the provided context to answer. "
            "Answer naturally in 2-3 sentences. Use 'we' when referring to the business. "
            "If context is insufficient, say so clearly and offer to connect them with someone who can help."
        )

        prompt = f"{convo_prefix}CONTEXT:\n{context_text}\n\nUser Question: {user_text}\n\nYour helpful answer:"

        partial = functools.partial(
            _gemini.generate_response,
            prompt=prompt,
            system_message=system_msg,
            temperature=0.3,
            max_tokens=250
        )

        try:
            fut = loop.run_in_executor(executor, partial)
            response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            sentiment = advanced_sentiment_score(response)

            metadata = {
                "intent": "factual",
                "confidence": confidence,
                "response_type": "rag",
                "sources_used": len(rag_results)
            }

            return (response.strip(), sentiment, metadata)
        except Exception as e:
            logger.exception("RAG response generation failed: %s", e)
            return ("I apologize, I'm having trouble accessing that information right now. Can I help with something else?", -0.2, {"intent": "factual", "error": True})

    else:
        # Conversational fallback - use Gemini for natural dialogue
        system_msg = (
            f"You are {BUSINESS_NAME}'s friendly voice assistant. "
            "Respond naturally and helpfully to the user. Keep responses brief (2-3 sentences). "
            "Be warm, professional, and offer assistance when appropriate."
        )

        prompt = f"{convo_prefix}User: {user_text}\n\nRespond naturally:"

        partial = functools.partial(
            _gemini.generate_response,
            prompt=prompt,
            system_message=system_msg,
            temperature=0.75,
            max_tokens=200
        )

        try:
            fut = loop.run_in_executor(executor, partial)
            response = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            sentiment = advanced_sentiment_score(response)

            metadata = {
                "intent": "conversational",
                "confidence": intent_info["confidence"],
                "response_type": "conversational"
            }

            return (response.strip(), sentiment, metadata)
        except Exception as e:
            logger.exception("Conversational response failed: %s", e)
            return ("I'm happy to chat, but I'm having some technical difficulties. Could you try asking me again?", 0.0, {"intent": "conversational", "error": True})


# ================== Streaming LLM Implementation ==================
def _stream_writer_thread(loop_ref, prompt: str, system_msg: str, token_queue: asyncio.Queue, stop_evt: threading.Event):
    """Thread worker for streaming LLM tokens"""
    try:
        logger.info("🌊 Starting streaming generation")
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

        logger.info("✅ Streaming generation complete")
    except Exception as e:
        logger.exception("Stream writer error: %s", e)
    finally:
        try:
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop_ref)
        except Exception:
            pass


async def run_streaming_response(
    user_text: str,
    language_code: str,
    conversation_history: deque,
    ws: WebSocket,
    send_tts_callback
) -> Tuple[str, float]:
    """
    Stream LLM response with progressive TTS generation
    Returns: (full_text, sentiment)
    """
    loop = asyncio.get_running_loop()
    token_q = asyncio.Queue()
    stop_evt = threading.Event()

    # Build context
    convo_context = []
    for entry in list(conversation_history)[-6:]:
        role = entry[0]
        txt = entry[1]
        prefix = "User:" if role == "user" else "Assistant:"
        convo_context.append(f"{prefix} {txt}")

    convo_prefix = "\n".join(convo_context) + "\n\n" if convo_context else ""

    system_msg = (
        f"You are {BUSINESS_NAME}'s warm, helpful voice assistant. "
        "Respond naturally and conversationally. Keep responses concise (2-3 sentences). "
        "Be friendly, professional, and helpful."
    )

    prompt = f"{convo_prefix}User: {user_text}\n\nYour natural response:"

    # Start streaming thread
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

            # Detect complete sentences
            sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)

            if len(sentences) > 1:
                complete_sentences = sentences[:-1]
                sentence_buffer = sentences[-1]
            else:
                complete_sentences = []

            # Send completed sentences to TTS
            for sentence in complete_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sent_score = advanced_sentiment_score(sentence)

                # Send first sentence immediately, others in background
                if not first_sentence_sent:
                    first_sentence_sent = True
                    logger.info(f"📢 First sentence ready: {sentence[:50]}...")
                    await send_tts_callback(sentence, language_code=language_code, sentiment=sent_score)
                else:
                    asyncio.create_task(send_tts_callback(sentence, language_code=language_code, sentiment=sent_score))

        # Send any remaining text in buffer
        remaining = sentence_buffer.strip()
        if remaining:
            sent_score = advanced_sentiment_score(remaining)
            await send_tts_callback(remaining, language_code=language_code, sentiment=sent_score)

        # Calculate overall sentiment
        overall_sentiment = advanced_sentiment_score(full_text)

        logger.info(f"✅ Streaming complete. Total length: {len(full_text)} chars")
        METRICS["streaming_uses"] += 1

        return (full_text, overall_sentiment)

    except Exception as e:
        logger.exception("Streaming response error: %s", e)
        record_error("streaming_failed")
        # Fallback to empty response
        return ("I apologize, but I encountered an issue generating that response. Could you please ask again?", 0.0)
    finally:
        stop_evt.set()
        try:
            writer_thread.join(timeout=1.0)
        except Exception:
            pass


# ================== WebSocket Handler ==================
@router.websocket("/ws/hd-audio")
async def hd_audio_ws(ws: WebSocket):
    """Main WebSocket handler for real-time voice conversation"""

    # Authentication
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

    # Conversation state with rich metadata
    conversation: deque = deque(maxlen=20)  # Increased for better context
    utterance_buffer: List[str] = []
    pending_debounce_task: Optional[asyncio.Task] = None

    last_voice_ts = time.time()
    restarting_lock = threading.Lock()
    last_restart_ts = 0.0

    # TTS and streaming state
    is_bot_speaking = False
    current_tts_task: Optional[asyncio.Task] = None
    current_stream_stop_event: Optional[threading.Event] = None

    # Session metadata
    session_start = time.time()
    session_id = f"session_{int(session_start)}"

    async def _do_tts_and_send(ai_text: str, language_code: str, sentiment: float):
        """Internal TTS generation and transmission"""
        nonlocal is_bot_speaking, current_tts_task

        try:
            is_bot_speaking = True
            pcm = await synthesize_text_to_pcm(
                ai_text,
                language_code=language_code,
                sample_rate_hz=24000,
                sentiment=sentiment
            )

            if current_tts_task and current_tts_task.cancelled():
                logger.info("TTS task was cancelled during generation")
                return

            if pcm:
                wav_bytes = make_wav_from_pcm16(pcm, sample_rate=24000)
                b64wav = base64.b64encode(wav_bytes).decode("ascii")

                try:
                    await ws.send_text(json.dumps({
                        "type": "audio",
                        "audio": b64wav,
                        "metadata": {
                            "sentiment": sentiment,
                            "length": len(ai_text)
                        }
                    }))
                    logger.debug(f"✅ Sent TTS audio: {len(ai_text)} chars")
                except Exception as e:
                    logger.error(f"Failed to send audio: {e}")
            else:
                await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))

        except asyncio.CancelledError:
            logger.info("TTS task cancelled (barge-in)")
            raise
        except Exception as e:
            logger.exception("TTS generation failed: %s", e)
            record_error("tts_failed")
            try:
                await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))
            except:
                pass
        finally:
            is_bot_speaking = False
            current_tts_task = None

    async def send_tts_and_audio(ai_text: str, language_code: str, sentiment: float = 0.0):
        """Public TTS interface with cancellation support"""
        nonlocal current_tts_task

        if current_tts_task and not current_tts_task.done():
            try:
                current_tts_task.cancel()
                await asyncio.sleep(0.05)  # Brief pause for cleanup
            except Exception:
                pass

        current_tts_task = asyncio.create_task(_do_tts_and_send(ai_text, language_code, sentiment))

        try:
            await current_tts_task
        except asyncio.CancelledError:
            logger.info("TTS task cancelled successfully")

    async def handle_final_utterance(text: str):
        """Process complete user utterance and generate response"""
        nonlocal conversation, is_bot_speaking, current_stream_stop_event

        METRICS["requests"] += 1
        start_ms = time.time() * 1000

        user_text = text.strip()
        if not user_text:
            return

        logger.info(f"💬 Processing utterance: {user_text}")

        # Add to conversation history
        ts = time.time()
        conversation.append(("user", user_text, None, None, 0.0, ts))

        try:
            # Step 1: Perform RAG search
            norm_q = normalize_and_expand_query(user_text)
            logger.info(f"🔍 Normalized query: {norm_q}")

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
                    logger.info(f"✅ RAG found {len(rag_results)} results")
                else:
                    logger.info("ℹ️ No RAG results found")

            except Exception as e:
                logger.exception("Pinecone search failed: %s", e)
                record_error("rag_search_failed")
                rag_results = None

            # Step 2: Classify intent and determine response strategy
            intent_info = await classify_query_intent(user_text, rag_results is not None and len(rag_results) > 0)
            logger.info(f"🎯 Intent classified: {intent_info}")

            # Step 3: Generate contextual response using Gemini
            response_text, sentiment, metadata = await generate_contextual_response(
                user_text=user_text,
                language_code=language,
                conversation_history=conversation,
                rag_results=rag_results
            )

            # Record metrics
            record_metric_intent(metadata.get("intent", "unknown"))
            record_metric_sentiment(sentiment)

            if "confidence" in metadata:
                record_metric_confidence(metadata["confidence"])

            # Add assistant response to conversation
            conversation.append(
                (
                    "assistant",
                    response_text,
                    metadata.get("intent"),
                    metadata.get("entities", {}),
                    sentiment,
                    time.time()
                )
            )

            # Send response to client
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

            # Generate and send TTS
            await send_tts_and_audio(response_text, language_code=language, sentiment=sentiment)

            # Record latency
            latency_ms = time.time() * 1000 - start_ms
            record_latency(latency_ms)
            logger.info(f"⏱️ Total response time: {latency_ms:.0f}ms")

        except Exception as e:
            logger.exception("Error in handle_final_utterance: %s", e)
            record_error("utterance_processing_failed")

            # Send error response
            error_msg = "I apologize, but I encountered an issue processing your request. Could you please try again?"
            try:
                await ws.send_text(json.dumps({
                    "type": "ai_text",
                    "text": error_msg,
                    "metadata": {"error": True}
                }))
                await send_tts_and_audio(error_msg, language_code=language, sentiment=0.0)
            except:
                pass

    async def debounce_and_handle():
        """Debounce user speech to detect utterance completion"""
        nonlocal pending_debounce_task, utterance_buffer, last_voice_ts

        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)

            # Check if new voice activity occurred during debounce
            if (time.time() - last_voice_ts) < (DEBOUNCE_SECONDS - 0.05):
                logger.debug("Debounce cancelled by new voice activity")
                return

            # Process buffered utterances
            text = " ".join(utterance_buffer).strip()
            utterance_buffer.clear()

            if not text:
                return

            await handle_final_utterance(text)

        finally:
            pending_debounce_task = None

    async def process_transcripts_task():
        """Process streaming STT transcripts"""
        nonlocal language, utterance_buffer, pending_debounce_task, is_bot_speaking
        nonlocal current_tts_task, current_stream_stop_event

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

                # Barge-in detection: User speaks while bot is speaking
                if interim_text and is_bot_speaking:
                    logger.info("🛑 Barge-in detected")

                    try:
                        await ws.send_text(json.dumps({
                            "type": "control",
                            "action": "stop_playback"
                        }))
                    except Exception:
                        pass

                    # Cancel current TTS
                    if current_tts_task and not current_tts_task.done():
                        try:
                            current_tts_task.cancel()
                        except Exception:
                            pass

                    # Stop streaming generation
                    if current_stream_stop_event:
                        try:
                            current_stream_stop_event.set()
                        except Exception:
                            pass

                    is_bot_speaking = False

                # Send interim results to client for UI display
                if interim_text:
                    try:
                        await ws.send_text(json.dumps({
                            "type": "transcript",
                            "text": interim_text,
                            "is_final": is_final
                        }))
                    except Exception:
                        pass

                # Process final transcripts
                if is_final and interim_text:
                    logger.info(f"📝 Final transcript: {interim_text}")
                    utterance_buffer.append(interim_text)

                    # Update language if detected
                    detected_lang = getattr(result, "language_code", None)
                    if detected_lang:
                        language = detected_lang

                    # Schedule debounce to process utterance
                    if not pending_debounce_task or pending_debounce_task.done():
                        pending_debounce_task = asyncio.create_task(debounce_and_handle())

    # Start STT thread
    transcript_consumer_task = None

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

        await ws.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
            "language": language
        }))

        logger.info(f"🎤 Session {session_id} ready")

        # ----------------------
        # Main WebSocket message loop (robust: supports text and binary frames)
        # ----------------------
        while True:
            msg = await ws.receive()  # returns dict with 'type' and optionally 'text' or 'bytes' keys

            # handle disconnect event
            if msg is None:
                logger.debug("ws.receive returned None, continuing loop")
                continue

            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                logger.info("Client websocket disconnected")
                break

            # Text frame (JSON control messages)
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
                            logger.info(f"Language change: {language} -> {new_lang}")
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
                                    logger.exception("Error stopping STT thread: %s", e)
                                # restart stt thread
                                stop_event = threading.Event()
                                stt_thread = threading.Thread(
                                    target=grpc_stt_worker,
                                    args=(loop, audio_queue, transcripts_queue, stop_event, language),
                                    daemon=True
                                )
                                stt_thread.start()
                                last_restart_ts = time.time()
                                logger.info(f"Restarted STT worker with language={language}")

                    try:
                        await ws.send_text(json.dumps({"type": "ack", "message": "started"}))
                    except Exception:
                        pass

                elif mtype == "audio":
                    # legacy base64 audio path
                    b64 = ctrl.get("payload")
                    if not b64:
                        continue
                    try:
                        pcm = base64.b64decode(b64)
                    except Exception:
                        try:
                            await ws.send_text(json.dumps({"type": "error", "error": "bad_audio_b64"}))
                        except:
                            pass
                        continue

                    try:
                        silent = is_silence(pcm)
                    except Exception:
                        silent = False

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
                    try:
                        stop_event.set()
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
                    # metrics sending code (unchanged)
                    session_duration = time.time() - session_start
                    try:
                        avg_response_time = (
                            sum(METRICS["response_times_ms"]) / len(METRICS["response_times_ms"])
                            if METRICS["response_times_ms"] else None
                        )
                        avg_sentiment = (
                            sum(METRICS["sentiments"]) / len(METRICS["sentiments"])
                            if METRICS["sentiments"] else None
                        )
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
                    try:
                        await ws.send_text(json.dumps({"type": "error", "error": "unknown_type"}))
                    except:
                        pass

            # Binary frame (raw audio from client)
            elif "bytes" in msg and msg["bytes"] is not None:
                pcm = msg["bytes"]
                try:
                    silent = is_silence(pcm)
                except Exception:
                    silent = False

                if not silent:
                    last_voice_ts = time.time()

                # queue with protection
                if audio_queue.qsize() > 350:
                    logger.warning(f"Audio queue large ({audio_queue.qsize()}), dropping input")
                    continue

                try:
                    audio_queue.put_nowait(pcm)
                except queue.Full:
                    logger.warning("Audio queue full, dropping chunk")
                    continue

            else:
                # unhandled message type - ignore but log
                logger.debug(f"Received unexpected websocket message: {msg}")
                continue

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
        record_error("websocket_error")
    finally:
        # Cleanup
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
        logger.info(f"HD WS cleanup complete. Session duration: {session_duration:.1f}s")
