
"""
Real-time WebSocket handler for multilingual voice AI
Handles: Twilio Media Streams → Google STT → Gemini → Google TTS → Twilio
Patched: queue.Queue for raw audio, synchronous STT consumer, aggregation option,
run_in_executor for blocking LLM/TTS, drop-oldest behavior when buffer full.
"""
import os
import json
import asyncio
import logging
import base64
import threading
import functools
import audioop
import queue
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions   # <-- ADDED

from app.services.gemini_service import GeminiService
from app.database import SessionLocal

# ====== Config & logging ======
logger = logging.getLogger(__name__)
router = APIRouter()

# Sanitize public URL
RENDER_PUBLIC_URL = os.getenv("RENDER_PUBLIC_URL", "").replace("https://", "").replace("http://", "").rstrip("/")

# Tuning (UPDATED DEFAULTS)
MAX_AUDIO_QUEUE = int(os.getenv("MAX_AUDIO_QUEUE", "50"))      # Reduced to 50 for low latency
EXECUTOR_WORKERS = int(os.getenv("EXECUTOR_WORKERS", "8"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15.0"))
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "12.0"))
AGGREGATE_FRAMES = int(os.getenv("AGGREGATE_FRAMES", "4"))      # Increased to 4 for stability

executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)

# ====== Inline audio utilities (no external dependency) ======
LANGUAGE_VOICE_MAP = {
    "en-US": {"name": "en-US-Neural2-A", "gender": "FEMALE"},
    "en-IN": {"name": "en-IN-Neural2-C", "gender": "FEMALE"},
    "hi-IN": {"name": "hi-IN-Neural2-A", "gender": "FEMALE"},
    "te-IN": {"name": "te-IN-Standard-A", "gender": "FEMALE"},
    "ta-IN": {"name": "ta-IN-Wavenet-A", "gender": "FEMALE"},
    "bn-IN": {"name": "bn-IN-Wavenet-A", "gender": "FEMALE"},
    "ml-IN": {"name": "ml-IN-Wavenet-A", "gender": "FEMALE"},
    "kn-IN": {"name": "kn-IN-Wavenet-A", "gender": "FEMALE"},
    "gu-IN": {"name": "gu-IN-Wavenet-A", "gender": "FEMALE"},
    "mr-IN": {"name": "mr-IN-Wavenet-A", "gender": "FEMALE"},
}
LANGUAGE_FALLBACK = {
    "en": "en-IN", "hi": "hi-IN", "te": "te-IN", "ta": "ta-IN",
    "bn": "bn-IN", "ml": "ml-IN", "kn": "kn-IN", "gu": "gu-IN", "mr": "mr-IN",
}

def get_best_voice(language_code: str) -> tuple:
    """
    Return (language_code, voice_name, gender_or_none).
    gender_or_none should be one of 'MALE', 'FEMALE' or None.
    If unknown, return None so we don't pass ssml_gender to Google.
    """
    # map voice name to its actual gender (adjust if you know a different mapping)
    LANGUAGE_VOICE_MAP = {
        "en-US": {"name": "en-US-Neural2-A", "gender": "FEMALE"},
        "en-IN": {"name": "en-IN-Neural2-C", "gender": "MALE"},   # <- NOTE: this voice is MALE
        "hi-IN": {"name": "hi-IN-Neural2-A", "gender": "FEMALE"},
        "te-IN": {"name": "te-IN-Standard-A", "gender": "FEMALE"},
        "ta-IN": {"name": "ta-IN-Wavenet-A", "gender": "FEMALE"},
        "bn-IN": {"name": "bn-IN-Wavenet-A", "gender": "FEMALE"},
        "ml-IN": {"name": "ml-IN-Wavenet-A", "gender": "FEMALE"},
        "kn-IN": {"name": "kn-IN-Wavenet-A", "gender": "FEMALE"},
        "gu-IN": {"name": "gu-IN-Wavenet-A", "gender": "FEMALE"},
        "mr-IN": {"name": "mr-IN-Wavenet-A", "gender": "FEMALE"},
    }

    LANGUAGE_FALLBACK = {
        "en": "en-IN", "hi": "hi-IN", "te": "te-IN", "ta": "ta-IN",
        "bn": "bn-IN", "ml": "ml-IN", "kn": "kn-IN", "gu": "gu-IN", "mr": "mr-IN",
    }

    if not language_code:
        v = LANGUAGE_VOICE_MAP["en-IN"]
        return ("en-IN", v["name"], v.get("gender"))

    if language_code in LANGUAGE_VOICE_MAP:
        v = LANGUAGE_VOICE_MAP[language_code]
        return (language_code, v["name"], v.get("gender"))

    base = language_code.split("-")[0] if "-" in language_code else language_code
    if base in LANGUAGE_FALLBACK:
        fallback = LANGUAGE_FALLBACK[base]
        v = LANGUAGE_VOICE_MAP.get(fallback, LANGUAGE_VOICE_MAP["en-IN"])
        return (fallback, v["name"], v.get("gender"))

    v = LANGUAGE_VOICE_MAP["en-IN"]
    logger.warning("No voice for %s, using en-IN", language_code)
    return ("en-IN", v["name"], v.get("gender"))

def twilio_payload_to_linear16(mu_law_b64: str) -> bytes:
    """Convert Twilio mu-law base64 to 16-bit PCM (LINEAR16) bytes."""
    try:
        mu_bytes = base64.b64decode(mu_law_b64)
        linear16 = audioop.ulaw2lin(mu_bytes, 2)
        return linear16
    except Exception as e:
        logger.exception("Audio conversion error: %s", e)
        return b""

# ====== Initialize AI & Google clients ======
_gemini = GeminiService()

GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON environment variable")

try:
    _gcreds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
    _speech_client = speech.SpeechClient(credentials=_gcreds)
    _tts_client = tts.TextToSpeechClient(credentials=_gcreds)
    logger.info("Google Cloud Speech/TTS clients initialized")
except Exception as e:
    logger.exception("Failed to initialize Google clients: %s", e)
    raise

ALTERNATIVE_LANGUAGES = [
    "en-IN", "hi-IN", "te-IN", "ta-IN", "bn-IN", "ml-IN", "kn-IN", "gu-IN", "mr-IN"
]
DEFAULT_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "DEFAULT_CLIENT")

def make_recognition_config(allow_alternatives: bool = True):
    """
    Build RecognitionConfig + StreamingRecognitionConfig.
    If allow_alternatives is False, don't include alternative_language_codes
    (some Google models reject that field).
    """
    kwargs = dict(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="phone_call",
        use_enhanced=True,
    )

    if allow_alternatives and ALTERNATIVE_LANGUAGES:
        # only set if allowed
        kwargs["alternative_language_codes"] = ALTERNATIVE_LANGUAGES

    config = speech.RecognitionConfig(**kwargs)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )
    return streaming_config

# ====== STT worker thread ======
def grpc_stt_worker(loop, audio_queue, transcripts_queue, stop_event):
    """
    STT worker that handles Google Speech-to-Text streaming recognition.
    Works with SpeechClient or SpeechHelpers and retries once without
    alternative_language_codes if the model rejects it (error occurs during iteration).
    """

    def gen_requests_with_config(config):
        # first request contains the streaming config, then audio chunks
        yield speech.StreamingRecognizeRequest(streaming_config=config)
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
                continue

    def gen_requests_audio_only():
        # for wrappers expecting (config, requests) signature: yield audio only
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
                continue

    def call_streaming_recognize_with_fallback(streaming_config):
        # Try standard Google client (generator includes config), fallback to wrapper signature
        try:
            return _speech_client.streaming_recognize(gen_requests_with_config(streaming_config))
        except TypeError as e:
            msg = str(e).lower()
            if "missing" in msg and ("requests" in msg or "request" in msg):
                logger.info("Detected SpeechHelpers signature; calling streaming_recognize(config, requests)")
                return _speech_client.streaming_recognize(streaming_config, gen_requests_audio_only())
            raise

    def consume_responses_with_retry(streaming_config, allow_alternatives=True):
        """
        Call streaming_recognize and consume responses.
        If InvalidArgument about alternative_language_codes appears during iteration,
        retry once with allow_alternatives=False.
        """
        try:
            responses = call_streaming_recognize_with_fallback(streaming_config)
            for response in responses:
                if stop_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(transcripts_queue.put(response), loop)
        except google_exceptions.InvalidArgument as e:
            err = str(e)
            if "alternative_language_codes" in err and allow_alternatives:
                logger.warning("Model rejected alternative_language_codes during streaming; retrying without them...")
                new_config = make_recognition_config(allow_alternatives=False)
                # Retry once without alternatives
                consume_responses_with_retry(new_config, allow_alternatives=False)
            else:
                raise

    try:
        logger.info("🎤 Starting STT stream (thread)")
        starting_config = make_recognition_config(allow_alternatives=True)
        consume_responses_with_retry(starting_config, allow_alternatives=True)
    except Exception as e:
        logger.exception("❌ STT worker error: %s", e)
    finally:
        try:
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(None), loop)
        except Exception:
            pass
        logger.info("🎤 STT stream ended")
async def get_ai_response(transcript: str, language_code: str) -> str:
    db = SessionLocal()
    try:
        try:
            from app.services.pinecone_service import pinecone_service
            results = await pinecone_service.search_similar_chunks(
                client_id=DEFAULT_CLIENT_ID,
                query=transcript,
                top_k=2
            )
            context = "\n\n".join([r.get("chunk_text", "") for r in results]) if results else ""
        except Exception as e:
            logger.debug("Pinecone context fetch failed: %s", e)
            context = ""

        system_msg = f"You are a helpful voice assistant. Respond in {language_code}. Keep responses concise for voice."
        if context:
            prompt = f"Context:\n{context}\n\nUser ({language_code}): {transcript}\n\nRespond naturally in {language_code}:"
        else:
            prompt = f"User ({language_code}): {transcript}\n\nRespond in {language_code}:"

        loop = asyncio.get_running_loop()
        llm_partial = functools.partial(
            _gemini.generate_response,
            prompt=prompt,
            temperature=0.7,
            max_tokens=200,
            system_message=system_msg
        )

        # Optional: short-circuit Gemini during tests (set USE_GEMINI=false in env)
        if os.getenv("USE_GEMINI", "true").lower() != "true":
            return f"I heard: {transcript}"

        try:
            response = await asyncio.wait_for(loop.run_in_executor(executor, llm_partial), timeout=LLM_TIMEOUT)
            if response:
                return response
            # empty response -> fallback
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out")
            return "Sorry, I'm taking too long to think. Please try again."
        except Exception as e:
            logger.warning("Primary LLM failed: %s — falling back to simple reply", e)
            return f"I heard: {transcript}"
    finally:
        db.close()

async def synthesize_and_send(ws: WebSocket, text: str, language_code: str):
    if not text:
        return
    try:
        lang_code, voice_name, gender = get_best_voice(language_code)
        logger.info("TTS request: lang=%s voice=%s text=%s", lang_code, voice_name, text[:60])

        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
            ssml_gender=getattr(tts.SsmlVoiceGender, gender)
        )
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MULAW, sample_rate_hertz=8000)

        def tts_call():
            return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(loop.run_in_executor(executor, tts_call), timeout=TTS_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("TTS timed out")
            return
        except Exception as e:
            logger.exception("TTS exec error: %s", e)
            return

        mulaw_b64 = base64.b64encode(resp.audio_content).decode("ascii")

        try:
            await ws.send_json({"event": "clear"})
        except Exception:
            logger.debug("Failed to send clear event")

        await ws.send_json({"event": "media", "media": {"payload": mulaw_b64}})
        await ws.send_json({"event": "mark", "streamSid": "tts_end"})
        logger.info("TTS audio sent to Twilio")

    except Exception as e:
        logger.exception("synthesize_and_send error: %s", e)

# ====== Adaptive Rate Limiter Class (ADDED) ======
class AdaptiveRateLimiter:
    """Skip audio chunks when queue is filling up"""
    def __init__(self, queue, threshold=0.6):
        self.queue = queue
        self.threshold = threshold

    def should_enqueue(self) -> bool:
        try:
            return (self.queue.qsize() / self.queue.maxsize) < self.threshold
        except:
            return True

# ====== WebSocket handler ======
@router.websocket("/media-stream")
async def handle_media_stream(ws: WebSocket):
    await ws.accept()
    call_sid = "unknown"
    logger.info("WebSocket accepted")

    audio_queue = queue.Queue(maxsize=MAX_AUDIO_QUEUE)
    transcripts_queue = asyncio.Queue()
    stop_event = threading.Event()
    stt_thread = None

    is_bot_speaking = False
    detected_language = "en-IN"

    try:
        loop = asyncio.get_event_loop()
        stt_thread = threading.Thread(
            target=grpc_stt_worker,
            args=(loop, audio_queue, transcripts_queue, stop_event),
            daemon=True
        )
        stt_thread.start()

        async def process_transcripts():
            nonlocal is_bot_speaking, detected_language
            while True:
                resp = await transcripts_queue.get()
                if resp is None:
                    break
                for result in resp.results:
                    if not result.alternatives:
                        continue
                    alt = result.alternatives[0]
                    transcript = alt.transcript.strip()
                    is_final = result.is_final
                    lang = getattr(result, "language_code", None) or detected_language

                    logger.info("%s transcript (lang=%s): %s", "FINAL" if is_final else "interim", lang, transcript[:120])

                    if not is_final and is_bot_speaking and len(transcript) > 3:
                        logger.info("Barge-in detected, clearing playback")
                        try:
                            await ws.send_json({"event": "clear"})
                        except Exception:
                            pass
                        is_bot_speaking = False

                    if is_final and len(transcript) > 0:
                        detected_language = lang
                        logger.info("Processing final transcript (lang=%s): %s", detected_language, transcript[:200])
                        is_bot_speaking = True
                        try:
                            ai_resp = await get_ai_response(transcript, detected_language)
                            await synthesize_and_send(ws, ai_resp, detected_language)
                        finally:
                            is_bot_speaking = False

        transcript_task = asyncio.create_task(process_transcripts())

        while True:
            try:
                msg = await ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "start":
                    call_sid = data.get("start", {}).get("callSid", "unknown")
                    logger.info("Call started: %s", call_sid)

                elif event == "media":
                    payload = data.get("media", {}).get("payload")
                    if not payload:
                        continue

                    linear16 = twilio_payload_to_linear16(payload)
                    if not linear16:
                        continue

                    # init rate limiter once
                    if not hasattr(ws, "_rate_limiter"):
                        ws._rate_limiter = AdaptiveRateLimiter(audio_queue)

                    # skip audio if queue is filling up
                    if not ws._rate_limiter.should_enqueue():
                        continue

                    # aggregation
                    if not hasattr(ws, "_agg_acc"):
                        ws._agg_acc = bytearray()
                        ws._agg_cnt = 0

                    ws._agg_acc.extend(linear16)
                    ws._agg_cnt += 1

                    if ws._agg_cnt < AGGREGATE_FRAMES:
                        continue

                    chunk = bytes(ws._agg_acc)
                    ws._agg_acc.clear()
                    ws._agg_cnt = 0

                    try:
                        audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass  # silently drop

                elif event == "stop":
                    logger.info("Call ended: %s", call_sid)
                    break

                elif event == "mark":
                    is_bot_speaking = False

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected: %s", call_sid)
                break
            except Exception as e:
                logger.exception("WS receive loop error: %s", e)
                break

    except Exception as e:
        logger.exception("Websocket handler top-level error: %s", e)

    finally:
        logger.info("Cleaning up call: %s", call_sid)
        stop_event.set()
        try:
            audio_queue.put_nowait(None)
        except Exception:
            pass
        try:
            await transcripts_queue.put(None)
        except Exception:
            pass
        if stt_thread:
            stt_thread.join(timeout=3.0)
        try:
            transcript_task.cancel()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Cleanup complete for call: %s", call_sid)
