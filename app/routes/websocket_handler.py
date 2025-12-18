"""
Real-time WebSocket handler for multilingual voice AI
Handles: Twilio Media Streams → Google STT → Gemini → Google TTS → Twilio
Patched: run_in_executor for blocking calls, larger queue, drop-oldest behavior.
"""
import os
import json
import asyncio
import logging
import base64
import threading
import functools
import audioop
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account

from app.services.gemini_service import GeminiService
from app.database import SessionLocal

# ====== Config & logging ======
logger = logging.getLogger(__name__)
router = APIRouter()

# Sanitize public URL (harmless if unused here; recommended to mirror in twilio_voice.py)
RENDER_PUBLIC_URL = os.getenv("RENDER_PUBLIC_URL", "").replace("https://", "").replace("http://", "").rstrip("/")

# Tuning (env overrideable)
MAX_AUDIO_QUEUE = int(os.getenv("MAX_AUDIO_QUEUE", "300"))      # number of audio chunks to buffer
EXECUTOR_WORKERS = int(os.getenv("EXECUTOR_WORKERS", "8"))      # threads for TTS/LLM
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15.0"))           # seconds
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "12.0"))           # seconds

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
    """Return (language_code, voice_name, gender) for TTS selection."""
    if not language_code:
        return ("en-IN", LANGUAGE_VOICE_MAP["en-IN"]["name"], LANGUAGE_VOICE_MAP["en-IN"]["gender"])
    if language_code in LANGUAGE_VOICE_MAP:
        v = LANGUAGE_VOICE_MAP[language_code]
        return (language_code, v["name"], v["gender"])
    base = language_code.split("-")[0] if "-" in language_code else language_code
    if base in LANGUAGE_FALLBACK:
        fallback = LANGUAGE_FALLBACK[base]
        v = LANGUAGE_VOICE_MAP[fallback]
        return (fallback, v["name"], v["gender"])
    v = LANGUAGE_VOICE_MAP["en-IN"]
    logger.warning("No voice for %s, using en-IN", language_code)
    return ("en-IN", v["name"], v["gender"])

def twilio_payload_to_linear16(mu_law_b64: str) -> bytes:
    """Convert Twilio mu-law base64 to 16-bit PCM (LINEAR16) bytes."""
    try:
        mu_bytes = base64.b64decode(mu_law_b64)
        linear16 = audioop.ulaw2lin(mu_bytes, 2)  # width=2 (16-bit)
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

# Preferred languages for recognition (tweak as needed)
ALTERNATIVE_LANGUAGES = [
    "en-IN", "hi-IN", "te-IN", "ta-IN", "bn-IN", "ml-IN", "kn-IN", "gu-IN", "mr-IN"
]
DEFAULT_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "DEFAULT_CLIENT")

def make_recognition_config():
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US",
        alternative_language_codes=ALTERNATIVE_LANGUAGES,
        enable_automatic_punctuation=True,
        model="phone_call",
        use_enhanced=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )
    return streaming_config

# ====== STT worker thread (unchanged pattern, feeds transcripts to async queue) ======
def grpc_stt_worker(loop, audio_queue: asyncio.Queue, transcripts_queue: asyncio.Queue, stop_event: threading.Event):
    def gen_requests():
        streaming_config = make_recognition_config()
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        while not stop_event.is_set():
            try:
                chunk = asyncio.run_coroutine_threadsafe(audio_queue.get(), loop).result(timeout=0.5)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except Exception:
                if stop_event.is_set():
                    break
                continue

    try:
        logger.info("Starting STT gRPC streaming worker")
        responses = _speech_client.streaming_recognize(requests=gen_requests())
        for resp in responses:
            if stop_event.is_set():
                break
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(resp), loop)
    except Exception as e:
        logger.exception("STT worker error: %s", e)
    finally:
        try:
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(None), loop)
        except Exception:
            pass
        logger.info("STT worker exiting")

# ====== Non-blocking LLM (Gemini) call ======
async def get_ai_response(transcript: str, language_code: str) -> str:
    """Get AI response using RAG + Gemini without blocking the event loop."""
    db = SessionLocal()
    try:
        # Get RAG context asynchronously (pinecone_service is async)
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

        try:
            response = await asyncio.wait_for(loop.run_in_executor(executor, llm_partial), timeout=LLM_TIMEOUT)
            return response or "I apologize, I couldn't process that request."
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out")
            return "Sorry, I'm taking too long to think. Please try again."
        except Exception as e:
            logger.exception("LLM execution error: %s", e)
            return "Sorry, I'm having trouble right now. Please try again."
    finally:
        db.close()

# ====== Non-blocking TTS (Google) and streaming to Twilio ======
async def synthesize_and_send(ws: WebSocket, text: str, language_code: str):
    """Convert text to speech (in executor) and send mu-law base64 to Twilio via WS."""
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

        # Barge-in protection: clear playing audio first
        try:
            await ws.send_json({"event": "clear"})
        except Exception:
            logger.debug("Failed to send clear event (non-fatal)")

        await ws.send_json({"event": "media", "media": {"payload": mulaw_b64}})
        await ws.send_json({"event": "mark", "streamSid": "tts_end"})
        logger.info("TTS audio sent to Twilio (len=%d bytes)", len(resp.audio_content))

    except Exception as e:
        logger.exception("synthesize_and_send error: %s", e)

# ====== WebSocket handler ======
@router.websocket("/media-stream")
async def handle_media_stream(ws: WebSocket):
    await ws.accept()
    call_sid = "unknown"
    logger.info("WebSocket accepted")

    audio_queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE)
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

                    # Barge-in: if interim user speech while bot speaking -> interrupt bot
                    if not is_final and is_bot_speaking and len(transcript) > 3:
                        logger.info("Barge-in detected, clearing playback")
                        try:
                            await ws.send_json({"event": "clear"})
                        except Exception:
                            logger.debug("Failed to send clear event during barge-in")
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

        # receive loop
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

                    # Robust enqueue: drop oldest when full to keep pipeline moving
                    try:
                        audio_queue.put_nowait(linear16)
                    except asyncio.QueueFull:
                        logger.warning("Audio queue full - dropping oldest chunk and enqueueing new chunk")
                        try:
                            audio_queue.get_nowait()  # drop oldest
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            audio_queue.put_nowait(linear16)
                        except asyncio.QueueFull:
                            logger.warning("Audio queue still full after drop; dropping incoming chunk")

                elif event == "stop":
                    logger.info("Call ended: %s", call_sid)
                    break

                elif event == "mark":
                    # Twilio mark (playback ended)
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
            await audio_queue.put(None)
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
