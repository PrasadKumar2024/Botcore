# app/routes/hd_audio_ws.py
import os
import io
import json
import time
import base64
import logging
import functools
import asyncio
import audioop
import wave
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions

# Local services in your repo
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
router = APIRouter()

# ----------------- Configuration -----------------
# environment-tunable parameters
EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "6"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "3"))         # concurrent TTS requests globally
STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "10.0"))       # seconds
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "12.0"))       # seconds
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "12.0"))       # seconds
STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))         # Hz — frontend must send this
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "1.0"))    # aggregate ~1s before STT
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "10"))   # max queued per-connection seconds
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)      # simple auth token (required)

# derived
BYTES_PER_SEC = STT_SAMPLE_RATE * 2   # 16-bit PCM mono
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

# executor for blocking IO (STT/TTS/LLM)
executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)

# optional global semaphore to prevent runaway TTS/LLM concurrency
global_tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)

# ----------------- Initialize Google Clients -----------------
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON environment variable")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = tts.TextToSpeechClient(credentials=_creds)

# ----------------- LLM -----------------
_gemini = GeminiService()

# ----------------- Voice map & SSML helpers -----------------
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
    fallback = {
        "en": "en-IN",
        "hi": "hi-IN",
    }
    if base in fallback:
        f = fallback[base]
        v = VOICE_MAP.get(f, DEFAULT_VOICE)
        return (f, v["name"], v.get("gender"))
    return ("en-IN", DEFAULT_VOICE["name"], DEFAULT_VOICE.get("gender"))

def ssml_for_text(text: str, prosody_rate: float = 0.98) -> str:
    # minimal escaping
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc = esc.replace(". ", ". <break time='250ms'/> ")
    esc = esc.replace("? ", "? <break time='250ms'/> ")
    esc = esc.replace("! ", "! <break time='250ms'/> ")
    return f"<speak><prosody rate='{prosody_rate}'>{esc}</prosody></speak>"

def make_wav_from_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

# ----------------- Utility: simple VAD (RMS) -----------------
def is_silence(pcm16: bytes, threshold: int = 300) -> bool:
    try:
        rms = audioop.rms(pcm16, 2)
        return rms < threshold
    except Exception:
        return False

# ----------------- Blocking service wrappers (run in executor) -----------------
def _sync_stt_recognize_bytes(pcm16_bytes: bytes, sample_rate: int, language_code: str):
    """
    Blocking wrapper for Speech-to-Text 'recognize' suitable for short chunks.
    Expects raw PCM16LE bytes.
    """
    audio = speech.RecognitionAudio(content=pcm16_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        enable_automatic_punctuation=True,
        model="default",
    )
    return _speech_client.recognize(config=config, audio=audio, timeout=STT_TIMEOUT)

async def stt_recognize_pcm16(pcm16_bytes: bytes, sample_rate: int = STT_SAMPLE_RATE, language_code: str = "en-IN") -> Optional[str]:
    loop = asyncio.get_running_loop()
    try:
        # Run STT in executor with timeout guard
        fut = loop.run_in_executor(executor, functools.partial(_sync_stt_recognize_bytes, pcm16_bytes, sample_rate, language_code))
        resp = await asyncio.wait_for(fut, timeout=STT_TIMEOUT)
        if not resp.results:
            return None
        transcripts = []
        for r in resp.results:
            if r.alternatives:
                transcripts.append(r.alternatives[0].transcript)
        return " ".join(transcripts).strip() if transcripts else None
    except asyncio.TimeoutError:
        logger.warning("STT timed out")
        return None
    except google_exceptions.GoogleAPICallError as e:
        logger.exception("STT API error: %s", e)
        return None
    except Exception as e:
        logger.exception("STT unexpected error: %s", e)
        return None

def _sync_tts_linear16(ssml: str, language_code: str, voice_name: str, gender: Optional[str], sample_rate_hz: int = 24000):
    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
    # Some client libraries accept effects_profile_id in AudioConfig; test & include only if available
    audio_cfg_kwargs = {"audio_encoding": tts.AudioEncoding.LINEAR16, "sample_rate_hertz": sample_rate_hz}
    # effects_profile_id mostly helps telephony; safe to include for Google TTS
    audio_cfg_kwargs["effects_profile_id"] = ["telephony-class-application"]
    audio_config = tts.AudioConfig(**audio_cfg_kwargs)
    synthesis_input = tts.SynthesisInput(ssml=ssml)
    return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

async def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", sample_rate_hz: int = 24000) -> Optional[bytes]:
    """
    Generate LINEAR16 PCM bytes at sample_rate_hz using Google TTS.
    Runs TTS in executor with timeout and concurrency limit.
    """
    ssml = ssml_for_text(text)
    lang_code, voice_name, gender = get_best_voice(language_code)
    loop = asyncio.get_running_loop()

    # Acquire global TTS semaphore to avoid burst costs
    try:
        await asyncio.wait_for(global_tts_semaphore.acquire(), timeout=3.0)
    except Exception:
        logger.warning("TTS semaphore busy; rejecting or delaying TTS")
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

# ----------------- RAG + LLM wrapper -----------------
async def get_ai_text_response(transcript: str, language_code: str = "en-IN") -> str:
    """
    Query Pinecone for KB, if found run RDF-style assistant (temp 0), else run conversational fallback (temp 0.6)
    Runs LLM in executor with timeout.
    """
    loop = asyncio.get_running_loop()
    try:
        q = transcript.strip()
        # small normalization (expand common terms)
        q = q.replace("timings", "business hours").replace("timing", "business hours")

        results = await pinecone_service.search_similar_chunks(
            client_id=os.getenv("DEFAULT_KB_CLIENT_ID", "default"),
            query=q,
            top_k=4,
            min_score=-1.0
        )

        if results:
            context_text = "\n\n".join([r.get("chunk_text", "") for r in results[:3]])
            system_msg = ("You are a helpful assistant. Use ONLY the context to answer. Keep responses brief and conversational.")
            user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {transcript}"
            partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=system_msg, temperature=0.0, max_tokens=160)
            fut = loop.run_in_executor(executor, partial)
            try:
                resp = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
                return resp.strip() if resp else "Sorry — I couldn't formulate a response from the records."
            except asyncio.TimeoutError:
                logger.warning("LLM RAG call timed out")
                return "Sorry, I couldn't fetch details right now."
        # fallback conversational LLM
        conv_sys = "You are a friendly assistant. Keep responses natural and brief."
        user_prompt = f"User said: {transcript}\nRespond naturally and briefly."
        partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=conv_sys, temperature=0.6, max_tokens=160)
        fut = loop.run_in_executor(executor, partial)
        try:
            conv = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
            return conv.strip() if conv else "Sorry, I didn't catch that — can you repeat?"
        except asyncio.TimeoutError:
            logger.warning("LLM conversational fallback timed out")
            return "Sorry, I'm having trouble answering right now."
    except Exception as e:
        logger.exception("get_ai_text_response error: %s", e)
        return "I'm sorry, I can't access that information right now."

# ----------------- WebSocket Handler -----------------
@router.websocket("/ws/hd-audio")
async def hd_audio_ws(ws: WebSocket):
    # simple query param token auth
    # client should connect with ws://.../ws/hd-audio?token=XYZ
    token = ws.query_params.get("token")
    if WEBSOCKET_API_TOKEN:
        if not token or token != WEBSOCKET_API_TOKEN:
            await ws.accept()
            await ws.send_text(json.dumps({"type": "error", "error": "unauthorized"}))
            await ws.close()
            logger.warning("WS rejected unauthenticated connection")
            return

    await ws.accept()
    logger.info("HD-WS accepted connection")
    audio_buffer = bytearray()
    language = "en-IN"
    client_active = True

    # per-connection concurrency limiter: don't spawn infinite tasks
    conn_semaphore = asyncio.BoundedSemaphore(2)

    async def spawn_task_safe(coro):
        try:
            await conn_semaphore.acquire()
            asyncio.create_task(_run_and_release(coro, conn_semaphore))
        except Exception as e:
            logger.exception("spawn_task_safe error: %s", e)

    async def _run_and_release(coro, sem):
        try:
            await coro
        finally:
            try:
                sem.release()
            except Exception:
                pass

    try:
        await ws.send_text(json.dumps({"type": "ready"}))
        last_activity = time.time()

        while True:
            data_text = await ws.receive_text()
            last_activity = time.time()
            try:
                msg = json.loads(data_text)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            mtype = msg.get("type")
            if mtype == "start":
                meta = msg.get("meta", {}) or {}
                language = meta.get("language", language) or language
                await ws.send_text(json.dumps({"type": "ack", "message": "started"}))

            elif mtype == "audio":
                # Expect base64 of PCM16LE @ STT_SAMPLE_RATE mono
                b64 = msg.get("payload")
                if not b64:
                    continue
                try:
                    pcm = base64.b64decode(b64)
                except Exception:
                    await ws.send_text(json.dumps({"type": "error", "error": "bad_audio_b64"}))
                    continue

                # validate size and append
                if len(audio_buffer) + len(pcm) > MAX_BUFFER_BYTES:
                    # backpressure: drop oldest or reject
                    # here we drop oldest to keep continuity
                    drop = len(audio_buffer) + len(pcm) - MAX_BUFFER_BYTES
                    del audio_buffer[:drop]
                    logger.warning("Dropping %d bytes to prevent buffer overflow", drop)

                audio_buffer.extend(pcm)

                # If we accumulated at least CHUNK_BYTES, process approx 1s
                if len(audio_buffer) >= CHUNK_BYTES:
                    chunk = bytes(audio_buffer[:CHUNK_BYTES])
                    del audio_buffer[:CHUNK_BYTES]

                    # optional VAD: skip silent chunks to save quota
                    if is_silence(chunk, threshold=300):
                        logger.debug("Skipping silent chunk (VAD)")
                        continue

                    # process in background, bounded per-connection by conn_semaphore
                    await spawn_task_safe(process_audio_chunk_and_respond(ws, chunk, language))

            elif mtype == "stop":
                # flush remaining buffer
                if audio_buffer:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()
                    await process_audio_chunk_and_respond(ws, chunk, language)
                await ws.send_text(json.dumps({"type": "bye"}))
                await ws.close()
                return

            else:
                await ws.send_text(json.dumps({"type": "error", "error": "unknown_type"}))

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("HD WS cleanup complete")

# ----------------- processing task -----------------
async def process_audio_chunk_and_respond(ws: WebSocket, pcm_chunk: bytes, language_code: str):
    """
    Steps:
     1) STT (Google recognize) on the PCM chunk
     2) RAG-first LLM response (via pinecone + Gemini)
     3) TTS (Google) -> LINEAR16 PCM @ 24k
     4) Wrap WAV and send base64 to client (type 'audio')
    """
    try:
        await ws.send_text(json.dumps({"type": "processing", "ts": time.time()}))

        # 1) STT
        transcript = await stt_recognize_pcm16(pcm_chunk, sample_rate=STT_SAMPLE_RATE, language_code=language_code)
        if not transcript:
            # nothing recognized; report partial
            await ws.send_text(json.dumps({"type": "transcript", "text": ""}))
            return
        await ws.send_text(json.dumps({"type": "transcript", "text": transcript}))

        # 2) Generate AI text (RAG-first + LLM fallback)
        ai_text = await get_ai_text_response(transcript, language_code=language_code)
        await ws.send_text(json.dumps({"type": "ai_text", "text": ai_text}))

        # 3) TTS synthesize (24k)
        tts_pcm = await synthesize_text_to_pcm(ai_text, language_code=language_code, sample_rate_hz=24000)
        if not tts_pcm:
            await ws.send_text(json.dumps({"type": "error", "error": "tts_failed"}))
            return

        # 4) Create WAV and send back as base64 (client will decode and play)
        wav_bytes = make_wav_from_pcm16(tts_pcm, sample_rate=24000)
        b64wav = base64.b64encode(wav_bytes).decode("ascii")
        await ws.send_text(json.dumps({"type": "audio", "audio": b64wav}))

    except Exception as e:
        logger.exception("process_audio_chunk error: %s", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "error": "processing_failed"}))
        except Exception:
            pass
