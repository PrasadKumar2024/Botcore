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
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

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

# -------------- Config --------------
EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "6"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "3"))
STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "10.0"))
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "12.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "12.0"))

# Must match frontend (we recommend 16000)
STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "0.35"))  # smaller for responsive streaming
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "10"))
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)

# Use the client id env var you set
DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", os.getenv("DEFAULT_KB_CLIENT_ID", "default"))

BYTES_PER_SEC = STT_SAMPLE_RATE * 2
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
global_tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)

# -------------- Google clients --------------
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = tts.TextToSpeechClient(credentials=_creds)

# -------------- LLM --------------
_gemini = GeminiService()

# -------------- Utilities: voices / ssml / wav --------------
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

def ssml_for_text(text: str, prosody_rate: float = 0.98) -> str:
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc = esc.replace(". ", ". <break time='200ms'/> ")
    esc = esc.replace("? ", "? <break time='200ms'/> ")
    esc = esc.replace("! ", "! <break time='200ms'/> ")
    return f"<speak><prosody rate='{prosody_rate}'>{esc}</prosody></speak>"

def make_wav_from_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def is_silence(pcm16: bytes, threshold: int = 300) -> bool:
    try:
        return audioop.rms(pcm16, 2) < threshold
    except Exception:
        return False

# ----------------- Blocking wrappers -----------------
def _sync_tts_linear16(ssml: str, language_code: str, voice_name: str, gender: Optional[str], sample_rate_hz: int = 24000):
    # IMPORTANT: do NOT include telephony profile for browser usage
    voice = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_cfg_kwargs = {"audio_encoding": tts.AudioEncoding.LINEAR16, "sample_rate_hertz": sample_rate_hz}
    audio_config = tts.AudioConfig(**audio_cfg_kwargs)
    synthesis_input = tts.SynthesisInput(ssml=ssml)
    return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

async def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", sample_rate_hz: int = 24000) -> Optional[bytes]:
    ssml = ssml_for_text(text)
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

# ----------------- Query normalization (from PSTN) -----------------
def normalize_and_expand_query(transcript: str) -> str:
    if not transcript:
        return ""
    s = transcript.lower().strip()
    toks = s.split()
    dedup = [toks[0]] if toks else []
    for i in range(1, len(toks)):
        if toks[i] != toks[i-1]:
            dedup.append(toks[i])
    s = " ".join(dedup)
    mappings = {
        "timings": "business hours operating hours schedule",
        "timing": "business hours operating hours schedule",
        "whats": "what are",
        "what's": "what are",
        "when's": "when are",
        "where's": "where is",
        "phone number": "contact number telephone",
        "open": "business hours operating hours",
        "closed": "business hours operating hours",
        "appointment": "appointment booking consultation",
        "doctor": "doctor physician consultant",
        "payments": "payment methods accepted cash upi card",
    }
    out = []
    for w in s.split():
        out.append(w)
        if w in mappings:
            out.extend(mappings[w].split())
    return " ".join(out)

# ----------------- Streaming STT worker (thread) -----------------
def grpc_stt_worker(loop, audio_queue: queue.Queue, transcripts_queue: asyncio.Queue, stop_event: threading.Event, language_code: str):
    """
    Thread that maintains a long-lived Google streaming_recognize session.
    Pulls raw PCM16 bytes from audio_queue and yields them to Google.
    Puts Google responses (StreamingRecognizeResponse) into transcripts_queue.
    IMPORTANT: this worker does NOT put a termination sentinel into transcripts_queue.
    Main thread will handle final sentinel on overall shutdown.
    """
    def gen_requests():
        # first request = config
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True, single_utterance=False)
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        # then audio chunks
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
                if chunk is None:
                    # explicit stream end requested
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error in STT gen_requests: %s", e)
                if stop_event.is_set():
                    break

    try:
        logger.info("Starting Google streaming STT worker (thread) language=%s", language_code)
        responses = _speech_client.streaming_recognize(gen_requests())
        for response in responses:
            if stop_event.is_set():
                break
            # push the whole response object to async queue to be handled in event loop
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(response), loop)
    except Exception as e:
        logger.exception("STT worker error: %s", e)
    finally:
        logger.info("STT worker exiting (language=%s)", language_code)
        # DO NOT put None here — main controls termination to support restarts.

# ----------------- RAG + LLM (uses DEFAULT_CLIENT_ID) -----------------
async def get_ai_text_response(transcript: str, language_code: str = "en-IN") -> str:
    loop = asyncio.get_running_loop()
    try:
        q = transcript.strip()
        q = q.replace("timings", "business hours").replace("timing", "business hours")
        norm_q = normalize_and_expand_query(q)

        # try normalized query first
        results = await pinecone_service.search_similar_chunks(
            client_id=DEFAULT_CLIENT_ID,
            query=norm_q or q,
            top_k=4,
            min_score=-1.0
        )
        if not results:
            # try raw transcript (fallback)
            results = await pinecone_service.search_similar_chunks(
                client_id=DEFAULT_CLIENT_ID,
                query=q,
                top_k=4,
                min_score=-1.0
            )

        if results:
            # build context and call LLM deterministically
            context_text = "\n\n".join([r.get("chunk_text", "") for r in results[:3]])
            system_msg = ("You are a helpful assistant. Use ONLY the context to answer. Keep responses brief and conversational.")
            user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {transcript}"
            partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=system_msg, temperature=0.0, max_tokens=160)
            fut = loop.run_in_executor(executor, partial)
            try:
                resp = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
                return resp.strip() if resp else "Sorry — I couldn't formulate a response from the records."
            except asyncio.TimeoutError:
                logger.warning("LLM RAG timed out")
                return "Sorry, I couldn't fetch details right now."

        # fallback to conversational LLM
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

    # queue for streaming STT (thread-safe)
    audio_queue = queue.Queue(maxsize=400)  # backpressure if client floods
    transcripts_queue = asyncio.Queue()
    stop_event = threading.Event()

    language = "en-IN"
    stt_thread = None

    # VAD / silence detection settings
    SILENCE_TIMEOUT = float(os.getenv("HD_WS_SILENCE_TIMEOUT", "0.6"))  # seconds
    last_voice_ts = time.time()
    restarting_lock = threading.Lock()  # avoid concurrent restarts

    async def process_transcripts_task():
        """
        Consume streaming responses from Google (placed into transcripts_queue by the STT thread).
        For each final result -> run RAG -> TTS -> send back WAV base64.
        """
        nonlocal language
        while True:
            resp = await transcripts_queue.get()
            if resp is None:
                logger.info("Transcript consumer received sentinel; exiting")
                break
            # resp is a StreamingRecognizeResponse
            for result in resp.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                interim_text = alt.transcript.strip()
                is_final = getattr(result, "is_final", False)
                if interim_text:
                    # send interim for UI
                    try:
                        await ws.send_text(json.dumps({"type":"transcript", "text": interim_text, "is_final": is_final}))
                    except Exception:
                        pass

                if is_final and interim_text:
                    logger.info("Final transcript (lang=%s): %s", getattr(result, "language_code", language), interim_text[:200])
                    # update language if Google provides it
                    language = getattr(result, "language_code", language) or language

                    # RAG + LLM
                    try:
                        ai_text = await get_ai_text_response(interim_text, language_code=language)
                        await ws.send_text(json.dumps({"type":"ai_text", "text": ai_text}))
                    except Exception as e:
                        logger.exception("RAG call failed: %s", e)
                        ai_text = "Sorry, I'm having trouble fetching information."

                    # TTS -> send WAV base64
                    try:
                        tts_pcm = await synthesize_text_to_pcm(ai_text, language_code=language, sample_rate_hz=24000)
                        if tts_pcm:
                            wav_bytes = make_wav_from_pcm16(tts_pcm, sample_rate=24000)
                            b64wav = base64.b64encode(wav_bytes).decode("ascii")
                            await ws.send_text(json.dumps({"type":"audio", "audio": b64wav}))
                        else:
                            await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
                    except Exception as e:
                        logger.exception("TTS/send failed: %s", e)
                        try:
                            await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
                        except:
                            pass

    # start transcript consumer task placeholder
    transcript_consumer_task = None

    try:
        # start STT thread (uses audio_queue)
        loop = asyncio.get_event_loop()
        stop_event.clear()
        stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
        stt_thread.start()

        transcript_consumer_task = asyncio.create_task(process_transcripts_task())

        await ws.send_text(json.dumps({"type":"ready"}))

        while True:
            data_text = await ws.receive_text()
            try:
                msg = json.loads(data_text)
            except Exception:
                await ws.send_text(json.dumps({"type":"error","error":"invalid_json"}))
                continue

            mtype = msg.get("type")
            if mtype == "start":
                meta = msg.get("meta", {}) or {}
                new_lang = meta.get("language")
                # If language changed, restart STT thread with new language
                if new_lang and new_lang != language:
                    logger.info("Language change requested: %s -> %s", language, new_lang)
                    language = new_lang
                    # graceful restart: signal worker to stop and wait, then start new worker
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
                        # start new worker
                        stop_event = threading.Event()
                        stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
                        stt_thread.start()
                        logger.info("Restarted STT worker with language=%s", language)
                await ws.send_text(json.dumps({"type":"ack","message":"started"}))

            elif mtype == "audio":
                b64 = msg.get("payload")
                if not b64:
                    continue
                try:
                    pcm = base64.b64decode(b64)
                except Exception:
                    await ws.send_text(json.dumps({"type":"error","error":"bad_audio_b64"}))
                    continue

                # VAD: check if chunk is silent
                try:
                    silent = is_silence(pcm, threshold=int(os.getenv("HD_WS_VAD_THRESHOLD", "300")))
                except Exception:
                    silent = False

                now = time.time()
                if not silent:
                    last_voice_ts = now

                # push chunk to audio queue (backpressure guard)
                if audio_queue.qsize() > 350:
                    logger.warning("Audio queue large (%d) dropping input", audio_queue.qsize())
                    continue
                try:
                    audio_queue.put_nowait(pcm)
                except queue.Full:
                    logger.warning("Audio queue full, drop chunk")
                    continue

                # If we've seen silence for long enough, force a stream flush (end utterance)
                if silent and (now - last_voice_ts) >= float(os.getenv("HD_WS_SILENCE_TIMEOUT", "0.6")):
                    # perform controlled restart to force Google to finalize the previous utterance
                    if restarting_lock.acquire(blocking=False):
                        try:
                            logger.info("Silence timeout reached -> flushing STT stream (language=%s)", language)
                            stop_event.set()
                            try:
                                audio_queue.put_nowait(None)
                            except Exception:
                                pass
                            if stt_thread and stt_thread.is_alive():
                                stt_thread.join(timeout=2.0)
                            # clear stop flag and start new STT worker
                            stop_event = threading.Event()
                            stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
                            stt_thread.start()
                            last_voice_ts = time.time()
                        except Exception as e:
                            logger.exception("Error during STT flush/restart: %s", e)
                        finally:
                            restarting_lock.release()

            elif mtype == "stop":
                logger.info("Client stop received; flushing and closing")
                # signal stream end
                try:
                    stop_event.set()
                    audio_queue.put_nowait(None)
                except Exception:
                    pass
                # tell consumer to exit once transcripts drained
                await transcripts_queue.put(None)
                await ws.send_text(json.dumps({"type":"bye"}))
                await ws.close()
                return

            else:
                await ws.send_text(json.dumps({"type":"error","error":"unknown_type"}))

    except WebSocketDisconnect:
        logger.info("HD WS disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
    finally:
        # cleanup
        try:
            stop_event.set()
            audio_queue.put_nowait(None)
        except Exception:
            pass
        # ensure transcripts consumer exits
        try:
            transcripts_queue.put_nowait(None)
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
