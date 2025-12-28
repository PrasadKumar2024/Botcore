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
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import deque

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
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "14.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "18.0"))

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

# VAD / silence & debounce tuning
SILENCE_TIMEOUT = float(os.getenv("HD_WS_SILENCE_TIMEOUT", "0.9"))  # used for debounce logic (seconds)
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

def ssml_for_text(text: str, prosody_rate: float = 0.95) -> str:
    # richer SSML with modestly slower rate and pauses
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc = esc.replace(". ", ". <break time='220ms'/> ")
    esc = esc.replace("? ", "? <break time='220ms'/> ")
    esc = esc.replace("! ", "! <break time='220ms'/> ")
    # small breathing pause at commas
    esc = esc.replace(", ", ", <break time='120ms'/> ")
    return f"<speak><prosody rate='{prosody_rate}'>{esc}</prosody></speak>"

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
    Robust STT worker: supports both google client signatures:
      - streaming_recognize(requests_iterable)
      - streaming_recognize(streaming_config, requests_iterable)
    Does NOT inject termination sentinel (main controls lifecycle).
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
        # first message is config
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_cfg)
        # then audio content
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
        # generator that yields only audio_content (for signature that accepts (config, requests))
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
        # Try the single-arg signature first; if it raises TypeError, call the two-arg signature.
        try:
            return _speech_client.streaming_recognize(gen_requests_with_config())
        except TypeError as e:
            # Look for the specific complaint and fallback
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

# ----------------- RAG + LLM (uses DEFAULT_CLIENT_ID) -----------------
async def get_ai_text_response(transcript: str, language_code: str = "en-IN", conversation_history: Optional[List[Tuple[str,str]]] = None) -> str:
    """
    Build a prompt that includes short conversation history and RAG context (if present).
    conversation_history: list of tuples ('user'|'assistant', text)
    """
    loop = asyncio.get_running_loop()
    try:
        q = transcript.strip()
        q = q.replace("timings", "business hours").replace("timing", "business hours")
        norm_q = normalize_and_expand_query(q)

        # attempt RAG search
        results = await pinecone_service.search_similar_chunks(
            client_id=DEFAULT_CLIENT_ID,
            query=norm_q or q,
            top_k=4,
            min_score=-1.0
        )
        if not results:
            results = await pinecone_service.search_similar_chunks(
                client_id=DEFAULT_CLIENT_ID,
                query=q,
                top_k=4,
                min_score=-1.0
            )

        # create conversation prefix text
        convo_pref = ""
        if conversation_history:
            lines = []
            for role, txt in conversation_history[-6:]:
                if role == "user":
                    lines.append(f"User: {txt}")
                else:
                    lines.append(f"Assistant: {txt}")
            convo_pref = "\n".join(lines) + "\n\n"

        if results:
            context_text = "\n\n".join([r.get("chunk_text", "") for r in results[:3]])
            system_msg = ("You are a helpful, friendly voice assistant. Use ONLY the provided context to answer the question. "
                          "Answer in a warm, conversational tone appropriate for a phone assistant. Keep responses brief but natural.")
            user_prompt = f"{convo_pref}CONTEXT:\n{context_text}\n\nQUESTION: {transcript}"
            partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=system_msg, temperature=0.0, max_tokens=200)
            fut = loop.run_in_executor(executor, partial)
            try:
                resp = await asyncio.wait_for(fut, timeout=LLM_TIMEOUT)
                return resp.strip() if resp else "Sorry — I couldn't formulate a response from the records."
            except asyncio.TimeoutError:
                logger.warning("LLM RAG timed out")
                return "Sorry, I couldn't fetch details right now."

        # no RAG hits: conversational fallback (use short convo history and warmer temperature)
        conv_sys = "You are a friendly voice assistant. Use the conversation history to respond naturally and helpfully."
        user_prompt = f"{convo_pref}User said: {transcript}\nRespond naturally and briefly."
        partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=conv_sys, temperature=0.7, max_tokens=220)
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

    # conversation state
    conversation = deque(maxlen=6)  # tuples: (role, text)
    utterance_buffer: List[str] = []
    pending_debounce_task: Optional[asyncio.Task] = None

    # VAD / silence detection state
    last_voice_ts = time.time()
    restarting_lock = threading.Lock()  # used only around language restarts
    last_restart_ts = 0.0

    # TTS playback state (simple)
    is_bot_speaking = False

    # canned replies for tiny transcripts
    CANNED = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hey": "Hey — how can I help?",
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "good morning": "Good morning! How can I assist?",
        "good evening": "Good evening! How can I assist?",
    }

    async def send_tts_and_audio(ai_text: str, language_code: str):
        nonlocal is_bot_speaking
        try:
            is_bot_speaking = True
            tts_pcm = await synthesize_text_to_pcm(ai_text, language_code=language_code, sample_rate_hz=24000)
            if tts_pcm:
                wav_bytes = make_wav_from_pcm16(tts_pcm, sample_rate=24000)
                b64wav = base64.b64encode(wav_bytes).decode("ascii")
                # send audio payload
                await ws.send_text(json.dumps({"type":"audio", "audio": b64wav}))
            else:
                await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
        except Exception as e:
            logger.exception("TTS/send failed: %s", e)
            try:
                await ws.send_text(json.dumps({"type":"error", "error":"tts_failed"}))
            except:
                pass
        finally:
            is_bot_speaking = False

    async def handle_final_utterance(text: str):
        """
        Called when debounce decides the user finished an utterance.
        Adds to conversation history, calls RAG/LLM, sends ai_text and TTS.
        """
        nonlocal conversation
        user_text = text.strip()
        if not user_text:
            return

        # add to conversation
        conversation.append(("user", user_text))

        # tiny transcript shortcuts
        t_low = user_text.lower().strip()
        if len(t_low.split()) < 3 and t_low in CANNED:
            ai_text = CANNED[t_low]
            conversation.append(("assistant", ai_text))
            try:
                await ws.send_text(json.dumps({"type":"ai_text", "text": ai_text}))
            except:
                pass
            await send_tts_and_audio(ai_text, language_code=language)
            return

        # call RAG/LLM with conversation context
        try:
            ai_text = await get_ai_text_response(user_text, language_code=language, conversation_history=list(conversation))
            conversation.append(("assistant", ai_text))
            try:
                await ws.send_text(json.dumps({"type":"ai_text", "text": ai_text}))
            except:
                pass
            # TTS playback (barge-in handling: client should stop playback if user speaks)
            await send_tts_and_audio(ai_text, language_code=language)
        except Exception as e:
            logger.exception("Error in handle_final_utterance: %s", e)
            try:
                await ws.send_text(json.dumps({"type":"error","error":"ai_failed"}))
            except:
                pass

    async def debounce_and_handle():
        """
        Wait DEBOUNCE_SECONDS after last final result; if no new voice arrives, process buffered utterances.
        """
        nonlocal pending_debounce_task, utterance_buffer, last_voice_ts
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
            # if recent voice arrived during sleep, skip
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
        """
        Consume streaming responses from Google (placed into transcripts_queue by the STT thread).
        For each final result -> append to buffer and schedule debounce which will call RAG once.
        """
        nonlocal language, utterance_buffer, pending_debounce_task, is_bot_speaking
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

                # If user speaks while assistant is speaking, signal client to stop playback (barge-in)
                if interim_text and is_bot_speaking:
                    try:
                        await ws.send_text(json.dumps({"type": "control", "action": "stop_playback"}))
                    except Exception:
                        pass
                    is_bot_speaking = False  # we assume client will stop

                if interim_text:
                    # send interim for UI
                    try:
                        await ws.send_text(json.dumps({"type":"transcript", "text": interim_text, "is_final": is_final}))
                    except Exception:
                        pass

                if is_final and interim_text:
                    # Append final text to buffer and schedule debounce
                    utterance_buffer.append(interim_text)
                    # update language if Google provides it
                    language = getattr(result, "language_code", language) or language
                    # schedule debounce if not already scheduled
                    if not pending_debounce_task:
                        pending_debounce_task = asyncio.create_task(debounce_and_handle())

    # start transcript consumer task
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
                # If language changed, restart STT thread with new language (rare)
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
                            # start new worker
                            stop_event = threading.Event()
                            stt_thread = threading.Thread(target=grpc_stt_worker, args=(loop, audio_queue, transcripts_queue, stop_event, language), daemon=True)
                            stt_thread.start()
                            last_restart_ts = time.time()
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
                    silent = is_silence(pcm)
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

                # NOTE: We do NOT restart the STT worker on silence anymore.
                # Silence detection is handled by the transcript-side debounce (is_final + debounce).
                # This prevents fragmentation and frequent worker churn.

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
