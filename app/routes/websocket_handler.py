
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
import time
import wave
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
def normalize_and_expand_query(transcript: str) -> str:
    """
    Normalize noisy, colloquial speech into richer query terms for KB matching.
    """
    if not transcript:
        return ""

    s = transcript.lower().strip()

    # de-duplicate immediate repeats: "what's what's your" -> "what's your"
    toks = s.split()
    dedup = [toks[0]] if toks else []
    for i in range(1, len(toks)):
        if toks[i] != toks[i-1]:
            dedup.append(toks[i])
    s = " ".join(dedup)

    # simple map of colloquial -> formal/expanded tokens
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
# Use explicit known default client id if env not provided
DEFAULT_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")

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
    """
    RAG-first, conversation-fallback. Normalize query for better Pinecone matching.
    """
    TARGET_CLIENT_ID = DEFAULT_CLIENT_ID  # use env default already set

    try:
        from app.services.pinecone_service import pinecone_service

        # 0) Normalize and expand noisy transcript
        norm_query = normalize_and_expand_query(transcript)
        logger.info("RAG START: original=%s | normalized=%s", transcript[:160], norm_query[:160])

        # 1) Try KB lookup with normalized query first (get more results to filter)
        results = await pinecone_service.search_similar_chunks(
            client_id=TARGET_CLIENT_ID,
            query=norm_query or transcript,
            top_k=5,
            min_score=-1.0
        )

        # 2) If we still have zero results, try raw transcript once before falling back
        if not results:
            logger.info("No KB matches with normalized query; trying raw transcript.")
            results = await pinecone_service.search_similar_chunks(
                client_id=TARGET_CLIENT_ID,
                query=transcript,
                top_k=5,
                min_score=-1.0
            )

        # 3) If KB results exist -> RAG response (temperature 0.0)
        if results:
            context_text = "\n\n".join([f"--- START CHUNK ---\n{r.get('chunk_text','')}\n--- END CHUNK ---" for r in results[:3]])
            system_msg = (
                "You are a professional assistant. Use ONLY the context provided to answer. "
                "Keep answers short and phone-friendly (1-2 sentences). If not in context, say you don't know."
            )
            user_prompt = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {transcript}"
            loop = asyncio.get_running_loop()
            llm_partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=system_msg, temperature=0.0, max_tokens=150)
            try:
                response = await asyncio.wait_for(loop.run_in_executor(executor, llm_partial), timeout=LLM_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("LLM timed out on RAG call")
                response = None
            return (response.strip() if response else "I am sorry, but I don't have that information in my records.")

        # 4) No KB results -> natural conversation fallback (higher temp)
        logger.info("No KB results — using LLM conversational fallback.")
        conv_system = (
            "You are a friendly phone assistant. Answer conversationally and briefly. "
            "If user asks for something you can't verify, offer to follow up rather than invent facts."
        )
        user_prompt = f"User said: {transcript}\n\nRespond naturally and briefly (1-2 sentences)."
        loop = asyncio.get_running_loop()
        llm_partial = functools.partial(_gemini.generate_response, prompt=user_prompt, system_message=conv_system, temperature=0.6, max_tokens=150)
        try:
            response = await asyncio.wait_for(loop.run_in_executor(executor, llm_partial), timeout=LLM_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("LLM timed out on conversational fallback")
            response = None

        return response.strip() if response else "Sorry, I didn't catch that — can you repeat?"

    except Exception as e:
        logger.exception("❌ RAG Failure: %s", e)
        return "I'm sorry, I'm having trouble accessing my files right now."



async def synthesize_and_send(ws: WebSocket, text: str, language_code: str):
    if not text:
        return

    try:
        lang_code, voice_name, gender = get_best_voice(language_code)
        logger.info("TTS request: lang=%s voice=%s text=%s", lang_code, voice_name, text[:80])

        # Build SSML for slightly more natural prosody
        # Adjust rate to ~0.97 for smoother speech on phone
        ssml = (
            "<speak>"
            f"<prosody rate='0.97'>{text}</prosody>"
            "</speak>"
        )
        synthesis_input = tts.SynthesisInput(ssml=ssml)

        # prefer direct 8000Hz to avoid resampling if supported
        preferred_rate = 8000
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=preferred_rate)

        # Build voice params; only set ssml_gender if known
        voice_params = {"language_code": lang_code, "name": voice_name}
        if gender:
            voice_params["ssml_gender"] = getattr(tts.SsmlVoiceGender, gender)

        voice = tts.VoiceSelectionParams(**voice_params)

        def do_tts_call(cfg):
            return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=cfg)

        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(loop.run_in_executor(executor, functools.partial(do_tts_call, audio_config)), timeout=TTS_TIMEOUT)
            linear16 = resp.audio_content
            used_rate = preferred_rate
            logger.info("TTS succeeded at %d Hz", used_rate)
        except Exception as e:
            # fallback: try a safer higher sample rate (24k) and mark for resample
            logger.warning("Preferred TTS rate failed (%s). Falling back to 24000 and will resample. Error: %s", preferred_rate, e)
            audio_config2 = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=24000)
            try:
                resp2 = await asyncio.wait_for(loop.run_in_executor(executor, functools.partial(do_tts_call, audio_config2)), timeout=TTS_TIMEOUT)
                linear16 = resp2.audio_content
                used_rate = 24000
                logger.info("TTS produced %d Hz audio (will resample to 8000).", used_rate)
            except Exception as e2:
                logger.exception("TTS fallback failed: %s", e2)
                return

        # Save temp linear16 raw file for possible ffmpeg resampling
        tmp_in = tempfile.NamedTemporaryFile(delete=False)
        tmp_in_name = tmp_in.name
        tmp_in.write(linear16)
        tmp_in.flush()
        tmp_in.close()

        # If rate already 8000 and linear16 is 16-bit PCM, just convert to μ-law
        if used_rate == 8000:
            try:
                # linear16 is raw 16-bit LITTLE-ENDIAN PCM bytes; convert to μ-law
                pcm_8k = linear16
                mulaw_audio = audioop.lin2ulaw(pcm_8k, 2)
            except Exception as e:
                logger.exception("lin2ulaw conversion failed (8000Hz): %s", e)
                return
        else:
            # Try ffmpeg to resample and produce μ-law (best quality)
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                tmp_out = tempfile.NamedTemporaryFile(delete=False)
                tmp_out_name = tmp_out.name
                tmp_out.close()
                # ffmpeg: read raw s16le at used_rate, output mulaw at 8k
                cmd = [
                    ffmpeg_path,
                    "-f", "s16le",
                    "-ar", str(used_rate),
                    "-ac", "1",
                    "-i", tmp_in_name,
                    "-ar", "8000",
                    "-ac", "1",
                    "-f", "mulaw",
                    tmp_out_name
                ]
                try:
                    subprocess.check_call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=10)
                    mulaw_audio = open(tmp_out_name, "rb").read()
                    os.unlink(tmp_out_name)
                except Exception as e:
                    logger.warning("ffmpeg resample failed: %s — falling back to audioop.ratecv", e)
                    # fallback to audioop.chain: convert linear16->pcm_8k then to μ-law
                    try:
                        # read raw s16le
                        data = open(tmp_in_name, "rb").read()
                        pcm_8k = audioop.ratecv(data, 2, 1, used_rate, 8000, None)[0]
                        mulaw_audio = audioop.lin2ulaw(pcm_8k, 2)
                    except Exception as e2:
                        logger.exception("audioop resample fallback failed: %s", e2)
                        os.unlink(tmp_in_name)
                        return
            else:
                # no ffmpeg: use audioop.ratecv (less quality)
                try:
                    data = open(tmp_in_name, "rb").read()
                    pcm_8k = audioop.ratecv(data, 2, 1, used_rate, 8000, None)[0]
                    mulaw_audio = audioop.lin2ulaw(pcm_8k, 2)
                except Exception as e:
                    logger.exception("audioop.ratecv conversion failed: %s", e)
                    os.unlink(tmp_in_name)
                    return

        # remove temp input
        try:
            os.unlink(tmp_in_name)
        except Exception:
            pass

        # --- Send to Twilio in 20ms μ-law frames ---
        stream_sid = getattr(ws, "_twilio_stream_sid", None)
        if not stream_sid:
            logger.warning("No Twilio streamSid known; outbound TTS may not be played by Twilio.")

        CHUNK_BYTES = 160  # 20ms at 8k => 160 bytes
        try:
            try:
                await ws.send_json({"event": "clear", "streamSid": stream_sid} if stream_sid else {"event": "clear"})
            except Exception:
                pass

            import base64, time
            for i in range(0, len(mulaw_audio), CHUNK_BYTES):
                chunk = mulaw_audio[i:i + CHUNK_BYTES]
                if not chunk:
                    continue
                payload = base64.b64encode(chunk).decode("ascii")
                msg = {"event": "media", "media": {"payload": payload}}
                if stream_sid:
                    msg["streamSid"] = stream_sid
                await ws.send_json(msg)
                await asyncio.sleep(0.020)

            try:
                mark = {"event": "mark", "streamSid": stream_sid or "tts_end"}
                await ws.send_json(mark)
            except Exception:
                pass

            logger.info("✅ TTS audio sent to Twilio (stream)")
        except Exception as e:
            logger.exception("Error sending TTS to Twilio stream: %s", e)

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
                    call_info = data.get("start", {}) or {}
                    call_sid = call_info.get("callSid", "unknown")
    # Twilio start event also contains streamSid — save it for outbound media
                    ws._twilio_stream_sid = call_info.get("streamSid") or call_info.get("stream_sid") or None
                    logger.info("Call started: %s streamSid=%s", call_sid, getattr(ws, "_twilio_stream_sid", None))
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
