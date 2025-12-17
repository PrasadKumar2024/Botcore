import os
import json
import asyncio
import logging
import base64
import threading
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as texttospeech
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor
from app.utils.audio import twilio_payload_to_linear16, linear16_to_twilio_mulaw
from app.services.gemini_service import gemini_service
from app.services.pinecone_service import pinecone_service

router = APIRouter()
logger = logging.getLogger("websocket_handler")

# env & clients
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing env GOOGLE_CREDENTIALS_JSON")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = texttospeech.TextToSpeechClient(credentials=_creds)

# languages to prefer for detection (tweak as needed)
PREFERRED_LANGS = os.getenv("PREFERRED_LANGS", "en-US,hi-IN,te-IN").split(",")

# concurrency & safety
MAX_AUDIO_QUEUE = 120   # approx ~2-3 seconds buffer (tune)
executor = ThreadPoolExecutor(max_workers=4)

def make_streaming_config():
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code=PREFERRED_LANGS[0],
        alternative_language_codes=PREFERRED_LANGS[1:],
        audio_channel_count=1
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )
    return streaming_config

async def send_twilio_clear(ws: WebSocket):
    try:
        await ws.send_json({"event": "clear"})
    except Exception as e:
        logger.debug("Failed to send clear: %s", e)

async def send_twilio_media(ws: WebSocket, mulaw_b64: str):
    await ws.send_json({"event": "media", "media": {"payload": mulaw_b64}})

async def synthesize_and_send(ws: WebSocket, text: str, language_code: str):
    """
    Synthesize with Google TTS -> MULAW@8000 -> send to Twilio via media event.
    """
    if not text:
        return

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MULAW, sample_rate_hertz=8000)

    # TTS is blocking network call; run in executor
    def tts_call():
        return _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    try:
        resp = await asyncio.get_event_loop().run_in_executor(executor, tts_call)
        mulaw_b64 = base64.b64encode(resp.audio_content).decode("ascii")
        # clear and send
        await send_twilio_clear(ws)
        await send_twilio_media(ws, mulaw_b64)
        # optional mark to indicate end
        await ws.send_json({"event": "mark", "name": "tts_end"})
    except Exception as e:
        logger.exception("TTS failed: %s", e)

def stt_generator(loop, audio_queue: asyncio.Queue, stop_event: threading.Event):
    """
    Generator for streaming_recognize. Runs in a thread.
    Pulls LINEAR16 chunks from the async audio_queue.
    """
    # yield initial config
    streaming_config = make_streaming_config()
    yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)

    while not stop_event.is_set():
        # get chunk from async queue using run_coroutine_threadsafe
        fut = asyncio.run_coroutine_threadsafe(audio_queue.get(), loop)
        try:
            chunk = fut.result(timeout=10)
        except Exception:
            # if no chunk for a while, continue to check stop_event
            continue
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def stt_worker(loop, audio_queue: asyncio.Queue, results_queue: asyncio.Queue, stop_event: threading.Event):
    """
    Background thread to run Google gRPC streaming_recognize.
    Pushes speech.StreamingRecognizeResponse objects to results_queue (via asyncio).
    """
    try:
        requests = stt_generator(loop, audio_queue, stop_event)
        responses = _speech_client.streaming_recognize(requests=requests)
        for response in responses:
            # send response back to async loop
            asyncio.run_coroutine_threadsafe(results_queue.put(response), loop)
            if stop_event.is_set():
                break
    except Exception as e:
        logger.exception("STT worker error: %s", e)
    finally:
        # signal termination
        asyncio.run_coroutine_threadsafe(results_queue.put(None), loop)

@router.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WS accepted")
    audio_queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE)
    results_queue = asyncio.Queue()
    stop_event = threading.Event()

    # per-call state
    is_bot_speaking = False
    current_lang = PREFERRED_LANGS[0]

    # start background STT thread
    loop = asyncio.get_event_loop()
    stt_thread = threading.Thread(target=stt_worker, args=(loop, audio_queue, results_queue, stop_event), daemon=True)
    stt_thread.start()

    async def results_consumer():
        nonlocal is_bot_speaking, current_lang
        while True:
            resp = await results_queue.get()
            if resp is None:
                break
            for result in resp.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                transcript = alt.transcript.strip()
                is_final = result.is_final
                detected_lang = None
                # Google may attach language_code at result-level or alternative-level
                if hasattr(result, "language_code") and result.language_code:
                    detected_lang = result.language_code
                elif alt.language_code:
                    detected_lang = alt.language_code

                logger.debug("STT [%s] interim=%s text=%s", detected_lang, not is_final, transcript[:80])

                # Barge-in: if interim and bot is speaking → clear
                if not is_final and is_bot_speaking and transcript:
                    await send_twilio_clear(ws)
                    is_bot_speaking = False

                if is_final:
                    if detected_lang:
                        current_lang = detected_lang
                    # RAG: fetch context if available
                    context = ""
                    try:
                        # call pinecone to fetch context; remember search_similar_chunks is async
                        if pinecone_service.is_configured():
                            chunks = await pinecone_service.search_similar_chunks(client_id="DEFAULT", query=transcript, top_k=3)
                            context = "\n\n".join([c.get("chunk_text","") for c in chunks]) if chunks else ""
                    except Exception as e:
                        logger.exception("Pinecone search failed: %s", e)

                    # Build prompt using gemini_service helper if available
                    if context:
                        prompt = gemini_service.create_rag_prompt(context=context, query=transcript, business_name="BrightCare")
                    else:
                        prompt = f"Customer question: {transcript}\nAnswer concisely."

                    # run LLM in executor (gemini_service.generate_response is sync in your code)
                    def llm_call():
                        try:
                            return gemini_service.generate_response(prompt, temperature=0.6, max_tokens=400, system_message="You are a helpful assistant.")
                        except Exception as e:
                            logger.exception("LLM call failed sync: %s", e)
                            return "Sorry, I'm having trouble thinking right now."

                    is_bot_speaking = True
                    answer = await asyncio.get_event_loop().run_in_executor(executor, llm_call)
                    if not answer:
                        answer = "I apologize, I don't have that information right now."

                    # synthesize and send (await will block until TTS done & media sent)
                    await synthesize_and_send(ws, answer, language_code=current_lang)
                    is_bot_speaking = False

    consumer_task = asyncio.create_task(results_consumer())

    try:
        while True:
            text = await ws.receive_text()
            data = json.loads(text)
            event = data.get("event")
            if event == "start":
                logger.info("Call started: %s", data.get("start", {}))
            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if not payload:
                    continue
                try:
                    linear16 = twilio_payload_to_linear16(payload)
                except Exception as e:
                    logger.exception("Conversion error: %s", e)
                    continue
                # queue audio for STT
                try:
                    await audio_queue.put(linear16)
                except asyncio.QueueFull:
                    logger.warning("Audio queue full — dropping chunk")
            elif event == "stop":
                logger.info("Twilio stop event")
                break
            else:
                logger.debug("Unhandled WS event: %s", event)
    except WebSocketDisconnect:
        logger.info("Websocket disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
    finally:
        # cleanup
        stop_event.set()
        try:
            await audio_queue.put(None)
        except Exception:
            pass
        try:
            await results_queue.put(None)
        except Exception:
            pass
        try:
            consumer_task.cancel()
        except Exception:
            pass
        try:
            stt_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Connection cleaned up")
