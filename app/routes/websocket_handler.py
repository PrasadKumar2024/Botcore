"""
Real-time WebSocket handler for multilingual voice AI
Handles: Twilio Media Streams → Google STT → Gemini → Google TTS → Twilio
"""
import os
import json
import asyncio
import logging
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from app.utils.audio import twilio_payload_to_linear16, get_best_voice
from app.services.gemini_service import GeminiService
from app.database import SessionLocal

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
_gemini = GeminiService()
executor = ThreadPoolExecutor(max_workers=5)

# Load Google credentials
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("❌ Missing GOOGLE_CREDENTIALS_JSON environment variable")

try:
    _gcreds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
    _speech_client = speech.SpeechClient(credentials=_gcreds)
    _tts_client = tts.TextToSpeechClient(credentials=_gcreds)
    logger.info("✅ Google Cloud Speech services initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Google services: {e}")
    raise

# Supported Indian languages (priority order for auto-detection)
ALTERNATIVE_LANGUAGES = [
    "en-IN",  # Indian English
    "hi-IN",  # Hindi
    "te-IN",  # Telugu
    "ta-IN",  # Tamil
    "bn-IN",  # Bengali
    "ml-IN",  # Malayalam
    "kn-IN",  # Kannada
    "gu-IN",  # Gujarati
    "mr-IN",  # Marathi
]

DEFAULT_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")
MAX_AUDIO_QUEUE = 100

def make_recognition_config():
    """Configure Google STT for multilingual recognition"""
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US",  # Primary (will be overridden by auto-detect)
        alternative_language_codes=ALTERNATIVE_LANGUAGES,
        enable_automatic_punctuation=True,
        model="phone_call",  # Optimized for phone calls
        use_enhanced=True,    # Better quality
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False,  # Continuous listening
    )
    return streaming_config

async def get_ai_response(transcript: str, language_code: str) -> str:
    """Get AI response using RAG + Gemini"""
    db = SessionLocal()
    try:
        # Get context from knowledge base
        from app.services.pinecone_service import pinecone_service
        results = await pinecone_service.search_similar_chunks(
            client_id=DEFAULT_CLIENT_ID,
            query=transcript,
            top_k=2
        )
        context = "\n\n".join([r.get("chunk_text", "") for r in results]) if results else ""
        
        # Build prompt
        system_msg = f"You are a helpful voice assistant for BrightCare Mini Health Service. Respond in {language_code} language. Keep responses concise for voice (under 100 words)."
        
        if context:
            prompt = f"Context:\n{context}\n\nUser ({language_code}): {transcript}\n\nRespond naturally in {language_code}:"
        else:
            prompt = f"User ({language_code}): {transcript}\n\nRespond about BrightCare Mini Health Service in {language_code}:"
        
        # Generate response
        response = _gemini.generate_response(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200,  # Short for voice
            system_message=system_msg
        )
        
        return response if response else "I apologize, I couldn't process that request."
        
    except Exception as e:
        logger.exception(f"AI response error: {e}")
        return "Sorry, I'm having trouble right now. Please try again."
    finally:
        db.close()

async def synthesize_and_send(ws: WebSocket, text: str, language_code: str):
    """Convert text to speech and send to Twilio"""
    try:
        # Get best voice for language
        lang_code, voice_name, gender = get_best_voice(language_code)
        
        logger.info(f"🔊 TTS: {text[:50]}... (lang: {lang_code}, voice: {voice_name})")
        
        # Configure TTS
        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
            ssml_gender=getattr(tts.SsmlVoiceGender, gender)
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        # Synthesize
        response = _tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Send to Twilio
        mulaw_b64 = base64.b64encode(response.audio_content).decode("ascii")
        
        # Clear any queued audio first (barge-in protection)
        await ws.send_json({"event": "clear"})
        
        # Send audio
        await ws.send_json({
            "event": "media",
            "media": {"payload": mulaw_b64}
        })
        
        # Mark end of speech
        await ws.send_json({"event": "mark", "streamSid": "tts_end"})
        
        logger.info("✅ Audio sent to Twilio")
        
    except Exception as e:
        logger.exception(f"TTS error: {e}")

def grpc_stt_worker(loop, audio_queue, transcripts_queue, stop_event):
    """
    Background thread for Google STT streaming.
    Runs blocking gRPC call in separate thread.
    """
    def gen_requests():
        # First request: config
        streaming_config = make_recognition_config()
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        
        # Stream audio chunks
        while not stop_event.is_set():
            try:
                chunk = asyncio.run_coroutine_threadsafe(
                    audio_queue.get(), loop
                ).result(timeout=0.5)
                
                if chunk is None:
                    break
                    
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except Exception:
                if stop_event.is_set():
                    break
                continue
    
    try:
        logger.info("🎤 Starting STT stream")
        responses = _speech_client.streaming_recognize(requests=gen_requests())
        
        for response in responses:
            if stop_event.is_set():
                break
            
            # Push response to async queue
            asyncio.run_coroutine_threadsafe(
                transcripts_queue.put(response), loop
            )
            
    except Exception as e:
        logger.exception(f"STT worker error: {e}")
    finally:
        # Signal end
        try:
            asyncio.run_coroutine_threadsafe(transcripts_queue.put(None), loop)
        except:
            pass
        logger.info("🎤 STT stream ended")

@router.websocket("/media-stream")
async def handle_media_stream(ws: WebSocket):
    """
    Main WebSocket handler for real-time voice AI.
    Manages: Audio input → STT → AI → TTS → Audio output
    """
    await ws.accept()
    call_sid = "unknown"
    logger.info("🔌 WebSocket connected")
    
    # Per-call state
    audio_queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE)
    transcripts_queue = asyncio.Queue()
    stop_event = threading.Event()
    stt_thread = None
    
    is_bot_speaking = False
    detected_language = "en-IN"  # Default
    
    try:
        # Start STT worker thread
        loop = asyncio.get_event_loop()
        stt_thread = threading.Thread(
            target=grpc_stt_worker,
            args=(loop, audio_queue, transcripts_queue, stop_event),
            daemon=True
        )
        stt_thread.start()
        
        # Task: Process STT transcripts
        async def process_transcripts():
            nonlocal is_bot_speaking, detected_language
            
            while True:
                response = await transcripts_queue.get()
                
                if response is None:
                    break
                
                # Process STT results
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    alt = result.alternatives[0]
                    transcript = alt.transcript.strip()
                    is_final = result.is_final
                    
                    # Extract detected language
                    lang = getattr(result, "language_code", None) or detected_language
                    
                    logger.info(f"🎤 {'FINAL' if is_final else 'interim'}: '{transcript}' (lang: {lang})")
                    
                    # BARGE-IN: If user speaks while bot is speaking
                    if not is_final and is_bot_speaking and len(transcript) > 5:
                        logger.info("⚡ Barge-in detected - stopping bot")
                        await ws.send_json({"event": "clear"})
                        is_bot_speaking = False
                    
                    # FINAL TRANSCRIPT: Generate response
                    if is_final and len(transcript) > 3:
                        detected_language = lang
                        logger.info(f"💬 Processing: '{transcript}' in {detected_language}")
                        
                        # Get AI response
                        ai_response = await get_ai_response(transcript, detected_language)
                        
                        # Synthesize and send
                        is_bot_speaking = True
                        try:
                            await synthesize_and_send(ws, ai_response, detected_language)
                        finally:
                            is_bot_speaking = False
        
        # Start transcript processing
        transcript_task = asyncio.create_task(process_transcripts())
        
        # Main loop: Receive Twilio messages
        while True:
            try:
                msg = await ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")
                
                if event == "start":
                    call_sid = data.get("start", {}).get("callSid", "unknown")
                    logger.info(f"📞 Call started: {call_sid}")
                    
                elif event == "media":
                    # Inbound audio from Twilio
                    payload = data.get("media", {}).get("payload")
                    if not payload:
                        continue
                    
                    # Convert to LINEAR16
                    linear16 = twilio_payload_to_linear16(payload)
                    if not linear16:
                        continue
                    
                    # Push to STT queue
                    try:
                        audio_queue.put_nowait(linear16)
                    except asyncio.QueueFull:
                        logger.warning("⚠️ Audio queue full, dropping chunk")
                
                elif event == "stop":
                    logger.info(f"📴 Call ended: {call_sid}")
                    break
                
                elif event == "mark":
                    # Audio playback finished
                    is_bot_speaking = False
                    
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected: {call_sid}")
                break
            except Exception as e:
                logger.error(f"❌ Message processing error: {e}")
                break
    
    except Exception as e:
        logger.exception(f"❌ WebSocket error: {e}")
    
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up call: {call_sid}")
        
        stop_event.set()
        
        try:
            await audio_queue.put(None)
        except:
            pass
        
        try:
            await transcripts_queue.put(None)
        except:
            pass
        
        if stt_thread:
            stt_thread.join(timeout=3.0)
        
        try:
            transcript_task.cancel()
        except:
            pass
        
        try:
            await ws.close()
        except:
            pass
        
        logger.info(f"✅ Cleanup complete: {call_sid}")
