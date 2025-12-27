# app/routes/hd_audio_ws.py
import os
import json
import time
import base64
import asyncio
import logging
import wave
import io
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Google speech & tts
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account

# Existing services in your repo (you said they exist)
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Initialize Google clients (reuse env used in your existing app) ---
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

_creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
_speech_client = speech.SpeechClient(credentials=_creds)
_tts_client = tts.TextToSpeechClient(credentials=_creds)

# LLM service
_gemini = GeminiService()

# STT / chunking parameters
STT_SAMPLE_RATE = 16000      # frontend sends 16k PCM
STT_CHUNK_SECONDS = 1.0     # buffer ~1s of audio before sending to speech.recognize()
STT_BYTES_PER_SEC = STT_SAMPLE_RATE * 2  # 16-bit PCM => 2 bytes per sample
CHUNK_BYTE_TARGET = int(STT_BYTES_PER_SEC * STT_CHUNK_SECONDS)

# Simple in-memory per-connection aggregator
class ConnState:
    def __init__(self):
        self.audio_buffer = bytearray()
        self.lang = "en-IN"

# --- Helpers ---
def make_wav_from_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw PCM16LE bytes into a WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

async def stt_recognize_pcm16(pcm16_bytes: bytes, sample_rate: int = STT_SAMPLE_RATE, language_code: str = "en-IN") -> Optional[str]:
    """
    Use Google Speech 'recognize' for short chunks (1s). Returns best transcript or None.
    """
    try:
        audio = speech.RecognitionAudio(content=base64.b64encode(pcm16_bytes).decode('ascii'))
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="default"
        )
        resp = _speech_client.recognize(config=config, audio=audio, timeout=15)
        if not resp.results:
            return None
        transcripts = []
        for r in resp.results:
            if r.alternatives:
                transcripts.append(r.alternatives[0].transcript)
        return " ".join(transcripts).strip() if transcripts else None
    except Exception as e:
        logger.exception("STT error: %s", e)
        return None

def ssml_for_text(text: str) -> str:
    # small SSML wrapper with slight slower rate and small breath breaks
    # Do minimal escaping
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Insert gentle breaks after sentences
    esc = esc.replace('. ', '. <break time="250ms"/> ')
    esc = esc.replace('? ', '? <break time="250ms"/> ')
    esc = esc.replace('! ', '! <break time="250ms"/> ')
    ssml = f"<speak><prosody rate='0.98'>{esc}</prosody></speak>"
    return ssml

def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", sample_rate_hz: int = 24000) -> Optional[bytes]:
    """
    Synthesize text -> LINEAR16 PCM bytes (sample_rate_hz).
    """
    try:
        ssml = ssml_for_text(text)
        synthesis_input = tts.SynthesisInput(ssml=ssml)
        voice = tts.VoiceSelectionParams(language_code=language_code, name=None)
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hz,
            effects_profile_id=["telephony-class-application"] if hasattr(tts.AudioConfig, 'effects_profile_id') else None
        )
        response = _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        logger.exception("TTS error: %s", e)
        return None

async def get_ai_text_response(transcript: str, language_code: str = "en-IN"):
    """
    RAG-first -> LLM fallback.
    Uses pinecone_service.search_similar_chunks and GeminiService.
    Returns plain text to speak.
    """
    try:
        # normalize small: expand 'timings' -> 'business hours' etc
        q = transcript.strip()
        # Try KB lookup (normalized)
        results = await pinecone_service.search_similar_chunks(
            client_id=os.getenv("DEFAULT_KB_CLIENT_ID", None) or "default",
            query=q,
            top_k=4,
            min_score=-1.0
        )
        if results:
            # Build context from top chunks
            context_text = "\n\n".join([r.get("chunk_text", "") for r in results[:3]])
            system_msg = ("You are a helpful clinic assistant. Use ONLY the context to answer. Keep it short and phone-friendly.")
            user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {transcript}"
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: _gemini.generate_response(prompt=user_prompt, system_message=system_msg, temperature=0.0, max_tokens=150))
            if resp:
                return resp.strip()
            # fallback to simpler RAG synth if LLM fails
            return " ".join([r.get("chunk_text", "").strip() for r in results[:2]])
        # No KB results -> conversational fallback
        conv_system = "You are a friendly phone assistant. Answer conversationally and briefly."
        user_prompt = f"User said: {transcript}\nRespond naturally and briefly."
        loop = asyncio.get_running_loop()
        conv = await loop.run_in_executor(None, lambda: _gemini.generate_response(prompt=user_prompt, system_message=conv_system, temperature=0.6, max_tokens=150))
        if conv:
            return conv.strip()
        return "Sorry, I didn't catch that — can you repeat?"
    except Exception as e:
        logger.exception("LLM/RAG error: %s", e)
        return "I'm sorry, I can't access that information right now."

# --- WebSocket handler ---
@router.websocket("/ws/hd-audio")
async def hd_audio_ws(ws: WebSocket):
    await ws.accept()
    state = ConnState()
    try:
        await ws.send_text(json.dumps({"type":"ready"}))
        last_activity = time.time()
        while True:
            data_text = await ws.receive_text()
            last_activity = time.time()
            try:
                msg = json.loads(data_text)
            except Exception:
                await ws.send_text(json.dumps({"type":"error","error":"invalid_json"}))
                continue

            mtype = msg.get("type")
            if mtype == "start":
                meta = msg.get("meta", {})
                state.lang = meta.get("language", "en-IN")
                await ws.send_text(json.dumps({"type":"ack","message":"started"}))

            elif mtype == "audio":
                # payload is base64 of PCM16LE @ 16k mono
                b64 = msg.get("payload")
                if not b64:
                    continue
                try:
                    pcm = base64.b64decode(b64)
                    state.audio_buffer.extend(pcm)
                except Exception as e:
                    logger.exception("bad audio payload: %s", e)
                    continue

                # If buffer large enough, run STT + LLM -> respond
                if len(state.audio_buffer) >= CHUNK_BYTE_TARGET:
                    chunk = bytes(state.audio_buffer[:CHUNK_BYTE_TARGET])
                    # Remove processed bytes
                    del state.audio_buffer[:CHUNK_BYTE_TARGET]

                    # Do STT in background to avoid blocking loop
                    asyncio.create_task(process_audio_chunk_and_respond(ws, chunk, state.lang))

            elif mtype == "stop":
                # flush remaining audio if any
                if state.audio_buffer:
                    chunk = bytes(state.audio_buffer)
                    state.audio_buffer.clear()
                    await process_audio_chunk_and_respond(ws, chunk, state.lang)
                await ws.send_text(json.dumps({"type":"bye"}))
                await ws.close()
                return

            else:
                await ws.send_text(json.dumps({"type":"error","error":"unknown_type"}))

    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as e:
        logger.exception("WS loop error: %s", e)
    finally:
        try:
            await ws.close()
        except:
            pass

# --- processing task ---
async def process_audio_chunk_and_respond(ws: WebSocket, pcm_chunk: bytes, language_code: str):
    """
    Called for each audio chunk (approx 1s). Runs STT -> LLM -> TTS -> send audio back.
    """
    try:
        # 1) STT
        transcript = await stt_recognize_pcm16(pcm_chunk, sample_rate=STT_SAMPLE_RATE, language_code=language_code)
        if transcript:
            await ws.send_text(json.dumps({"type":"transcript", "text": transcript}))
        else:
            # nothing recognized, skip
            return

        # 2) Generate AI text (RAG-first + LLM fallback)
        ai_text = await get_ai_text_response(transcript, language_code=language_code)
        # send AI text message (for display)
        await ws.send_text(json.dumps({"type":"ai_text", "text": ai_text}))

        # 3) TTS synthesize to PCM (24k)
        tts_pcm = synthesize_text_to_pcm(ai_text, language_code=language_code, sample_rate_hz=24000)
        if not tts_pcm:
            await ws.send_text(json.dumps({"type":"error","error":"tts_failed"}))
            return

        # 4) Wrap into WAV (so browser can decode) and base64 encode
        wav_bytes = make_wav_from_pcm16(tts_pcm, sample_rate=24000)
        b64wav = base64.b64encode(wav_bytes).decode('ascii')
        await ws.send_text(json.dumps({"type":"audio", "audio": b64wav}))

    except Exception as e:
        logger.exception("process_audio_chunk error: %s", e)
        try:
            await ws.send_text(json.dumps({"type":"error","error":str(e)}))
        except:
            pass
