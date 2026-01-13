from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.context.session_state import SessionState
from app.services.stt_service import start_stt_worker
from app.services.gemini_service import gemini_service
from app.services.tts_service import (
    register_session,
    tts_enqueue,
    cancel_tts,
    close_session as close_tts_session,
)
from app.services import webrtc_service
from app.services.rag_engine import rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()

# ===================== TUNING CONSTANTS =====================

AUDIO_QUEUE_MAX = 400
DEBOUNCE_SECONDS = 0.45
SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")
MAX_CONTEXT_TURNS = 8

SYSTEM_PROMPT = "You are a helpful, real-time voice assistant."

# ============================================================


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()

    # ---------- CLIENT ID ----------
    client_id = ws.query_params.get("client_id")
    if not client_id:
        await ws.close(code=1008)
        return

    # ---------- SESSION ----------
    session = SessionState(client_id=client_id, memory_limit=20)

    # ---------- AUDIO PIPELINE ----------
    audio_queue: queue.Queue[Optional[bytes]] = queue.Queue(AUDIO_QUEUE_MAX)
    transcript_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    # ---------- STATE ----------
    last_voice_ts = time.monotonic()
    utterance_buffer: list[str] = []
    is_bot_speaking = False
    current_llm_task: Optional[asyncio.Task] = None
    webrtc_session = None
    
    # Track speaking rate for adaptive pacing
    user_speech_timestamps: list[float] = []
    user_speech_lengths: list[int] = []

    # ============================================================
    # ---------------- INTERNAL HELPERS --------------------------
    # ============================================================

    def stt_audio_callback(pcm_data: bytes):
        """Callback for WebRTC incoming audio"""
        try:
            audio_queue.put_nowait(pcm_data)
        except queue.Full:
            audio_queue.get_nowait()
            audio_queue.put_nowait(pcm_data)

    async def send_tts_audio(pcm_data: bytes):
        """Send TTS audio via WebRTC"""
        if webrtc_session:
            await webrtc_session.send_audio(pcm_data)

    async def hard_barge_in():
        nonlocal is_bot_speaking, current_llm_task

        if not is_bot_speaking:
            return

        cancel_tts(session.session_id)

        if current_llm_task and not current_llm_task.done():
            current_llm_task.cancel()

        is_bot_speaking = False

    async def send_acknowledgment(language: str):
        """Immediate filler to mask LLM latency"""
        acknowledgments = {
            "en-US": ["Let me check that for you...", "One moment...", "Sure, let me see..."],
            "en-IN": ["Sure, checking...", "One second...", "Let me find that..."],
        }
        
        import random
        ack = random.choice(acknowledgments.get(language, acknowledgments["en-US"]))
        
        await tts_enqueue(
            session_id=session.session_id,
            text=ack,
            language=language,
            sentiment=0.3,
            speaking_rate=1.1,
        )

        # In app/routes/voice.py

    async def finalize_utterance(text: str):
        nonlocal is_bot_speaking, current_llm_task
        
        # 1. Add User Turn to Memory
        session.add_turn(role="user", text=text)

        # 2. SMART FILLER LOGIC (The Fix)
        # We check if it's a greeting BEFORE playing the filler sound.
        cleaned_text = text.lower().strip()
        is_greeting = cleaned_text in ["hi", "hello", "hey", "good morning", "good evening", "hi there"]
        
        # Only say "Let me check..." if it is NOT a greeting
        if not is_greeting:
            await send_acknowledgment(session.language)
        
        # 3. Call the RAG Engine (Smart Router)
        rag_result = await rag_engine.answer(
            client_id=session.client_id,
            query=text,
            session_context=session.memory,
            language=session.language,
        )

        async def llm_stream():
            nonlocal is_bot_speaking
            is_bot_speaking = True
            try:
                # 4. Speak the RAG Result
                final_answer = rag_result.spoken_text
                
                if not final_answer:
                    final_answer = "I'm sorry, I don't have that information right now."

                sentences = SENTENCE_REGEX.split(final_answer)
                for sent in sentences:
                    clean_sent = sent.strip()
                    if clean_sent:
                        await tts_enqueue(
                            session_id=session.session_id,
                            text=clean_sent,
                            language=session.language,
                            sentiment=rag_result.sentiment,
                            speaking_rate=session.speaking_rate,
                        )
                
                session.add_turn(role="assistant", text=final_answer)

            except Exception as e:
                logger.error(f"LLM Logic Error: {e}")
            finally:
                is_bot_speaking = False

        current_llm_task = asyncio.create_task(llm_stream())


        async def llm_stream():
            nonlocal is_bot_speaking
            is_bot_speaking = True

            try:
        # Use the RAG-generated response directly
                response_text = rag_result.spoken_text
        
                if not response_text:
            # Fallback if RAG returned empty response
                    response_text = "I apologize, but I don't have enough information to answer that question accurately."
        
        # Split into sentences for progressive TTS
                sentences = SENTENCE_REGEX.split(response_text)
        
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        await tts_enqueue(
                            session_id=session.session_id,
                            text=sent,
                            language=session.language,
                            sentiment=rag_result.sentiment,
                            speaking_rate=session.speaking_rate,
                        )
        
                session.add_turn(role="assistant", text=response_text)

            except asyncio.CancelledError:
                logger.info("Response delivery cancelled")
            finally:
                is_bot_speaking = False

    # ============================================================
    # ---------------- TRANSCRIPT CONSUMER -----------------------
    # ============================================================

    async def transcript_consumer():
        nonlocal last_voice_ts

        debounce_task: Optional[asyncio.Task] = None

        async def debounce():
            await asyncio.sleep(DEBOUNCE_SECONDS)
            if time.monotonic() - last_voice_ts >= DEBOUNCE_SECONDS:
                final = " ".join(utterance_buffer).strip()
                utterance_buffer.clear()
                if final:
                    await finalize_utterance(final)

        while True:
            resp = await transcript_queue.get()
            if resp is None:
                return

            for result in resp.results:
                if not result.alternatives:
                    continue

                alt = result.alternatives[0]
                text = alt.transcript.strip()

                if text and is_bot_speaking:
                    await hard_barge_in()

                if text:
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "text": text,
                        "is_final": result.is_final,
                    }))

                if result.is_final:
                    utterance_buffer.append(text)
                    current_time = time.monotonic()
                    last_voice_ts = current_time
                    
                    # Track speaking rate
                    user_speech_timestamps.append(current_time)
                    user_speech_lengths.append(len(text))
                    
                    if len(user_speech_timestamps) > 5:
                        user_speech_timestamps.pop(0)
                        user_speech_lengths.pop(0)
                    
                    # Calculate adaptive pacing
                    if len(user_speech_timestamps) >= 2:
                        time_span = user_speech_timestamps[-1] - user_speech_timestamps[0]
                        total_chars = sum(user_speech_lengths)
                        if time_span > 0:
                            chars_per_sec = total_chars / time_span
                            if chars_per_sec > 18:
                                session.speaking_rate = 1.15
                            elif chars_per_sec < 10:
                                session.speaking_rate = 0.90
                            else:
                                session.speaking_rate = 1.0

                    if debounce_task:
                        debounce_task.cancel()
                    debounce_task = asyncio.create_task(debounce())

    consumer_task = asyncio.create_task(transcript_consumer())

    # ============================================================
    # ---------------- MAIN SIGNALING LOOP -----------------------
    # ============================================================

    try:
        while True:
            raw = await ws.receive()

            if raw["type"] == "websocket.disconnect":
                logger.info("WebSocket signaling disconnected (keeping session alive)")
                await asyncio.sleep(3600)  # keep session alive
                continue

            if raw["type"] != "websocket.receive":
                continue

            if "text" in raw:
                msg = json.loads(raw["text"])

                if msg["type"] == "webrtc_offer":
                    # Create WebRTC session
                    webrtc_session = await webrtc_service.create_session(
                        session_id=session.session_id,
                        stt_callback=stt_audio_callback,
                    )
                    
                    # Handle offer and get answer
                    answer_sdp = await webrtc_session.handle_offer(msg["sdp"])
                    
                    # Set language
                    session.set_language(msg.get("language", session.language))
                    
                    # Initialize TTS with WebRTC audio sender
                    await register_session(
                        session.session_id,
                        audio_sender=send_tts_audio,
                    )
                    
                    # Start STT worker
                    start_stt_worker(
                        audio_queue=audio_queue,
                        transcript_queue=transcript_queue,
                        stop_event=stop_event,
                        language=session.language,
                    )
                    
                    # Send answer back
                    await ws.send_text(json.dumps({
                        "type": "webrtc_answer",
                        "sdp": answer_sdp,
                    }))
                    
                    await ws.send_text(json.dumps({
                        "type": "ready",
                        "session_id": session.session_id,
                    }))

                elif msg["type"] == "ice_candidate":
                    if webrtc_session and msg.get("candidate"):
                        await webrtc_session.add_ice_candidate(msg["candidate"])

                elif msg["type"] == "stop":
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception:
        logger.exception("Voice WS error")

    finally:
        await asyncio.sleep(1.0)  # allow last TTS audio to play
        stop_event.set()
        audio_queue.put(None)
        await transcript_queue.put(None)

        consumer_task.cancel()
        await hard_barge_in()
        await close_tts_session(session.session_id)
        
        if webrtc_session:
            await webrtc_service.close_session(session.session_id)
