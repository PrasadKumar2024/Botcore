from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import re
import threading
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.utils.audio import is_silence, WebRTCVAD
from app.context.session_state import SessionState
from app.services.stt_service import start_stt_worker
from app.services.gemini_service import gemini_service
from app.services.tts_service import (
    register_session,
    tts_enqueue,
    cancel_tts,
    close_session,
)
from app.services.pinecone_service import pinecone_service

logger = logging.getLogger(__name__)
router = APIRouter()

# ===================== TUNING CONSTANTS =====================

AUDIO_QUEUE_MAX = 400                     # ~80–120s safety buffer
DEBOUNCE_SECONDS = 0.45                  # silence → utterance end
SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")
MAX_CONTEXT_TURNS = 8

# ============================================================


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    """
    REAL-TIME ORCHESTRATION LAYER
    """

    await ws.accept()

    # ---------------- SESSION ----------------
    session = SessionState(maxlen=20)
    vad = WebRTCVAD(sample_rate=16000)

    # ---------------- AUDIO PIPELINE ----------------
    audio_queue: queue.Queue[Optional[bytes]] = queue.Queue(AUDIO_QUEUE_MAX)
    transcript_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    # ---------------- STATE ----------------
    last_voice_ts = time.monotonic()
    utterance_buffer: list[str] = []
    is_bot_speaking = False
    current_llm_task: Optional[asyncio.Task] = None

    # ---------------- TTS INIT ----------------
    await register_session(
        session.id,
        audio_sender=lambda pcm: ws.send_text(
            json.dumps({
                "type": "audio",
                "audio": base64.b64encode(pcm).decode(),
            })
        )
    )

    # ---------------- STT THREAD ----------------
    stt_thread = start_stt_worker(
        audio_queue=audio_queue,
        transcript_queue=transcript_queue,
        stop_event=stop_event,
        language=session.language,
    )

    await ws.send_text(json.dumps({
        "type": "ready",
        "session_id": session.id,
        "language": session.language,
    }))

    # ============================================================
    # ---------------- INTERNAL HELPERS --------------------------
    # ============================================================

    async def hard_barge_in():
        nonlocal is_bot_speaking, current_llm_task
        if not is_bot_speaking:
            return

        await ws.send_text(json.dumps({
            "type": "control",
            "action": "stop_playback",
        }))

        cancel_tts(session.id)

        if current_llm_task and not current_llm_task.done():
            current_llm_task.cancel()

        is_bot_speaking = False

    async def finalize_utterance(text: str):
        nonlocal is_bot_speaking, current_llm_task

        session.add_user(text)

        # ---------- RAG ----------
        rag = await pinecone_service.search(
            client_id=session.client_id,
            query=text,
            top_k=5,
            min_score=0.5,
        )

        # ---------- STREAMING LLM ----------
        async def llm_stream():
            nonlocal is_bot_speaking
            is_bot_speaking = True

            token_buffer = ""
            sentence_buffer = ""

            try:
                async for token in gemini_service.generate_stream(
                    user_text=text,
                    context=session.recent_context(MAX_CONTEXT_TURNS),
                    rag_chunks=rag.chunks,
                ):
                    token_buffer += token
                    sentence_buffer += token

                    sentences = SENTENCE_REGEX.split(sentence_buffer)
                    if len(sentences) > 1:
                        for sent in sentences[:-1]:
                            sent = sent.strip()
                            if sent:
                                await tts_enqueue(
                                    session_id=session.id,
                                    text=sent,
                                    language=session.language,
                                    sentiment=session.estimate_sentiment(sent),
                                )
                        sentence_buffer = sentences[-1]

                if sentence_buffer.strip():
                    await tts_enqueue(
                        session_id=session.id,
                        text=sentence_buffer.strip(),
                        language=session.language,
                        sentiment=session.estimate_sentiment(sentence_buffer),
                    )

                session.add_assistant(token_buffer)

            except asyncio.CancelledError:
                logger.info("LLM stream cancelled (barge-in)")
            finally:
                is_bot_speaking = False

        current_llm_task = asyncio.create_task(llm_stream())

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
                break

            for result in resp.results:
                if not result.alternatives:
                    continue

                alt = result.alternatives[0]
                text = alt.transcript.strip()
                is_final = result.is_final

                if text and is_bot_speaking:
                    await hard_barge_in()

                if text:
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "text": text,
                        "is_final": is_final,
                    }))

                if is_final:
                    utterance_buffer.append(text)
                    last_voice_ts = time.monotonic()

                    if debounce_task:
                        debounce_task.cancel()
                    debounce_task = asyncio.create_task(debounce())

    consumer_task = asyncio.create_task(transcript_consumer())

    # ============================================================
    # ---------------- MAIN WS LOOP ------------------------------
    # ============================================================

    try:
        while True:
            raw = await ws.receive()
            if raw["type"] == "websocket.receive":
                if "bytes" in raw:
                    pcm = raw["bytes"]
                    if not is_silence(pcm, vad=vad):
                        last_voice_ts = time.monotonic()
                        try:
                            audio_queue.put_nowait(pcm)
                        except queue.Full:
                            audio_queue.get_nowait()
                            audio_queue.put_nowait(pcm)

                elif "text" in raw:
                    msg = json.loads(raw["text"])

                    if msg["type"] == "start":
                        session.language = msg.get("meta", {}).get("language", session.language)

                    elif msg["type"] == "stop":
                        break

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.exception("Voice WS error")
        await ws.send_text(json.dumps({
            "type": "error",
            "error": "internal_error",
        }))

    finally:
        stop_event.set()
        audio_queue.put(None)
        await transcript_queue.put(None)

        consumer_task.cancel()
        await hard_barge_in()
        await close_session(session.id)
        await ws.close()
