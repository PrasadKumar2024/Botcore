from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, Callable, Optional
from functools import partial
from dataclasses import dataclass

from google.cloud import texttospeech_v1 as tts

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

MAX_CONCURRENT_TTS = 4           # Global cost / rate protection
TTS_QUEUE_MAX = 32               # Per-session backpressure
PCM_SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH = 2             # 16-bit
PCM_CHANNELS = 1
AUDIO_CHUNK_SIZE = 3200          # ~100ms of 16kHz PCM

# ============================================================
# GLOBALS
# ============================================================

_tts_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
_sessions: Dict[str, "TTSSession"] = {}

_tts_client = None

def get_tts_client():
    global _tts_client
    if _tts_client is None:
        _tts_client = tts.TextToSpeechClient()
    return _tts_client

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TTSTask:
    text: str
    language: str
    sentiment: float
    speaking_rate: float = 1.0


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self):
        self._event.clear()


# ============================================================
# SSML BUILDER (EMOTION-AWARE)
# ============================================================

def build_ssml(text: str, sentiment: float, speaking_rate: float = 1.0) -> str:
    """
    Maps sentiment and speaking_rate to prosody for emotional mirroring.
    """
    if sentiment > 0.4:
        base_rate = 105
        pitch = "+2st"
    elif sentiment < -0.4:
        base_rate = 90
        pitch = "-2st"
    else:
        base_rate = 100
        pitch = "0st"
    
    final_rate = int(base_rate * speaking_rate)
    final_rate = max(80, min(120, final_rate))
    
    return f"""
<speak>
  <prosody rate="{final_rate}%" pitch="{pitch}">
    {text}
  </prosody>
</speak>
""".strip()


# ============================================================
# BLOCKING PROVIDER CALL (EXECUTOR ONLY)
# ============================================================

def synthesize_blocking(
    *,
    text: str,
    language: str,
    sentiment: float,
    speaking_rate: float = 1.0,
) -> bytes:
    """
    Blocking Google TTS call using STABLE PREMIUM (Neural2) voices.
    """
    # 1. Build SSML (Neural2 supports this; Journey does not)
    ssml = build_ssml(text, sentiment, speaking_rate)
    synthesis_input = tts.SynthesisInput(ssml=ssml)

    # 2. SELECT NEURAL2 VOICES (High Quality, Human-like)
    # Neural2-J is a deep, professional male voice.
    # Neural2-D is a natural Indian English voice.
    if language == "en-US":
        voice_name = "en-US-Neural2-J" 
    elif language == "en-IN":
        voice_name = "en-IN-Neural2-D"
    else:
        voice_name = "en-US-Neural2-C"

    voice = tts.VoiceSelectionParams(
        language_code=language,
        name=voice_name
    )

    # 3. AUDIO CONFIG (Optimized for WebRTC)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=PCM_SAMPLE_RATE,
        effects_profile_id=["telephony-class-application"], # Enhances clarity
    )

    client = get_tts_client()
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    return response.audio_content



# ============================================================
# PER-SESSION WORKER
# ============================================================

class TTSSession:
    def __init__(self, session_id: str, audio_sender: Callable[[bytes], asyncio.Future]):
        self.session_id = session_id
        self.audio_sender = audio_sender

        self.queue: asyncio.Queue[TTSTask] = asyncio.Queue(TTS_QUEUE_MAX)
        self.cancel_token = CancellationToken()
        self.worker_task: Optional[asyncio.Task] = None
        self.closed = False

    async def start(self):
        if not self.worker_task:
            self.worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        """
        OPTIMIZED WORKER: Atomic Sentence Delivery
        Sends the WHOLE sentence as one block to prevent resampling artifacts and gaps.
        """
        try:
            while not self.closed:
                task: TTSTask = await self.queue.get()

                if self.cancel_token.is_cancelled():
                    self.queue.task_done()
                    continue

                async with _tts_semaphore:
                    if self.cancel_token.is_cancelled():
                        self.queue.task_done()
                        continue

                    try:
                        loop = asyncio.get_running_loop()

                        # 1. Synthesize (Blocking Google Call)
                        pcm: bytes = await loop.run_in_executor(
                            None,
                            partial(
                                synthesize_blocking,
                                text=task.text,
                                language=task.language,
                                sentiment=task.sentiment,
                                speaking_rate=task.speaking_rate,
                             ),
                        )

                        # 2. ATOMIC SEND (The Fix)
                        # Do NOT loop. Do NOT chunk. Send it all at once.
                        # This ensures perfect resampling and zero scheduler gaps.
                        if not self.cancel_token.is_cancelled() and len(pcm) > 0:
                            await self.audio_sender(pcm)

                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("TTS synthesis failed")

                self.queue.task_done()

        except asyncio.CancelledError:
            logger.info("TTS worker stopped")


    async def enqueue(self, task: TTSTask):
        if not self.closed and not self.cancel_token.is_cancelled():
            await self.queue.put(task)

    def cancel(self):
        """
        Barge-in safe cancellation.
        """
        self.cancel_token.cancel()
        self._drain_queue()

    def _drain_queue(self):
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass

    async def close(self):
        self.closed = True
        self.cancel()
        if self.worker_task:
            self.worker_task.cancel()


# ============================================================
# PUBLIC API (USED BY voice.py ONLY)
# ============================================================

async def register_session(
    session_id: str,
    audio_sender: Callable[[bytes], asyncio.Future],
):
    if session_id in _sessions:
        return

    session = TTSSession(session_id, audio_sender)
    _sessions[session_id] = session
    await session.start()


async def tts_enqueue(
    *,
    session_id: str,
    text: str,
    language: str,
    sentiment: float = 0.0,
    speaking_rate: float = 1.0,
    
):
    session = _sessions.get(session_id)
    if not session:
        return

    await session.enqueue(
        TTSTask(
            text=text,
            language=language,
            sentiment=sentiment,
            speaking_rate=speaking_rate,
        )
    )


def cancel_tts(session_id: str):
    session = _sessions.get(session_id)
    if session:
        session.cancel()


async def close_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        await session.close()
