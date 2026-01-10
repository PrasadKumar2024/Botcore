from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Optional

from google.cloud import speech_v1 as speech
import webrtcvad

logger = logging.getLogger(__name__)

# ===================== CONFIG =====================

SAMPLE_RATE = 16000
LANGUAGE_CODE = "en-US"

STREAMING_LIMIT_SEC = 290 
RESTART_BACKOFF_SEC = 1.0  # Increased slightly to prevent log spam

VAD_AGGRESSIVENESS = 2
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# ==================================================

class STTWorker:
    def __init__(
        self,
        *,
        audio_queue: queue.Queue,
        transcript_queue: asyncio.Queue,
        stop_event: threading.Event,
        language: str,
        loop: asyncio.AbstractEventLoop,
    ):
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.stop_event = stop_event
        self.language = language or LANGUAGE_CODE
        self.loop = loop

        self.client = speech.SpeechClient()
        self.thread: Optional[threading.Thread] = None
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_buffer = bytearray()
    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="STTWorker",
            daemon=True,
        )
        self.thread.start()
        return self.thread
    def _is_speech(self, pcm_data: bytes) -> bool:
        """Temporarily bypassing VAD to ensure all audio is processed."""
        return True  # This ensures all audio is sent to Google STT

    def _run(self):
        """Main STT loop: Connects and handles restarts."""
        while not self.stop_event.is_set():
            try:
                self._stream_once()
            except Exception as e:
                # Filter out standard shutdown noises
                if "503" in str(e) or "11" in str(e):
                    logger.warning(f"STT stream interrupted (retrying): {e}")
                elif not self.stop_event.is_set():
                    logger.error(f"STT Critical Error: {e}")
                
                time.sleep(RESTART_BACKOFF_SEC)
    #
    def _stream_once(self):
        """
        Stream handler using Explicit Config Argument.
        Fixes: 'missing 1 required positional argument: config'
        """
        # 1. Define Config
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.language,
            enable_automatic_punctuation=True,
            model="default", 
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, 
            interim_results=True
        )

        try:
            # 2. Call Google (THE CRITICAL FIX)
            # We pass 'config' explicitly. The generator contains ONLY audio.
            responses = self.client.streaming_recognize(
                config=streaming_config,
                requests=self._audio_generator()
            )
            
            # ... (rest of the processing loop) ...

    def _audio_generator(self):
        """
        Yields ONLY audio chunks. 
        """
        last_send_time = time.monotonic()

        while not self.stop_event.is_set():
            try:
                # 1. Get Audio
                chunk = self.audio_queue.get(timeout=0.05)
                if chunk is None: return

                self.audio_buffer.extend(chunk)

                # 2. Send 100ms Chunks (3200 bytes)
                # Keep this buffer! It stabilizes the stream against 400 errors.
                if len(self.audio_buffer) >= 3200:
                    yield speech.StreamingRecognizeRequest(
                        audio_content=bytes(self.audio_buffer)
                    )
                    self.audio_buffer.clear()
                    last_send_time = time.monotonic()

            except queue.Empty:
                # 3. Heartbeat (Prevent Timeout)
                if time.monotonic() - last_send_time > 1.0:
                    yield speech.StreamingRecognizeRequest(audio_content=b'\x00' * 3200)
                    last_send_time = time.monotonic()
                continue


                    

    #
    
# ===================== FACTORY =====================

def start_stt_worker(
    *,
    audio_queue: queue.Queue,
    transcript_queue: asyncio.Queue,
    stop_event: threading.Event,
    language: str,
):
    loop = asyncio.get_running_loop()
    worker = STTWorker(
        audio_queue=audio_queue,
        transcript_queue=transcript_queue,
        stop_event=stop_event,
        language=language,
        loop=loop,
    )
    return worker.start()
