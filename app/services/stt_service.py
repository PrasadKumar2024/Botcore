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

VAD_AGGRESSIVENESS = 1
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

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="STTWorker",
            daemon=True,
        )
        self.thread.start()
        return self.thread

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

    def _stream_once(self):
        """
        Manages a single session with Google Cloud STT.
        USES LIBRARY NATIVE CONFIGURATION (No manual yielding).
        """
        
        # 1. Prepare Configuration
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.language,
            enable_automatic_punctuation=True,
            model="latest_long",
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False, 
        )

        # 2. Open the Stream
        # CRITICAL FIX: Pass 'streaming_config' directly to the client.
        # The library will automatically send it as the FIRST message.
        # The generator must ONLY yield audio.
        responses = self.client.streaming_recognize(
            config=streaming_config,
            requests=self._audio_generator()
        )

        start_ts = time.monotonic()

        # 3. Process Responses
        for response in responses:
            if self.stop_event.is_set():
                break

            if time.monotonic() - start_ts > STREAMING_LIMIT_SEC:
                logger.info("Restarting STT stream (time limit reached)")
                break

            if not response.results:
                continue

            result = response.results[0]
            if result.alternatives:
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put(response),
                    self.loop,
                )

    def _audio_generator(self, initial_config):
    # Send config ONCE
        yield speech.StreamingRecognizeRequest(
            streaming_config=initial_config
        )

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    return

            # ALWAYS send audio (speech OR silence)
                yield speech.StreamingRecognizeRequest(
                    audio_content=chunk
                )

            except queue.Empty:
            # Keep-alive: send 30ms silence
                yield speech.StreamingRecognizeRequest(
                    audio_content=b"\x00" * 960
                )
    def _is_speech(self, pcm_data: bytes) -> bool:
        """Simple VAD check."""
        if len(pcm_data) != FRAME_SIZE * 2: 
            return True 
        try:
            return self.vad.is_speech(pcm_data, SAMPLE_RATE)
        except:
            return True

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
