from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Optional, Iterator

from google.cloud import speech_v1 as speech
import webrtcvad

logger = logging.getLogger(__name__)

# ===================== CONFIG =====================

SAMPLE_RATE = 16000
LANGUAGE_CODE = "en-US"

STREAMING_LIMIT_SEC = 55
RESTART_BACKOFF_SEC = 1.5
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive silence detection
FRAME_DURATION_MS = 30  # Must be 10, 20, or 30ms for webrtcvad
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples

# ==================================================


class STTWorker:
    """
    Dedicated streaming STT worker.
    Runs entirely in a background thread.
    """

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
        self.silence_threshold = 3  # Number of consecutive silent frames before stopping
        self.consecutive_silence = 0

    # ------------------------------------------------

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="STTWorker",
            daemon=True,
        )
        self.thread.start()
        return self.thread

    # ------------------------------------------------

    def _audio_generator(self) -> Iterator[speech.StreamingRecognizeRequest]:
        """
        Converts raw PCM bytes into Google streaming requests.
        Applies Voice Activity Detection to filter silence.
        """
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:
            # Flush any remaining buffered audio
                if len(self.audio_buffer) > 0:
                    yield speech.StreamingRecognizeRequest(
                        audio_content=bytes(self.audio_buffer)
                    )
                    self.audio_buffer.clear()
                return
        
        # Apply Voice Activity Detection
            if self._is_speech(chunk):
                self.audio_buffer.extend(chunk)
            
            # Send when buffer reaches optimal size (3200 bytes = 100ms)
                if len(self.audio_buffer) >= 3200:
                    yield speech.StreamingRecognizeRequest(
                        audio_content=bytes(self.audio_buffer)
                    )
                    self.audio_buffer.clear()
            else:
            # Silent frame, but flush buffer if it has content
                if len(self.audio_buffer) > 0:
                    yield speech.StreamingRecognizeRequest(
                        audio_content=bytes(self.audio_buffer)
                    )
                    self.audio_buffer.clear()
    # ------------------------------------------------
    def _is_speech(self, pcm_data: bytes) -> bool:
            """
            Determine if audio frame contains speech using WebRTC VAD.
            Returns True if speech is detected, False for silence.
            """
            if len(pcm_data) < FRAME_SIZE * 2:  # 2 bytes per sample
                return False
    
            try:
        # VAD requires exact frame size
                frame = pcm_data[:FRAME_SIZE * 2]
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
        
                if is_speech:
                    self.consecutive_silence = 0
                else:
                    self.consecutive_silence += 1
        
                return is_speech or self.consecutive_silence < self.silence_threshold
        
            except Exception as e:
                logger.warning(f"VAD error: {e}, assuming speech")
                return True
    def _run(self):
        """
        Main STT loop.
        Handles auto-restart and failure recovery.
        """
        while not self.stop_event.is_set():
            try:
                self._stream_once()
            except Exception as e:
                logger.exception("STT stream crashed, restarting")
                time.sleep(RESTART_BACKOFF_SEC)

    # ------------------------------------------------
    
    def _stream_once(self):
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
        
        def request_generator():
        # First request contains only the config
            yield speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            )

        # Following requests contain audio only (already wrapped in StreamingRecognizeRequest)
            for audio_request in self._audio_generator():
                yield audio_request

        responses = self.client.streaming_recognize(
            config=streaming_config,
            requests=request_generator()
        )
        start_ts = time.monotonic()

        for response in responses:
            if self.stop_event.is_set():
                return

            if time.monotonic() - start_ts > STREAMING_LIMIT_SEC:
                logger.info("Restarting STT stream (time limit reached)")
                return

            if not response.results:
                continue

            asyncio.run_coroutine_threadsafe(
                self.transcript_queue.put(response),
                self.loop,
            )

# ==================================================
# ---------------- PUBLIC API -----------------------
# ==================================================

def start_stt_worker(
    *,
    audio_queue: queue.Queue,
    transcript_queue: asyncio.Queue,
    stop_event: threading.Event,
    language: str,
):
    """
    Factory function used by voice.py.

    Starts a dedicated STT thread and returns it.
    """

    loop = asyncio.get_running_loop()

    worker = STTWorker(
        audio_queue=audio_queue,
        transcript_queue=transcript_queue,
        stop_event=stop_event,
        language=language,
        loop=loop,
    )

    return worker.start()
