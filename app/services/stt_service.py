from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Optional, Iterator

from google.cloud import speech_v1 as speech

logger = logging.getLogger(__name__)

# ===================== CONFIG =====================

SAMPLE_RATE = 16000
LANGUAGE_CODE = "en-US"

STREAMING_LIMIT_SEC = 55          # Google recommends restarting streams < 1 min
RESTART_BACKOFF_SEC = 1.5

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
        """
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:
                return

            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    # ------------------------------------------------

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
        """
        Runs a single Google streaming recognition session.
        """

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

        requests = self._audio_generator()

        responses = self.client.streaming_recognize(
            streaming_config=streaming_config,
            requests=requests,
        )

        start_ts = time.monotonic()

        for response in responses:
            if self.stop_event.is_set():
                return

            # Google enforces stream duration limits
            if time.monotonic() - start_ts > STREAMING_LIMIT_SEC:
                logger.info("Restarting STT stream (time limit)")
                return

            if not response.results:
                continue

            # Push response safely into asyncio loop
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

    loop = asyncio.get_event_loop()

    worker = STTWorker(
        audio_queue=audio_queue,
        transcript_queue=transcript_queue,
        stop_event=stop_event,
        language=language,
        loop=loop,
    )

    return worker.start()
