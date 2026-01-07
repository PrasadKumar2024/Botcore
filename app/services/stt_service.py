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

SAMPLE_RATE = 16000  # Must match the rate sent from webrtc_service
LANGUAGE_CODE = "en-US"

# Google STT limits streams to ~5 minutes, we restart safely before that.
STREAMING_LIMIT_SEC = 290 
RESTART_BACKOFF_SEC = 0.5

VAD_AGGRESSIVENESS = 3  # 0-3 (3 is most aggressive at filtering background noise)
FRAME_DURATION_MS = 30  # Fixed for VAD (10, 20, or 30ms)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples @ 16k

# ==================================================

class STTWorker:
    """
    Dedicated streaming STT worker.
    Runs in a background thread to prevent blocking asyncio.
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
        
        # Buffer to accumulate small chunks if necessary (optional optimization)
        self.audio_buffer = bytearray()

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="STTWorker",
            daemon=True,
        )
        self.thread.start()
        return self.thread

    def _run(self):
        """Main loop: Connects to Google, handles disconnects/restarts."""
        while not self.stop_event.is_set():
            try:
                self._stream_once()
            except Exception as e:
                # Don't log "Cancelled" errors on shutdown
                if "503" in str(e) or "11" in str(e): # Common Google retryable errors
                    logger.warning(f"STT stream interrupted (retrying): {e}")
                elif not self.stop_event.is_set():
                    logger.error(f"STT Critical Error: {e}", exc_info=True)
                
                time.sleep(RESTART_BACKOFF_SEC)

    def _stream_once(self):
        """Manages a single session with Google Cloud STT."""
        
        # 1. Prepare Configuration
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.language,
            enable_automatic_punctuation=True,
            model="latest_long", # Use 'latest_long' or 'command_and_search'
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False, 
        )

        # 2. Open the Stream
        # The _audio_generator will yield the config FIRST, then audio.
        requests = self._audio_generator(streaming_config)
        
        # 3. Process Responses
        responses = self.client.streaming_recognize(
            config=None, # Config is in the first request yielded below
            requests=requests
        )

        start_ts = time.monotonic()

        for response in responses:
            if self.stop_event.is_set():
                break

            # Soft restart limit to prevent Google forcing a disconnect
            if time.monotonic() - start_ts > STREAMING_LIMIT_SEC:
                logger.info("Restarting STT stream (time limit reached)")
                break

            if not response.results:
                continue

            # Safely put result in queue for the main thread
            result = response.results[0]
            if result.alternatives:
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put(response),
                    self.loop,
                )

    def _audio_generator(self, initial_config):
        """
        Yields the configuration FIRST, then yields audio chunks from the queue.
        """
        
        # --- CRITICAL FIX: Send Config ONCE at the start ---
        yield speech.StreamingRecognizeRequest(streaming_config=initial_config)
        # ---------------------------------------------------

        while not self.stop_event.is_set():
            try:
                # Wait for audio (blocking with timeout to check stop_event)
                chunk = self.audio_queue.get(timeout=0.1)
                
                if chunk is None: 
                    return # Signal to stop

                # VAD Filter: Only send speech to Google
                if self._is_speech(chunk):
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio generator: {e}")
                break

    def _is_speech(self, pcm_data: bytes) -> bool:
        """Simple VAD check."""
        # VAD requires specific frame lengths (480 samples for 30ms at 16k)
        # If chunk is wrong size, just pass it through to be safe
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
