from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Optional, Callable

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from av import AudioFrame
from aiortc.sdp import candidate_from_sdp
import audioop

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
OPUS_SAMPLE_RATE = 48000
STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 16000  # input PCM rate (will be sent as 48kHz WebRTC)

# ============================================================
# AUDIO TRACK HANDLERS
# ============================================================

class IncomingAudioTrack(MediaStreamTrack):
    """
    Kept for compatibility, but STT is driven by _relay_audio.
    """
    kind = "audio"

    def __init__(self, stt_callback: Callable[[bytes], None]):
        super().__init__()
        self._running = True

    async def recv(self):
        frame: AudioFrame = await super().recv()
        return frame

    def stop(self):
        self._running = False


class OutgoingAudioTrack(MediaStreamTrack):
    """
    Sends TTS audio back to browser via WebRTC (48kHz).
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=100)
        self._timestamp = 0
        self._running = True

    async def recv(self):
        try:
            if not self._running:
                pcm = np.zeros(1440, dtype=np.int16)
            else:
                try:
                    pcm_bytes = await asyncio.wait_for(self.queue.get(), timeout=0.02)
                    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                except asyncio.TimeoutError:
                    pcm = np.zeros(1440, dtype=np.int16)

            if len(pcm) < 1440:
                pcm = np.pad(pcm, (0, 1440 - len(pcm)))
            elif len(pcm) > 1440:
                pcm = pcm[:1440]

            frame = AudioFrame.from_ndarray(
                pcm.reshape(1, -1),
                format="s16",
                layout="mono",
            )
            frame.sample_rate = OPUS_SAMPLE_RATE
            frame.pts = self._timestamp
            self._timestamp += 1440  # 30ms @ 48kHz

            return frame

        except Exception as e:
            logger.exception(f"Outgoing audio error: {e}")
            pcm = np.zeros(1440, dtype=np.int16)
            frame = AudioFrame.from_ndarray(
                pcm.reshape(1, -1),
                format="s16",
                layout="mono",
            )
            frame.sample_rate = OPUS_SAMPLE_RATE
            frame.pts = self._timestamp
            self._timestamp += 1440
            return frame

    async def send_audio(self, pcm_data: bytes):
        if self._running:
            try:
                self.queue.put_nowait(pcm_data)
            except asyncio.QueueFull:
                logger.warning("Outgoing audio queue full, dropping frame")

    def stop(self):
        self._running = False


# ============================================================
# WEBRTC SESSION MANAGER
# ============================================================

class WebRTCSession:
    def __init__(
        self,
        session_id: str,
        stt_callback: Callable[[bytes], None],
    ):
        self.session_id = session_id

        # ✅ STUN added
        self.pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
        )

        self.incoming_track: Optional[IncomingAudioTrack] = None
        self.outgoing_track = OutgoingAudioTrack()
        self.stt_callback = stt_callback

        self.pc.addTrack(self.outgoing_track)

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "audio":
                asyncio.create_task(self._relay_audio(track))

        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"WebRTC connection state: {self.pc.connectionState}")

    async def _relay_audio(self, source_track):
        logger.info("Starting unkillable audio relay...")
        while True:
            try:
                frame = await source_track.recv()
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._downsample, frame
                )
                if pcm_bytes:
                    self.stt_callback(pcm_bytes)
            except Exception as e:
                logger.warning(f"Relay hiccup: {e}, sending silence")
                self.stt_callback(b"\x00" * 640)
                await asyncio.sleep(0.02)

    def _downsample(self, frame):
        audio_data = frame.to_ndarray().astype(np.int16).tobytes()
        resampled, _ = audioop.ratecv(
            audio_data, 2, 1, frame.sample_rate, STT_SAMPLE_RATE, None
        )
        return resampled

    async def handle_offer(self, sdp: str) -> str:
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_dict: dict):
        try:
            candidate_string = candidate_dict.get("candidate")
            if not candidate_string:
                return
            candidate = candidate_from_sdp(candidate_string)
            candidate.sdpMid = candidate_dict.get("sdpMid")
            candidate.sdpMLineIndex = candidate_dict.get("sdpMLineIndex")
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Failed to add ICE candidate: {e}")

    async def send_audio(self, pcm_data: bytes):
        await self.outgoing_track.send_audio(pcm_data)

    async def close(self):
        try:
            if self.incoming_track:
                self.incoming_track.stop()
            self.outgoing_track.stop()
            await self.pc.close()
        except Exception:
            pass


# ============================================================
# GLOBAL SESSION MANAGER
# ============================================================

_sessions: dict[str, WebRTCSession] = {}

async def create_session(
    session_id: str,
    stt_callback: Callable[[bytes], None],
) -> WebRTCSession:
    if session_id in _sessions:
        await _sessions[session_id].close()

    session = WebRTCSession(session_id, stt_callback)
    _sessions[session_id] = session
    logger.info(f"Created WebRTC session: {session_id}")
    return session

def get_session(session_id: str) -> Optional[WebRTCSession]:
    return _sessions.get(session_id)

async def close_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        await session.close()
        logger.info(f"Closed WebRTC session: {session_id}")
