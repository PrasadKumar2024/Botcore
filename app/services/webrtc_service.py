from __future__ import annotations

import asyncio
import logging
import numpy as np
import audioop
from typing import Optional, Callable, Dict

from aiortc import (
    RTCPeerConnection, 
    RTCSessionDescription, 
    MediaStreamTrack, 
    RTCConfiguration, 
    RTCIceServer
)
from av import AudioFrame
from aiortc.sdp import candidate_from_sdp
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError

logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
OPUS_SAMPLE_RATE = 48000
STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 16000 
# ==================================================

class OutgoingAudioTrack(MediaStreamTrack):
    """Sends TTS audio back to browser at 48kHz."""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=200)
        self._timestamp = 0
        self._running = True
    
    async def recv(self):
        if not self._running:
            return await self._get_silence_frame()

        try:
            pcm_bytes = await asyncio.wait_for(self.queue.get(), timeout=0.02)
            
            # Upsample 16k -> 48k for WebRTC compatibility
            pcm_48k, _ = audioop.ratecv(pcm_bytes, 2, 1, 16000, 48000, None)
            audio_np = np.frombuffer(pcm_48k, dtype=np.int16)

            # Ensure frame size is 960 samples (20ms @ 48kHz)
            # Use simple padding or slicing
            target_samples = 960
            if len(audio_np) < target_samples:
                audio_np = np.pad(audio_np, (0, target_samples - len(audio_np)))
            elif len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]

            frame = AudioFrame.from_ndarray(audio_np.reshape(1, -1), format='s16', layout='mono')
            frame.sample_rate = 48000
            frame.pts = self._timestamp
            self._timestamp += target_samples
            return frame

        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return await self._get_silence_frame()
        except Exception as e:
            logger.error(f"OutgoingTrack error: {e}")
            return await self._get_silence_frame()

    async def _get_silence_frame(self):
        # 20ms silence @ 48kHz
        silence = np.zeros(960, dtype=np.int16)
        frame = AudioFrame.from_ndarray(silence.reshape(1, -1), format='s16', layout='mono')
        frame.sample_rate = 48000
        frame.pts = self._timestamp
        self._timestamp += 960
        return frame

    async def send_audio(self, pcm_data: bytes):
        if self._running:
            try:
                self.queue.put_nowait(pcm_data)
            except asyncio.QueueFull:
                pass # Drop frame if congested

    def stop(self):
        self._running = False


class WebRTCSession:
    def __init__(self, session_id: str, stt_callback: Callable[[bytes], None]):
        self.session_id = session_id
        
        # Google STUN for firewall traversal
        config = RTCConfiguration(iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
        ])
        
        self.pc = RTCPeerConnection(configuration=config)
        self.outgoing_track = OutgoingAudioTrack()
        self.stt_callback = stt_callback
        self.relay = MediaRelay()
        
        self.pc.addTrack(self.outgoing_track)
        
        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info(f"[{self.session_id}] Audio track received")
                # Use MediaRelay to safely duplicate the stream
                relayed_track = self.relay.subscribe(track)
                asyncio.create_task(self._relay_audio(relayed_track))
        
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"[{self.session_id}] WebRTC State: {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed"]:
                await self.close()
    #
    async def _relay_audio(self, source_track):
        """Reads audio from browser and sends to STT callback."""
        logger.info("Starting audio relay...")
        buffer = bytearray()  # ← CRITICAL: Buffer to fix chunk size
    
        while True:
            try:
                frame = await source_track.recv()
            
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._downsample, frame
                )
            
                if pcm_bytes:
    # This confirms audio is successfully moving to the STT worker
                    logger.info(f"🎤 Relaying {len(pcm_bytes)} bytes to STT") 
                    self.stt_callback(pcm_bytes)
                else:
    # ✅ Send exactly 30ms of silence (960 bytes) if the buffer is empty
    # This prevents the 400 "Long duration without audio" error
                    self.stt_callback(b"\x00" * 960)
                
            except MediaStreamError:
                logger.info("Track ended. Stopping relay.")
                break
            except Exception as e:
                logger.error(f"Relay error: {e}")
                self.stt_callback(b'\x00' * 960)
                await asyncio.sleep(0.01)
    #
    def _downsample(self, frame):
        """Safely convert WebRTC AudioFrame to 16kHz mono PCM."""
        try:
            # 1. Convert to numpy
            pcm = frame.to_ndarray()
            
            # 2. Handle Stereo / Planar (Multi-channel)
            # Chrome sends "planar" audio. We MUST mix to mono.
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=0) 
            
            # 3. Handle Empty Frames
            if pcm.size == 0:
                return None
                
            # 4. Resample 48k -> 16k
            pcm = pcm.astype(np.int16).tobytes()
            
            resampled, _ = audioop.ratecv(
                pcm, 2, 1, frame.sample_rate, STT_SAMPLE_RATE, None
            )
            return resampled
        except Exception as e:
            logger.error(f"Downsample error: {e}")
            return None

    async def handle_offer(self, sdp: str) -> str:
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_dict: dict):
        try:
            if not candidate_dict.get("candidate"): return
            candidate = candidate_from_sdp(candidate_dict["candidate"])
            candidate.sdpMid = candidate_dict.get("sdpMid")
            candidate.sdpMLineIndex = candidate_dict.get("sdpMLineIndex")
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"ICE Error: {e}")

    async def send_audio(self, pcm_data: bytes):
        await self.outgoing_track.send_audio(pcm_data)

    async def close(self):
        self.outgoing_track.stop()
        await self.pc.close()

# ===================== GLOBAL MANAGER =====================

_sessions: Dict[str, WebRTCSession] = {}

async def create_session(session_id: str, stt_callback: Callable[[bytes], None]) -> WebRTCSession:
    if session_id in _sessions:
        await _sessions[session_id].close()
    session = WebRTCSession(session_id, stt_callback)
    _sessions[session_id] = session
    return session

async def close_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        await session.close()
