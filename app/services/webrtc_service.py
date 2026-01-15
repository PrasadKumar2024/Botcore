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

# In app/services/webrtc_service.py

class OutgoingAudioTrack(MediaStreamTrack):
    """
    STABLE AUDIO TRACK (High Capacity)
    """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        # INCREASED SIZE: 500 chunks = ~10 seconds of audio.
        # This allows the "Full Response" to be dumped into memory without blocking.
        self.queue = asyncio.Queue(maxsize=500) 
        self._timestamp = 0
        self._running = True
        self.buffer = bytearray()
        self.buffering = True 
        self.JITTER_TARGET = 3840 # 80ms buffer

    async def recv(self):
        if not self._running:
            return await self._get_silence_frame()

        REQUIRED_BYTES = 1920 
        
        # Jitter Buffer Logic (Prevents Stutter)
        if len(self.buffer) < REQUIRED_BYTES:
            self.buffering = True
        
        while len(self.buffer) < self.JITTER_TARGET and self._running:
            try:
                # Fast Fetch (10ms)
                chunk_48k = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                self.buffer.extend(chunk_48k)
                if len(self.buffer) >= self.JITTER_TARGET:
                    self.buffering = False
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        if self.buffering and len(self.buffer) < self.JITTER_TARGET:
            return await self._get_silence_frame()

        if len(self.buffer) >= REQUIRED_BYTES:
            chunk = bytes(self.buffer[:REQUIRED_BYTES])
            del self.buffer[:REQUIRED_BYTES]
            
            audio_np = np.frombuffer(chunk, dtype=np.int16)
            frame = AudioFrame.from_ndarray(audio_np.reshape(1, -1), format='s16', layout='mono')
            frame.sample_rate = 48000
            frame.pts = self._timestamp
            self._timestamp += 960
            return frame
        else:
            return await self._get_silence_frame()

    async def _get_silence_frame(self):
        silence = np.zeros(960, dtype=np.int16)
        frame = AudioFrame.from_ndarray(silence.reshape(1, -1), format='s16', layout='mono')
        frame.sample_rate = 48000
        frame.pts = self._timestamp
        self._timestamp += 960
        return frame

    async def send_audio(self, pcm_16k_data: bytes):
        """
        SAFE RESAMPLING & ENQUEUE
        """
        if self._running and len(pcm_16k_data) > 0:
            # 1. Resample (Producer Side)
            pcm_48k, _ = audioop.ratecv(pcm_16k_data, 2, 1, 16000, 48000, None)
            
            # 2. Safe Enqueue (Backpressure enabled)
            # We use await to prevent data loss. The increased queue size (500)
            # ensures we rarely actually block.
            await self.queue.put(pcm_48k)

    def stop(self):
        self._running = False



# Update WebRTCSession to support async send_audio
class WebRTCSession:
    # ... (init and other methods remain the same) ...
    def __init__(self, session_id: str, stt_callback: Callable[[bytes], None]):
        # ... (Keep existing __init__ code) ...
        self.session_id = session_id
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
                relayed_track = self.relay.subscribe(track)
                asyncio.create_task(self._relay_audio(relayed_track))
        
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            if self.pc.connectionState in ["failed", "closed"]:
                await self.close()

    # ... (Keep _relay_audio and _downsample and handle_offer and add_ice_candidate) ...
    async def _relay_audio(self, source_track):
        # Keep your existing relay code
        buffer = bytearray()
        while True:
            try:
                frame = await source_track.recv()
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(None, self._downsample, frame)
                if pcm_bytes:
                    buffer.extend(pcm_bytes)
                    while len(buffer) >= 960:
                        chunk = bytes(buffer[:960])
                        buffer = buffer[960:]
                        self.stt_callback(chunk)
                else:
                    self.stt_callback(b'\x00' * 960)
            except Exception:
                break
    
    def _downsample(self, frame):
        try:
            import av
            resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
            frames = resampler.resample(frame)
            return b''.join(f.to_ndarray().tobytes() for f in frames)
        except Exception:
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
        except Exception:
            pass

    async def send_audio(self, pcm_data: bytes):
        # Propagate the await
        await self.outgoing_track.send_audio(pcm_data)

    async def close(self):
        self.outgoing_track.stop()
        await self.pc.close()


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
        """Reads audio from browser and sends EXACTLY 960-byte chunks to STT."""
        logger.info("Starting audio relay with 960-byte chunking...")
        buffer = bytearray()  # ← CRITICAL: Accumulate audio here
    
        while True:
            try:
                frame = await source_track.recv()
            
            # Convert to 16kHz PCM
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._downsample, frame
                )
            
                if pcm_bytes:
                    buffer.extend(pcm_bytes)
                
                # Send EXACTLY 960-byte chunks (30ms @ 16kHz)
                    while len(buffer) >= 960:
                        chunk = bytes(buffer[:960])
                        buffer = buffer[960:]  # Remove sent data
                        logger.info(f"✅ Sending 960-byte chunk to STT")
                        self.stt_callback(chunk)
                else:
                # Send silence to keep pipeline alive
                    logger.warning("Empty frame, sending silence")
                    self.stt_callback(b'\x00' * 960)
                
            except MediaStreamError:
                logger.info("Track ended. Stopping relay.")
                break
            except Exception as e:
                logger.error(f"Relay error: {e}", exc_info=True)
            # Send silence on error to prevent pipeline stall
                self.stt_callback(b'\x00' * 960)
                await asyncio.sleep(0.01)
    #
    def _downsample(self, frame):
        """
        Universal PyAV Conversion.
        Fixes 'AttributeError: to_bytes' by using the numpy interface.
        """
        try:
            import av
            # 1. Initialize Resampler (Matches Google's Requirements)
            resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
            
            # 2. Resample
            frames = resampler.resample(frame)
            
            # 3. Pack Bytes (THE FIX)
            # using to_ndarray().tobytes() works on ALL PyAV versions
            pcm_bytes = b''.join(f.to_ndarray().tobytes() for f in frames)
            
            return pcm_bytes
        except Exception as e:
            logger.error(f"Resampling Error: {e}")
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
