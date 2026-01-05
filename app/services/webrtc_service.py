from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Optional, Callable

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from av import AudioFrame
import librosa

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
OPUS_SAMPLE_RATE = 48000
STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 16000

# ============================================================
# AUDIO TRACK HANDLERS
# ============================================================

class IncomingAudioTrack(MediaStreamTrack):
    """
    Receives Opus audio from browser, downsamples to 16kHz,
    and feeds to STT service.
    """
    kind = "audio"
    
    def __init__(self, stt_callback: Callable[[bytes], None]):
        super().__init__()
        self.stt_callback = stt_callback
        self._running = True
    
    async def recv(self):
        """Process incoming audio frames"""
        try:
            frame: AudioFrame = await super().recv()
            
            if not self._running:
                return frame
            
            # Convert to numpy array (48kHz Opus)
            audio_data = frame.to_ndarray()
            
            # Downsample 48kHz → 16kHz for Google STT
            if frame.sample_rate == OPUS_SAMPLE_RATE:
                # Flatten to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=0)
                
                # Resample using librosa
                audio_16k = librosa.resample(
                    audio_data.astype(np.float32),
                    orig_sr=OPUS_SAMPLE_RATE,
                    target_sr=STT_SAMPLE_RATE
                )
                
                # Convert to int16 PCM
                audio_16k = (audio_16k * 32767).astype(np.int16)
                pcm_bytes = audio_16k.tobytes()
                
                # Send to STT service
                self.stt_callback(pcm_bytes)
            
            return frame
            
        except Exception as e:
            logger.exception(f"Error processing incoming audio: {e}")
            return frame
    
    def stop(self):
        self._running = False


class OutgoingAudioTrack(MediaStreamTrack):
    """
    Sends TTS audio back to browser via WebRTC.
    """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=100)
        self._timestamp = 0
        self._running = True
    
    async def recv(self):
        """Send audio frames to browser"""
        try:
            if not self._running:
                # Send silence when stopped
                silence = np.zeros(480, dtype=np.int16)
                frame = AudioFrame.from_ndarray(
                    silence.reshape(1, -1),
                    format='s16',
                    layout='mono'
                )
                frame.sample_rate = TTS_SAMPLE_RATE
                frame.pts = self._timestamp
                self._timestamp += 480
                return frame
            
            # Get PCM data from queue (non-blocking with timeout)
            try:
                pcm_data = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.02
                )
            except asyncio.TimeoutError:
                # No data available, send silence
                silence = np.zeros(480, dtype=np.int16)
                pcm_data = silence.tobytes()
            
            # Convert PCM bytes to numpy array
            audio_np = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Pad or trim to consistent frame size (480 samples = 30ms @ 16kHz)
            if len(audio_np) < 480:
                audio_np = np.pad(audio_np, (0, 480 - len(audio_np)))
            elif len(audio_np) > 480:
                audio_np = audio_np[:480]
            
            # Create audio frame
            frame = AudioFrame.from_ndarray(
                audio_np.reshape(1, -1),
                format='s16',
                layout='mono'
            )
            frame.sample_rate = TTS_SAMPLE_RATE
            frame.pts = self._timestamp
            self._timestamp += 480
            
            return frame
            
        except Exception as e:
            logger.exception(f"Error sending audio frame: {e}")
            # Return silence on error
            silence = np.zeros(480, dtype=np.int16)
            frame = AudioFrame.from_ndarray(
                silence.reshape(1, -1),
                format='s16',
                layout='mono'
            )
            frame.sample_rate = TTS_SAMPLE_RATE
            frame.pts = self._timestamp
            self._timestamp += 480
            return frame
    
    async def send_audio(self, pcm_data: bytes):
        """Queue audio data to be sent"""
        if self._running:
            try:
                self.queue.put_nowait(pcm_data)
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping frame")
    
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
        self.pc = RTCPeerConnection()
        self.incoming_track: Optional[IncomingAudioTrack] = None
        self.outgoing_track = OutgoingAudioTrack()
        self.stt_callback = stt_callback
        
        # Add outgoing track immediately
        self.pc.addTrack(self.outgoing_track)
        
        # Handle incoming tracks
        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "audio":
                self.incoming_track = IncomingAudioTrack(stt_callback)
                # Relay incoming audio through our custom track
                asyncio.create_task(self._relay_audio(track))
        
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"WebRTC connection state: {self.pc.connectionState}")
    
    async def _relay_audio(self, source_track):
        try:
            while True:
                frame: AudioFrame = await source_track.recv()

                audio = frame.to_ndarray()

                if len(audio.shape) > 1:
                    audio = audio.mean(axis=0)

                audio_16k = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=frame.sample_rate,
                    target_sr=STT_SAMPLE_RATE,
                )

                pcm = (audio_16k * 32767).astype(np.int16).tobytes()

                self.stt_callback(pcm)

        except Exception as e:
            logger.exception(f"Audio relay error: {e}")
    
    async def handle_offer(self, sdp: str) -> str:
        """Handle WebRTC offer and return answer"""
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        return self.pc.localDescription.sdp
    
    async def add_ice_candidate(self, candidate_dict: dict):
        """Add ICE candidate"""
        # Ensure the candidate mapping matches the aiortc requirements we discussed
        candidate = RTCIceCandidate(
            sdpMid=candidate_dict.get("sdpMid"),
            sdpMLineIndex=candidate_dict.get("sdpMLineIndex"),
            candidate=candidate_dict.get("candidate")
        )
        # EXACT FIX: This line MUST be indented inside the function
        await self.pc.addIceCandidate(candidate)

    
    async def send_audio(self, pcm_data: bytes):
        """Send TTS audio to browser"""
        await self.outgoing_track.send_audio(pcm_data)
    
    async def close(self):
        """Clean up WebRTC resources"""
        if self.incoming_track:
            self.incoming_track.stop()
        self.outgoing_track.stop()
        await self.pc.close()


# ============================================================
# GLOBAL SESSION MANAGER
# ============================================================

_sessions: dict[str, WebRTCSession] = {}

async def create_session(
    session_id: str,
    stt_callback: Callable[[bytes], None],
) -> WebRTCSession:
    """Create new WebRTC session"""
    if session_id in _sessions:
        await _sessions[session_id].close()
    
    session = WebRTCSession(session_id, stt_callback)
    _sessions[session_id] = session
    logger.info(f"Created WebRTC session: {session_id}")
    return session

def get_session(session_id: str) -> Optional[WebRTCSession]:
    """Get existing session"""
    return _sessions.get(session_id)

async def close_session(session_id: str):
    """Close and remove session"""
    session = _sessions.pop(session_id, None)
    if session:
        await session.close()
        logger.info(f"Closed WebRTC session: {session_id}")
