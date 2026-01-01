import os
import io
import json
import time
import base64
import logging
import asyncio
import threading
import queue
import audioop
import wave
import re
import random
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, AsyncGenerator
from datetime import datetime
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai  # Updated import

# Local services
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
router = APIRouter()

# ================== Configuration ==================
EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "12"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "6"))
MAX_CONCURRENT_STT = int(os.getenv("HD_WS_MAX_STT", "4"))
MAX_CONCURRENT_LLM = int(os.getenv("HD_WS_MAX_LLM", "6"))

STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "12.0"))
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "15.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "20.0"))

STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "0.32"))
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "15"))
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)

# Audio processing
VOICE_ACTIVITY_RMS_THRESHOLD = int(os.getenv("HD_WS_VAD_THRESHOLD", "250"))
MIN_SPEECH_DURATION = float(os.getenv("HD_WS_MIN_SPEECH_DURATION", "0.3"))
MAX_SILENCE_DURATION = float(os.getenv("HD_WS_MAX_SILENCE", "1.5"))

# Business persona
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BrightCare")
BUSINESS_TYPE = os.getenv("BUSINESS_TYPE", "healthcare")
ASSISTANT_PERSONA = os.getenv("ASSISTANT_PERSONA", "friendly and professional assistant")
DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", "default")

# ================== Constants ==================
BYTES_PER_SEC = STT_SAMPLE_RATE * 2
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

TTS_SAMPLE_RATE = 24000
MAX_CONVERSATION_TURNS = 20
MIN_RESPONSE_WORDS = 5
MAX_RESPONSE_WORDS = 100

# ================== Global Executors & Semaphores ==================
executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)
stt_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_STT)
llm_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_LLM)

# ================== Google Clients ==================
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

try:
    _creds_info = json.loads(GOOGLE_CREDS_JSON)
    _creds = service_account.Credentials.from_service_account_info(_creds_info)
    _speech_client = speech.SpeechClient(credentials=_creds)
    _tts_client = tts.TextToSpeechClient(credentials=_creds)
    logger.info("✅ Google Cloud Speech/TTS clients initialized")
except Exception as e:
    logger.error(f"Failed to initialize Google clients: {e}")
    raise

# ================== Voice Configuration ==================
VOICE_MAP = {
    "en-IN": {
        "name": "en-IN-Neural2-C",
        "gender": "MALE",
        "tempo": 1.0,
        "pitch_range": 0.0,  # Changed from string to float
        "emotion_sensitivity": 0.9
    },
    "en-US": {
        "name": "en-US-Neural2-F",
        "gender": "FEMALE", 
        "tempo": 1.05,
        "pitch_range": 0.8,  # Changed from string to float
        "emotion_sensitivity": 0.95
    },
    "en-GB": {
        "name": "en-GB-Neural2-B",
        "gender": "MALE",
        "tempo": 0.98,
        "pitch_range": 0.3,  # Changed from string to float
        "emotion_sensitivity": 0.85
    },
    "hi-IN": {
        "name": "hi-IN-Neural2-A",
        "gender": "FEMALE",
        "tempo": 1.02,
        "pitch_range": 0.5,  # Changed from string to float
        "emotion_sensitivity": 0.92
    }
}

DEFAULT_VOICE = VOICE_MAP["en-IN"]

# ================== Emotional Intelligence System ==================

class EmotionalIntelligence:
    """Emotional intelligence for natural conversation"""
    
    EMOTION_CATEGORIES = {
        "joy": {"keywords": ["happy", "great", "wonderful", "excited", "love", "thanks"], "intensity": 1.2},
        "sadness": {"keywords": ["sad", "unhappy", "disappointed", "sorry", "problem"], "intensity": -1.0},
        "anger": {"keywords": ["angry", "mad", "frustrated", "annoyed", "hate"], "intensity": -1.5},
        "fear": {"keywords": ["worried", "scared", "anxious", "concerned"], "intensity": -0.8},
        "surprise": {"keywords": ["wow", "amazing", "unbelievable", "shocked"], "intensity": 0.7},
        "neutral": {"keywords": [], "intensity": 0.0}
    }
    
    EMOTION_MODIFIERS = {
        "very": 1.5, "extremely": 2.0, "really": 1.4, "so": 1.3,
        "somewhat": 0.7, "slightly": 0.6, "a bit": 0.6, "kind of": 0.8
    }
    
    NEGATION_WORDS = {"not", "no", "never", "don't", "can't", "won't", "isn't", "aren't"}
    
    @staticmethod
    def analyze_emotion(text: str) -> Dict[str, Any]:
        """Analyze emotion from text"""
        if not text:
            return {"primary": "neutral", "intensity": 0.0, "confidence": 0.0}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        emotion_scores = defaultdict(float)
        
        for i, word in enumerate(words):
            # Check for emotion modifiers
            intensity_multiplier = 1.0
            if i > 0 and words[i-1] in EmotionalIntelligence.EMOTION_MODIFIERS:
                intensity_multiplier = EmotionalIntelligence.EMOTION_MODIFIERS[words[i-1]]
            
            # Check for emotion keywords
            for emotion, data in EmotionalIntelligence.EMOTION_CATEGORIES.items():
                for keyword in data["keywords"]:
                    if keyword in word:
                        base_intensity = data["intensity"]
                        emotion_scores[emotion] += base_intensity * intensity_multiplier
        
        # Check for negations
        has_negation = any(neg in text_lower for neg in EmotionalIntelligence.NEGATION_WORDS)
        if has_negation:
            for emotion in emotion_scores:
                emotion_scores[emotion] *= -0.8
        
        # Determine primary emotion
        if not emotion_scores:
            primary_emotion = "neutral"
            intensity = 0.0
        else:
            primary_emotion = max(emotion_scores.items(), key=lambda x: abs(x[1]))[0]
            intensity = emotion_scores[primary_emotion]
        
        # Normalize intensity
        intensity = max(-2.0, min(2.0, intensity))
        
        # Calculate confidence
        total_score = sum(abs(score) for score in emotion_scores.values())
        confidence = min(1.0, total_score / 5.0) if total_score > 0 else 0.0
        
        return {
            "primary": primary_emotion,
            "intensity": intensity,
            "confidence": confidence,
            "all_scores": dict(emotion_scores)
        }
    
    @staticmethod
    def get_emotional_response_pattern(emotion: str, intensity: float) -> Dict[str, Any]:
        """Get response patterns based on detected emotion"""
        patterns = {
            "joy": {
                "tts_rate": 1.05 + (intensity * 0.05),
                "tts_pitch": 0.5 + (intensity * 0.2),  # Float value
                "tts_volume": 0.8,
                "response_style": "enthusiastic and warm",
                "empathy_level": 0.7
            },
            "sadness": {
                "tts_rate": 0.92 - (abs(intensity) * 0.03),
                "tts_pitch": -0.3 + (intensity * 0.1),  # Float value
                "tts_volume": 0.7,
                "response_style": "compassionate and understanding",
                "empathy_level": 0.9
            },
            "anger": {
                "tts_rate": 0.95,
                "tts_pitch": 0.0,
                "tts_volume": 0.8,
                "response_style": "calm and professional",
                "empathy_level": 0.8
            },
            "fear": {
                "tts_rate": 1.0,
                "tts_pitch": 0.2,
                "tts_volume": 0.75,
                "response_style": "reassuring and clear",
                "empathy_level": 0.85
            },
            "neutral": {
                "tts_rate": 1.0,
                "tts_pitch": 0.0,
                "tts_volume": 0.8,
                "response_style": "professional and helpful",
                "empathy_level": 0.6
            }
        }
        
        return patterns.get(emotion, patterns["neutral"])

# ================== Enhanced TTS with Emotional Intelligence ==================

def create_emotional_ssml(text: str, emotion_data: Dict[str, Any], 
                         language_code: str = "en-IN") -> str:
    """Create SSML with emotional intelligence"""
    
    emotion_pattern = EmotionalIntelligence.get_emotional_response_pattern(
        emotion_data["primary"], emotion_data["intensity"]
    )
    
    # Enhanced text processing for natural pauses
    processed_text = text
    
    # Add natural pauses
    pause_patterns = [
        (", ", ", <break time='150ms'/> "),
        (". ", ". <break time='300ms'/> "),
        ("? ", "? <break time='350ms'/> "),
        ("! ", "! <break time='320ms'/> "),
        (": ", ": <break time='200ms'/> "),
        ("; ", "; <break time='180ms'/> ")
    ]
    
    for pattern, replacement in pause_patterns:
        processed_text = processed_text.replace(pattern, replacement)
    
    # Clean up SSML
    processed_text = processed_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Build SSML with emotional intelligence
    ssml = f"""<speak>
    <prosody rate="{emotion_pattern['tts_rate']:.2f}" 
             pitch="{emotion_pattern['tts_pitch']:.1f}st" 
             volume="{emotion_pattern['tts_volume']}">
        {processed_text}
    </prosody>
    </speak>"""
    
    ssml = re.sub(r'\s+', ' ', ssml).strip()
    return ssml

# ================== Audio Processing ==================

class VoiceActivityDetector:
    """Voice Activity Detection"""
    
    def __init__(self, initial_threshold=VOICE_ACTIVITY_RMS_THRESHOLD):
        self.threshold = initial_threshold
        self.silence_history = deque(maxlen=50)
        self.speech_history = deque(maxlen=50)
        self.state = "silence"
        self.state_duration = 0.0
        self.last_change_time = time.time()
        
    def detect(self, pcm16_data: bytes) -> Tuple[bool, float]:
        """Detect speech in audio data"""
        try:
            rms = audioop.rms(pcm16_data, 2)
        except Exception:
            rms = 0
        
        current_time = time.time()
        time_since_change = current_time - self.last_change_time
        
        # Update state
        if rms > self.threshold:
            if self.state == "silence":
                if time_since_change > 0.05:  # Debounce
                    self.state = "speech"
                    self.last_change_time = current_time
                    self.state_duration = 0.0
            else:
                self.state_duration += time_since_change
        else:
            if self.state == "speech":
                if self.state_duration > MIN_SPEECH_DURATION and time_since_change > MAX_SILENCE_DURATION:
                    self.state = "silence"
                    self.last_change_time = current_time
                    self.state_duration = 0.0
        
        # Update threshold
        is_speech = (rms > self.threshold)
        if is_speech:
            self.speech_history.append(rms)
        else:
            self.silence_history.append(rms)
        
        # Adaptive threshold
        if len(self.silence_history) > 10 and len(self.speech_history) > 5:
            avg_silence = np.mean(list(self.silence_history)[-10:])
            avg_speech = np.mean(list(self.speech_history)[-5:])
            new_threshold = (avg_silence * 0.3 + avg_speech * 0.7) * 0.5
            self.threshold = max(100, min(500, new_threshold))
        
        return is_speech, rms

# ================== STT Service ==================

class STTService:
    """Speech-to-Text Service"""
    
    def __init__(self, language_code: str = "en-IN"):
        self.language_code = language_code
        
    def get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get streaming recognition configuration"""
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_SAMPLE_RATE,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            model="latest_long",
            use_enhanced=True,
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False,
        )
        
        return streaming_config
    
    def create_request_generator(self, audio_queue: queue.Queue, stop_event: threading.Event):
        """Create request generator for streaming recognition"""
        # First request contains the config
        yield speech.StreamingRecognizeRequest(streaming_config=self.get_streaming_config())
        
        # Then yield audio chunks
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
                if chunk is None:  # Sentinel for shutdown
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error getting audio chunk: {e}")
                break

# ================== Natural Language Understanding ==================

class NLUSystem:
    """Natural Language Understanding System"""
    
    def __init__(self, gemini_service: GeminiService):
        self.gemini = gemini_service
        
    async def understand_intent(self, text: str, context: List[Dict]) -> Dict[str, Any]:
        """Understand intent using Gemini"""
        try:
            # Build context string
            context_str = "\n".join([
                f"{ctx['role']}: {ctx['text']}"
                for ctx in context[-5:]
            ]) if context else "No previous context"
            
            prompt = f"""
            Analyze this user message and determine:
            1. Primary intent (greeting, question, command, complaint, inquiry, feedback, other)
            2. Key entities mentioned
            3. Urgency level (1-5)
            4. Emotional tone
            
            Context:
            {context_str}
            
            User: {text}
            
            Respond in JSON format with: primary_intent, entities, urgency, emotional_tone, confidence
            """
            
            response = await self.gemini.generate_response(
                prompt=prompt,
                system_message="You are an expert NLU system. Be precise.",
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse JSON response
            try:
                result = json.loads(response)
            except:
                # Fallback parsing
                result = {
                    "primary_intent": "question",
                    "entities": [],
                    "urgency": 3,
                    "emotional_tone": "neutral",
                    "confidence": 0.7
                }
            
            return result
            
        except Exception as e:
            logger.error(f"NLU error: {e}")
            return {
                "primary_intent": "question",
                "entities": [],
                "urgency": 3,
                "emotional_tone": "neutral",
                "confidence": 0.5
            }

# ================== Natural Response Generation ==================

class NaturalResponseGenerator:
    """Generate natural, human-like responses"""
    
    def __init__(self, gemini_service: GeminiService, business_name: str, persona: str):
        self.gemini = gemini_service
        self.business_name = business_name
        self.persona = persona
        
    async def generate_response(self, 
                               user_message: str,
                               intent: Dict[str, Any],
                               context: List[Dict],
                               rag_results: Optional[List[Dict]] = None,
                               emotion_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate natural response"""
        
        # Build conversation context
        context_str = "\n".join([
            f"{ctx['role']}: {ctx['text']}"
            for ctx in context[-6:]
        ]) if context else "No previous conversation"
        
        # Prepare knowledge base context
        kb_context = ""
        if rag_results and intent.get("primary_intent") in ["question", "inquiry"]:
            kb_context = "\n\nRelevant Information:\n"
            for i, result in enumerate(rag_results[:3], 1):
                kb_context += f"{i}. {result.get('chunk_text', '')}\n"
        
        # Build system message
        emotional_style = ""
        if emotion_data and emotion_data.get("primary") != "neutral":
            emotional_pattern = EmotionalIntelligence.get_emotional_response_pattern(
                emotion_data["primary"], emotion_data["intensity"]
            )
            emotional_style = f" Respond in a {emotional_pattern['response_style']} manner."
        
        system_message = f"""You are {self.persona} for {self.business_name}. 
        Your conversation style: warm, professional, and helpful.{emotional_style}
        
        Guidelines:
        1. Be natural and conversational
        2. Show empathy and understanding
        3. Be concise but complete
        4. Use the user's name if mentioned
        5. End with a helpful question when relevant
        """
        
        # Build prompt
        prompt = f"""
        Conversation History:
        {context_str}
        
        {kb_context}
        
        User's Current Message: {user_message}
        
        User's Emotional State: {emotion_data.get('primary', 'neutral') if emotion_data else 'neutral'}
        
        Your natural, helpful response (2-3 sentences):
        """
        
        try:
            response = await self.gemini.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.7 + (emotion_data.get('intensity', 0) * 0.1 if emotion_data else 0),
                max_tokens=150
            )
            
            # Post-process response
            response = response.strip()
            if not response.endswith(('.', '!', '?')):
                response = response + '.'
            
            return {
                "text": response,
                "word_count": len(response.split()),
                "emotion": emotion_data,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            # Fallback responses
            fallbacks = [
                "I understand. Let me help you with that.",
                "Thanks for sharing that. Here's what I can tell you.",
                "I appreciate you asking. Based on what I know...",
                "That's a good question. Let me provide some information."
            ]
            return {
                "text": random.choice(fallbacks),
                "word_count": 10,
                "emotion": {"primary": "neutral", "intensity": 0.0},
                "timestamp": time.time()
            }

# ================== Conversation Memory ==================

class ConversationMemory:
    """Conversation memory with context retention"""
    
    def __init__(self, max_turns: int = 20):
        self.history = deque(maxlen=max_turns)
        self.user_preferences = {}
        self.mentioned_topics = set()
        
    def add_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add conversation turn"""
        entry = {
            "role": role,
            "text": text,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.history.append(entry)
        
        # Extract user preferences
        if role == "user":
            self._extract_preferences(text)
    
    def get_context(self, turns: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        return list(self.history)[-turns:]
    
    def _extract_preferences(self, text: str):
        """Extract user preferences from text"""
        text_lower = text.lower()
        if "morning" in text_lower:
            self.user_preferences["time_preference"] = "morning"
        if "afternoon" in text_lower:
            self.user_preferences["time_preference"] = "afternoon"
        if "evening" in text_lower:
            self.user_preferences["time_preference"] = "evening"

# ================== TTS Synthesis ==================

async def synthesize_with_emotion(text: str, 
                                 language_code: str = "en-IN",
                                 emotion_data: Optional[Dict] = None,
                                 voice_config: Optional[Dict] = None) -> Optional[bytes]:
    """TTS synthesis with emotional intelligence"""
    
    # Get voice configuration
    if voice_config is None:
        voice_config = VOICE_MAP.get(language_code, DEFAULT_VOICE)
    
    # Create SSML with emotion
    if emotion_data:
        ssml_text = create_emotional_ssml(text, emotion_data, language_code)
    else:
        ssml_text = f"<speak>{text}</speak>"
    
    # Voice selection
    voice_selection = tts.VoiceSelectionParams(
        language_code=language_code,
        name=voice_config["name"],
        ssml_gender=tts.SsmlVoiceGender.MALE if voice_config.get("gender") == "MALE" else tts.SsmlVoiceGender.FEMALE
    )
    
    # Audio configuration - FIXED: Using float for pitch
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=TTS_SAMPLE_RATE,
        speaking_rate=float(voice_config.get("tempo", 1.0)),  # Ensure float
        pitch=float(voice_config.get("pitch_range", 0.0)),    # Ensure float
    )
    
    synthesis_input = tts.SynthesisInput(ssml=ssml_text)
    
    try:
        async with tts_semaphore:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: _tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice_selection,
                    audio_config=audio_config
                )
            )
        
        return response.audio_content
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return None

def create_wav_from_pcm(pcm_bytes: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Create WAV from PCM bytes"""
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    
    return buffer.getvalue()

# ================== Main WebSocket Handler ==================

@router.websocket("/ws/hd-audio")
async def hd_audio_websocket(websocket: WebSocket):
    """WebSocket handler for voice conversations"""
    
    # Authentication
    token = websocket.query_params.get("token")
    if WEBSOCKET_API_TOKEN and token != WEBSOCKET_API_TOKEN:
        await websocket.close(code=1008)
        return
    
    await websocket.accept()
    logger.info("✅ New HD voice connection established")
    
    # Initialize services
    gemini_service = GeminiService()
    nlu_system = NLUSystem(gemini_service)
    response_generator = NaturalResponseGenerator(
        gemini_service, 
        BUSINESS_NAME, 
        ASSISTANT_PERSONA
    )
    conversation_memory = ConversationMemory(MAX_CONVERSATION_TURNS)
    vad_detector = VoiceActivityDetector()
    stt_service = STTService()
    
    # State management
    audio_queue = queue.Queue(maxsize=500)
    transcripts_queue = asyncio.Queue()
    stop_event = threading.Event()
    
    # TTS management
    tts_queue = asyncio.Queue()
    is_speaking = False
    
    # Conversation state
    current_language = "en-IN"
    utterance_buffer = []
    last_activity_time = time.time()
    
    async def tts_worker():
        """TTS worker that processes TTS requests"""
        while True:
            try:
                # Get TTS request
                try:
                    request = await asyncio.wait_for(tts_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.debug("TTS worker idle timeout")
                    break
                
                if request is None:  # Shutdown signal
                    break
                
                text = request.get("text", "")
                language = request.get("language", current_language)
                emotion = request.get("emotion", {})
                
                if not text:
                    tts_queue.task_done()
                    continue
                
                # Synthesize speech
                pcm_audio = await synthesize_with_emotion(
                    text=text,
                    language_code=language,
                    emotion_data=emotion
                )
                
                if pcm_audio:
                    # Convert to WAV and send
                    wav_bytes = create_wav_from_pcm(pcm_audio)
                    b64_audio = base64.b64encode(wav_bytes).decode('ascii')
                    
                    await websocket.send_json({
                        "type": "audio",
                        "audio": b64_audio,
                        "metadata": {
                            "text_length": len(text),
                            "emotion": emotion.get("primary", "neutral")
                        }
                    })
                
                tts_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
                tts_queue.task_done()
                continue
    
    async def process_transcripts():
        """Process STT transcripts and generate responses"""
        while not stop_event.is_set():
            try:
                # Get transcript
                transcript_data = await asyncio.wait_for(
                    transcripts_queue.get(), 
                    timeout=1.0
                )
                
                if transcript_data is None:
                    break
                
                # Process transcript
                transcript = transcript_data.get("text", "").strip()
                is_final = transcript_data.get("is_final", False)
                confidence = transcript_data.get("confidence", 0.0)
                
                if not transcript:
                    continue
                
                # Send interim transcript to client
                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                    "is_final": is_final,
                    "confidence": confidence
                })
                
                # Process final transcript
                if is_final and confidence > 0.6:
                    logger.info(f"📝 Final transcript: {transcript}")
                    
                    # Analyze emotion
                    emotion_data = EmotionalIntelligence.analyze_emotion(transcript)
                    
                    # Add to conversation memory
                    conversation_memory.add_turn(
                        role="user",
                        text=transcript,
                        metadata={
                            "emotion": emotion_data,
                            "confidence": confidence
                        }
                    )
                    
                    # Get conversation context
                    context = conversation_memory.get_context(turns=5)
                    
                    # NLU understanding
                    intent_data = await nlu_system.understand_intent(transcript, context)
                    
                    # Perform RAG search for relevant information
                    rag_results = None
                    if intent_data.get("primary_intent") in ["question", "inquiry"]:
                        try:
                            rag_results = await pinecone_service.search_similar_chunks(
                                client_id=DEFAULT_CLIENT_ID,
                                query=transcript,
                                top_k=5,
                                min_score=0.3
                            )
                        except Exception as e:
                            logger.error(f"RAG search error: {e}")
                            rag_results = None
                    
                    # Generate natural response
                    response_data = await response_generator.generate_response(
                        user_message=transcript,
                        intent=intent_data,
                        context=context,
                        rag_results=rag_results,
                        emotion_data=emotion_data
                    )
                    
                    response_text = response_data["text"]
                    
                    # Add to conversation memory
                    conversation_memory.add_turn(
                        role="assistant",
                        text=response_text,
                        metadata={
                            "intent": intent_data,
                            "rag_used": rag_results is not None
                        }
                    )
                    
                    # Send text response
                    await websocket.send_json({
                        "type": "ai_text",
                        "text": response_text,
                        "metadata": {
                            "intent": intent_data.get("primary_intent"),
                            "emotion": emotion_data,
                        }
                    })
                    
                    # Queue TTS with emotion
                    await tts_queue.put({
                        "text": response_text,
                        "language": current_language,
                        "emotion": emotion_data
                    })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Transcript processing error: {e}")
                continue
    
    def stt_worker():
        """STT worker thread - FIXED API CALL"""
        try:
            # Create request generator
            requests = stt_service.create_request_generator(audio_queue, stop_event)
            
            # Start streaming recognition - FIXED: Correct API call
            responses = _speech_client.streaming_recognize(requests)
            
            for response in responses:
                if stop_event.is_set():
                    break
                
                if not response.results:
                    continue
                
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        confidence = alternative.confidence if alternative.confidence else 0.0
                        
                        # Queue for processing
                        asyncio.run_coroutine_threadsafe(
                            transcripts_queue.put({
                                "text": transcript,
                                "is_final": result.is_final,
                                "confidence": confidence
                            }),
                            asyncio.get_event_loop()
                        )
        
        except Exception as e:
            logger.error(f"STT worker error: {e}")
            # Try to restart STT
            if not stop_event.is_set():
                logger.info("Restarting STT worker...")
                time.sleep(1)
                stt_worker()
    
    # Start workers
    tts_worker_task = asyncio.create_task(tts_worker())
    transcript_processor_task = asyncio.create_task(process_transcripts())
    
    # Start STT thread
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()
    
    # Send welcome message
    welcome_message = f"Hello! I'm your {ASSISTANT_PERSONA} from {BUSINESS_NAME}. How can I help you today?"
    await websocket.send_json({
        "type": "ai_text",
        "text": welcome_message,
        "metadata": {"welcome": True}
    })
    
    await tts_queue.put({
        "text": welcome_message,
        "language": current_language,
        "emotion": {"primary": "joy", "intensity": 0.5, "confidence": 0.8}
    })
    
    # Main WebSocket message loop
    try:
        while True:
            # Receive message
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                logger.info("WebSocket disconnected by client")
                break
            
            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                
                # Voice activity detection
                is_speech, rms_value = vad_detector.detect(audio_data)
                
                if is_speech:
                    last_activity_time = time.time()
                    
                    # Send throttle control if bot is speaking
                    if is_speaking:
                        await websocket.send_json({
                            "type": "control",
                            "action": "throttle"
                        })
                    
                    # Queue audio for STT
                    try:
                        audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")
                    
                    # Send audio metrics
                    await websocket.send_json({
                        "type": "audio_level",
                        "level": rms_value,
                        "threshold": vad_detector.threshold
                    })
            
            # Handle text messages (control)
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "start":
                        # Handle start with language
                        language = data.get("meta", {}).get("language", "en-IN")
                        if language in VOICE_MAP:
                            current_language = language
                            stt_service.language_code = language
                        
                        await websocket.send_json({
                            "type": "ready",
                            "language": current_language,
                            "persona": ASSISTANT_PERSONA
                        })
                    
                    elif msg_type == "stop":
                        logger.info("Client requested stop")
                        break
                    
                    elif msg_type == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": time.time()
                        })
                
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received")
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        # Cleanup
        logger.info("Starting cleanup")
        
        # Signal shutdown
        stop_event.set()
        
        # Stop audio processing
        try:
            audio_queue.put(None, block=False)
        except:
            pass
        
        # Stop TTS worker
        if tts_worker_task and not tts_worker_task.done():
            try:
                await tts_queue.put(None)
                await asyncio.wait_for(tts_worker_task, timeout=2.0)
            except:
                tts_worker_task.cancel()
        
        # Stop transcript processor
        if transcript_processor_task and not transcript_processor_task.done():
            transcript_processor_task.cancel()
        
        # Wait for STT thread
        if stt_thread.is_alive():
            stt_thread.join(timeout=2.0)
        
        # Close WebSocket
        try:
            await websocket.close()
        except:
            pass
        
        logger.info("Cleanup complete")
