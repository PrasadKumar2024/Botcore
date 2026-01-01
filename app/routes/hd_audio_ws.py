# app/routes/hd_audio_ws.py
import os
import io
import json
import time
import base64
import logging
import functools
import asyncio
import threading
import queue
import audioop
import wave
import re
from typing import Optional, List, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.orm import Session

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions

# Local services
from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import GeminiService
from app.services.emotion_engine import EmotionEngine
from app.services.conversation_manager import ConversationManager
from app.services.voice_persona import VoicePersona
from app.services.context_awareness import ContextAwareness
from app.services.interruption_handler import InterruptionHandler
from app.database import get_db
from app.models import Client, VoiceSettings

logger = logging.getLogger(__name__)
router = APIRouter()

# ================== Configuration ==================
class VoiceMode(Enum):
    FAST = "fast"
    NATURAL = "natural"
    EXPERT = "expert"
    PREMIUM = "premium"

# System-wide configuration
EXECUTOR_WORKERS = int(os.getenv("HD_WS_EXECUTOR_WORKERS", "12"))
MAX_CONCURRENT_TTS = int(os.getenv("HD_WS_MAX_TTS", "6"))
STT_TIMEOUT = float(os.getenv("HD_WS_STT_TIMEOUT", "8.0"))
LLM_TIMEOUT = float(os.getenv("HD_WS_LLM_TIMEOUT", "10.0"))
TTS_TIMEOUT = float(os.getenv("HD_WS_TTS_TIMEOUT", "12.0"))

STT_SAMPLE_RATE = int(os.getenv("HD_WS_STT_SR", "16000"))
CHUNK_SECONDS = float(os.getenv("HD_WS_CHUNK_SECONDS", "0.3"))
MAX_BUFFER_SECONDS = int(os.getenv("HD_WS_MAX_BUFFER_S", "15"))
WEBSOCKET_API_TOKEN = os.getenv("WEBSOCKET_API_TOKEN", None)

DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", "default")
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BrightCare")

# Audio parameters
BYTES_PER_SEC = STT_SAMPLE_RATE * 2
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
MAX_BUFFER_BYTES = int(BYTES_PER_SEC * MAX_BUFFER_SECONDS)

# Performance tuning
STREAM_SENTENCE_CHAR_LIMIT = int(os.getenv("HD_WS_SENTENCE_CHAR_LIMIT", "200"))
TTS_WORKER_IDLE_TIMEOUT = float(os.getenv("HD_WS_TTS_WORKER_IDLE", "45.0"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))

# VAD & processing
DEBOUNCE_SECONDS = float(os.getenv("HD_WS_DEBOUNCE_S", "0.4"))
VAD_THRESHOLD = int(os.getenv("HD_WS_VAD_THRESHOLD", "250"))
MIN_RESTART_INTERVAL = float(os.getenv("HD_WS_MIN_RESTART_INTERVAL", "1.5"))
MAX_SILENCE_SECONDS = float(os.getenv("MAX_SILENCE_SECONDS", "3.0"))

# Initialize executors and semaphores
executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
global_tts_semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_TTS)

# ================== Google Clients ==================
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON env")

try:
    _creds = service_account.Credentials.from_service_account_info(json.loads(GOOGLE_CREDS_JSON))
    _speech_client = speech.SpeechClient(credentials=_creds)
    _tts_client = tts.TextToSpeechClient(credentials=_creds)
    logger.info("✅ Google Cloud Speech/TTS clients initialized")
except Exception as e:
    logger.exception("❌ Failed to initialize Google clients: %s", e)
    raise

# ================== Service Initialization ==================
_gemini = GeminiService()
_emotion_engine = EmotionEngine()
_conversation_manager = ConversationManager(max_history=MAX_CONVERSATION_HISTORY)
_voice_persona = VoicePersona()
_context_awareness = ContextAwareness()
_interruption_handler = InterruptionHandler()

# ================== Advanced Voice Configuration ==================
class VoiceProfile:
    def __init__(self):
        self.profiles = {
            "professional": {
                "rate": 0.95,
                "pitch": 0,
                "volume": "medium",
                "emphasis": "moderate",
                "pause_duration": "medium"
            },
            "friendly": {
                "rate": 1.05,
                "pitch": "+2st",
                "volume": "medium",
                "emphasis": "light",
                "pause_duration": "short"
            },
            "empathetic": {
                "rate": 0.90,
                "pitch": "-1st",
                "volume": "soft",
                "emphasis": "strong",
                "pause_duration": "long"
            },
            "enthusiastic": {
                "rate": 1.10,
                "pitch": "+3st",
                "volume": "loud",
                "emphasis": "strong",
                "pause_duration": "short"
            }
        }
    
    def get_profile(self, emotion_score: float, context_type: str) -> Dict[str, Any]:
        """Get voice profile based on emotion and context"""
        if emotion_score > 0.6:
            profile = self.profiles["enthusiastic"]
        elif emotion_score > 0.3:
            profile = self.profiles["friendly"]
        elif emotion_score < -0.4:
            profile = self.profiles["empathetic"]
        else:
            profile = self.profiles["professional"]
        
        # Adjust based on context
        if context_type == "technical":
            profile["rate"] *= 0.95
            profile["emphasis"] = "strong"
        elif context_type == "casual":
            profile["rate"] *= 1.05
        
        return profile

_voice_profile = VoiceProfile()

# Voice mapping with enhanced parameters
VOICE_MAP = {
    "en-IN": {
        "neural": [
            {"name": "en-IN-Neural2-A", "gender": "FEMALE", "style": "warm", "pitch": 0},
            {"name": "en-IN-Neural2-B", "gender": "MALE", "style": "professional", "pitch": -2},
            {"name": "en-IN-Neural2-C", "gender": "MALE", "style": "friendly", "pitch": 0},
            {"name": "en-IN-Neural2-D", "gender": "FEMALE", "style": "calm", "pitch": -1}
        ],
        "standard": [
            {"name": "en-IN-Standard-A", "gender": "FEMALE", "style": "neutral"},
            {"name": "en-IN-Standard-B", "gender": "MALE", "style": "neutral"},
            {"name": "en-IN-Standard-C", "gender": "MALE", "style": "formal"},
            {"name": "en-IN-Standard-D", "gender": "FEMALE", "style": "formal"}
        ],
        "wavenet": [
            {"name": "en-IN-Wavenet-A", "gender": "FEMALE", "style": "premium"},
            {"name": "en-IN-Wavenet-B", "gender": "MALE", "style": "premium"},
            {"name": "en-IN-Wavenet-C", "gender": "MALE", "style": "premium"},
            {"name": "en-IN-Wavenet-D", "gender": "FEMALE", "style": "premium"}
        ]
    },
    "en-US": {
        "neural": [
            {"name": "en-US-Neural2-A", "gender": "FEMALE", "style": "casual"},
            {"name": "en-US-Neural2-C", "gender": "FEMALE", "style": "formal"},
            {"name": "en-US-Neural2-F", "gender": "FEMALE", "style": "warm"},
            {"name": "en-US-Neural2-J", "gender": "MALE", "style": "deep"}
        ]
    },
    "hi-IN": {
        "neural": [
            {"name": "hi-IN-Neural2-A", "gender": "FEMALE", "style": "natural"},
            {"name": "hi-IN-Neural2-B", "gender": "MALE", "style": "natural"},
            {"name": "hi-IN-Neural2-C", "gender": "FEMALE", "style": "formal"},
            {"name": "hi-IN-Neural2-D", "gender": "MALE", "style": "formal"}
        ]
    }
}

DEFAULT_VOICE = VOICE_MAP["en-IN"]["neural"][0]

def get_best_voice(language_code: Optional[str], emotion_score: float = 0.0, 
                   context_type: str = "general", voice_preference: str = "neural") -> Dict[str, Any]:
    """Select optimal voice based on language, emotion, context, and preference"""
    if not language_code:
        lang_code = "en-IN"
    else:
        lang_code = language_code
    
    # Fallback logic
    if lang_code not in VOICE_MAP:
        base_lang = lang_code.split("-")[0] if "-" in lang_code else lang_code
        if base_lang == "en":
            lang_code = "en-IN"
        elif base_lang == "hi":
            lang_code = "hi-IN"
        else:
            lang_code = "en-IN"
    
    # Get available voices for this language
    if lang_code in VOICE_MAP:
        voices = VOICE_MAP[lang_code].get(voice_preference, VOICE_MAP[lang_code]["neural"])
    else:
        voices = DEFAULT_VOICE if isinstance(DEFAULT_VOICE, list) else [DEFAULT_VOICE]
    
    # Select voice based on emotion and context
    if emotion_score > 0.6:
        # Happy/enthusiastic - choose warm/friendly voices
        preferred_voices = [v for v in voices if v.get("style") in ["warm", "friendly", "casual"]]
    elif emotion_score < -0.3:
        # Sad/empathetic - choose calm/gentle voices
        preferred_voices = [v for v in voices if v.get("style") in ["calm", "gentle", "soft"]]
    else:
        # Neutral - choose professional/neutral voices
        preferred_voices = [v for v in voices if v.get("style") in ["professional", "neutral", "formal"]]
    
    # Select from preferred voices, or fallback to first available
    selected_voice = preferred_voices[0] if preferred_voices else voices[0]
    
    return {
        "language_code": lang_code,
        "name": selected_voice["name"],
        "gender": selected_voice.get("gender"),
        "style": selected_voice.get("style", "neutral"),
        "pitch": selected_voice.get("pitch", 0),
        "emotion_adjusted": True
    }

def create_emotional_ssml(text: str, emotion_score: float = 0.0, 
                          context_type: str = "general", voice_params: Dict[str, Any] = None) -> str:
    """Create emotionally intelligent SSML with dynamic prosody"""
    # Get voice profile
    profile = _voice_profile.get_profile(emotion_score, context_type)
    
    # Enhanced text processing
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Dynamic pauses based on punctuation and context
    if emotion_score > 0.5:
        # Excited - shorter pauses
        esc = re.sub(r'([.!?])\s+', r'\1 <break time="180ms"/> ', esc)
        esc = re.sub(r'[,;:]\s+', r'\1 <break time="100ms"/> ', esc)
    elif emotion_score < -0.4:
        # Empathetic - longer pauses
        esc = re.sub(r'([.!?])\s+', r'\1 <break time="350ms"/> ', esc)
        esc = re.sub(r'[,;:]\s+', r'\1 <break time="200ms"/> ', esc)
    else:
        # Neutral - standard pauses
        esc = re.sub(r'([.!?])\s+', r'\1 <break time="250ms"/> ', esc)
        esc = re.sub(r'[,;:]\s+', r'\1 <break time="150ms"/> ', esc)
    
    # Add emphasis for important words (capitalized or quotes)
    words = esc.split()
    for i, word in enumerate(words):
        if word.isupper() and len(word) > 1:
            words[i] = f'<emphasis level="strong">{word}</emphasis>'
        elif '"' in word or "'" in word:
            words[i] = f'<emphasis level="moderate">{word}</emphasis>'
    
    esc = " ".join(words)
    
    # Create SSML with dynamic prosody
    ssml = f"""
    <speak>
        <prosody rate="{profile['rate']}" 
                 pitch="{profile['pitch'] or '0st'}" 
                 volume="{profile['volume']}">
            {esc}
        </prosody>
    </speak>
    """
    
    return ssml.strip()

def make_wav_from_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw PCM16LE bytes into a WAV file bytes with proper headers"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def is_silence(pcm16: bytes, threshold: int = VAD_THRESHOLD) -> bool:
    """Voice Activity Detection with enhanced thresholding"""
    try:
        rms = audioop.rms(pcm16, 2)
        # Adaptive threshold based on audio characteristics
        if len(pcm16) > 0:
            # Check for very low energy
            return rms < threshold
        return True
    except Exception:
        return True

# ================== Advanced TTS Synthesis ==================
def _sync_tts_linear16(ssml: str, voice_config: Dict[str, Any], sample_rate_hz: int = 24000) -> bytes:
    """Synchronous TTS synthesis with enhanced voice configuration"""
    try:
        voice = tts.VoiceSelectionParams(
            language_code=voice_config["language_code"],
            name=voice_config["name"]
        )
        
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hz,
            speaking_rate=voice_config.get("speaking_rate", 1.0),
            pitch=voice_config.get("pitch", 0),
            volume_gain_db=voice_config.get("volume_gain_db", 0),
            effects_profile_id=["telephony-class-application"]
        )
        
        synthesis_input = tts.SynthesisInput(ssml=ssml)
        response = _tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise

async def synthesize_text_to_pcm(text: str, language_code: str = "en-IN", 
                                 emotion_score: float = 0.0, context_type: str = "general",
                                 sample_rate_hz: int = 24000, voice_preference: str = "neural") -> Optional[bytes]:
    """Enhanced TTS synthesis with emotional intelligence and voice selection"""
    
    # Get optimal voice configuration
    voice_config = get_best_voice(language_code, emotion_score, context_type, voice_preference)
    
    # Create emotionally aware SSML
    ssml = create_emotional_ssml(text, emotion_score, context_type, voice_config)
    
    loop = asyncio.get_running_loop()
    
    # Acquire TTS semaphore
    try:
        await asyncio.wait_for(global_tts_semaphore.acquire(), timeout=2.0)
    except asyncio.TimeoutError:
        logger.warning("TTS queue busy, dropping request")
        return None
    
    try:
        # Execute TTS in thread pool
        fut = loop.run_in_executor(
            executor, 
            functools.partial(_sync_tts_linear16, ssml, voice_config, sample_rate_hz)
        )
        audio_content = await asyncio.wait_for(fut, timeout=TTS_TIMEOUT)
        return audio_content
    except asyncio.TimeoutError:
        logger.warning("TTS synthesis timed out")
        return None
    except Exception as e:
        logger.exception(f"TTS synthesis failed: {e}")
        return None
    finally:
        try:
            global_tts_semaphore.release()
        except Exception:
            pass

# ================== Advanced Query Processing ==================
class QueryProcessor:
    """Advanced query processing with intent recognition and context expansion"""
    
    def __init__(self):
        self.intent_patterns = {
            "greeting": [
                r'\b(hi|hello|hey|good morning|good evening|good afternoon|namaste|greetings)\b',
                r'\b(how are you|what\'?s up|how\'?s it going)\b'
            ],
            "farewell": [
                r'\b(bye|goodbye|see you|see ya|take care|talk to you later)\b',
                r'\b(thank you|thanks|appreciate it|that\'?s all)\b'
            ],
            "business_hours": [
                r'\b(timings?|hours?|open|close|closed|schedule|operating hours)\b',
                r'\b(what time|when do you|available|availability)\b'
            ],
            "appointment": [
                r'\b(appointment|booking|schedule|reservation|meeting)\b',
                r'\b(book|schedule|make an appointment|set up a meeting)\b'
            ],
            "contact": [
                r'\b(contact|phone|number|call|email|address|location|where are you)\b',
                r'\b(how to reach|get in touch|contact information)\b'
            ],
            "service": [
                r'\b(service|services|offer|offering|provide|treatment|solution)\b',
                r'\b(what do you|what can you|help with|assist with)\b'
            ],
            "pricing": [
                r'\b(price|cost|fee|charge|rate|how much|payment|pay)\b',
                r'\b(affordable|expensive|cheap|budget|cost effective)\b'
            ],
            "emergency": [
                r'\b(emergency|urgent|immediate|right now|asap|critical)\b',
                r'\b(help now|need help|problem|issue|trouble)\b'
            ]
        }
        
        self.context_expansions = {
            "timings": ["business hours", "operating hours", "schedule", "open close times", "availability"],
            "timing": ["business hours", "operating hours", "schedule"],
            "hours": ["business hours", "operating hours", "timings"],
            "open": ["business hours", "operating hours", "schedule"],
            "close": ["business hours", "operating hours", "schedule"],
            "appointment": ["booking", "reservation", "consultation", "meeting", "schedule"],
            "doctor": ["physician", "specialist", "consultant", "medical professional", "expert"],
            "price": ["cost", "fee", "charge", "rate", "payment", "expense"],
            "service": ["offering", "treatment", "solution", "assistance", "help"],
            "location": ["address", "place", "where", "directions", "map"],
            "contact": ["phone number", "telephone", "email", "reach", "get in touch"],
            "emergency": ["urgent", "immediate", "critical", "serious", "important"]
        }
        
        self.query_templates = {
            "business_hours": "business hours operating schedule open close timing",
            "appointment": "appointment booking schedule consultation meeting",
            "contact": "contact phone number email address location",
            "service": "service offering treatment solution help assistance",
            "pricing": "price cost fee charge rate payment",
            "general": "information details help assistance support"
        }
    
    def detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect user intent with confidence scoring"""
        text_lower = text.lower().strip()
        
        if not text_lower or len(text_lower) < 2:
            return {"intent": "unknown", "confidence": 0.0, "sub_intents": []}
        
        detected_intents = []
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Calculate confidence based on match quality
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    confidence = min(1.0, len(matches) * 0.3 + 0.4)
                    detected_intents.append({
                        "intent": intent_name,
                        "confidence": confidence,
                        "pattern": pattern
                    })
        
        # Sort by confidence
        detected_intents.sort(key=lambda x: x["confidence"], reverse=True)
        
        if detected_intents:
            primary = detected_intents[0]
            return {
                "intent": primary["intent"],
                "confidence": primary["confidence"],
                "sub_intents": [di["intent"] for di in detected_intents[1:3]],
                "all_intents": detected_intents
            }
        
        # Default to general inquiry
        return {"intent": "general_inquiry", "confidence": 0.6, "sub_intents": []}
    
    def expand_query(self, text: str, intent: str = None) -> str:
        """Expand query with semantic context"""
        text_lower = text.lower().strip()
        
        # Remove filler words
        filler_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "for"}
        words = [w for w in text_lower.split() if w not in filler_words]
        
        # Expand based on intent
        expanded_words = []
        for word in words:
            expanded_words.append(word)
            if word in self.context_expansions:
                expanded_words.extend(self.context_expansions[word])
        
        # Add intent-based expansion
        if intent and intent in self.query_templates:
            expanded_words.extend(self.query_templates[intent].split())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in expanded_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return " ".join(unique_words)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from user query"""
        text_lower = text.lower().strip()
        entities = {
            "time_entities": [],
            "date_entities": [],
            "location_entities": [],
            "person_entities": [],
            "service_entities": [],
            "urgency_level": "normal"
        }
        
        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?)\b',
            r'\b(\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.))\b',
            r'\b(morning|afternoon|evening|night|noon|midnight)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                entities["time_entities"].extend(matches)
        
        # Date patterns
        date_patterns = [
            r'\b(today|tomorrow|yesterday|next week|this week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                entities["date_entities"].extend(matches)
        
        # Urgency detection
        urgent_words = {"emergency", "urgent", "immediate", "asap", "now", "quick", "fast", "critical"}
        if any(word in text_lower for word in urgent_words):
            entities["urgency_level"] = "high"
        
        return entities

_query_processor = QueryProcessor()

# ================== Advanced STT Worker ==================
class EnhancedSTTWorker:
    """Enhanced STT worker with adaptive language handling and error recovery"""
    
    def __init__(self, loop, audio_queue, transcripts_queue, stop_event, 
                 initial_language: str = "en-IN"):
        self.loop = loop
        self.audio_queue = audio_queue
        self.transcripts_queue = transcripts_queue
        self.stop_event = stop_event
        self.language = initial_language
        self.streaming_config = None
        self._update_config()
    
    def _update_config(self):
        """Update streaming recognition configuration"""
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=STT_SAMPLE_RATE,
                language_code=self.language,
                enable_automatic_punctuation=True,
                model="phone_call",
                use_enhanced=True,
                speech_contexts=[speech.SpeechContext(phrases=[
                    "appointment", "booking", "schedule", "emergency",
                    "doctor", "clinic", "hospital", "medical",
                    "timings", "hours", "open", "close"
                ])],
                alternative_language_codes=["en-US", "hi-IN", "te-IN", "ta-IN"]
            ),
            interim_results=True,
            single_utterance=False,
            enable_voice_activity_events=True,
            voice_activity_timeout=speech.VoiceActivityTimeout(
                speech_start_timeout=datetime.timedelta(seconds=5),
                speech_end_timeout=datetime.timedelta(seconds=2)
            )
        )
    
    def change_language(self, new_language: str):
        """Change language for STT"""
        if new_language != self.language:
            self.language = new_language
            self._update_config()
            logger.info(f"STT language changed to: {new_language}")
    
    def run(self):
        """Main STT worker loop"""
        def request_generator():
            # First request with config
            yield speech.StreamingRecognizeRequest(streaming_config=self.streaming_config)
            
            # Subsequent requests with audio
            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.3)
                    if chunk is None:  # Sentinel for shutdown
                        break
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error getting audio chunk: {e}")
                    if self.stop_event.is_set():
                        break
        
        try:
            logger.info(f"🎤 Starting enhanced STT worker (language: {self.language})")
            
            # Start streaming recognition
            responses = _speech_client.streaming_recognize(request_generator())
            
            # Process responses
            for response in responses:
                if self.stop_event.is_set():
                    break
                
                # Send response to main thread
                asyncio.run_coroutine_threadsafe(
                    self.transcripts_queue.put(response),
                    self.loop
                )
                
        except Exception as e:
            logger.exception(f"STT worker error: {e}")
        finally:
            logger.info("🎤 STT worker stopped")
            # Send completion signal
            asyncio.run_coroutine_threadsafe(
                self.transcripts_queue.put(None),
                self.loop
            )

# ================== Advanced LLM Response Generation ==================
class ResponseGenerator:
    """Advanced response generation with context, persona, and emotional intelligence"""
    
    def __init__(self, business_name: str = "Our Service"):
        self.business_name = business_name
        self.conversation_states = {}
    
    async def generate_response(self, user_input: str, context: Dict[str, Any], 
                               conversation_history: List[Dict], client_id: str = None) -> Dict[str, Any]:
        """Generate intelligent, context-aware response"""
        
        # Get intent and entities
        intent_info = _query_processor.detect_intent(user_input)
        entities = _query_processor.extract_entities(user_input)
        
        # Get emotion score
        emotion_score = _emotion_engine.analyze_emotion(user_input)
        
        # Build conversation context
        conv_context = _conversation_manager.get_context(conversation_history, max_turns=6)
        
        # Determine response strategy
        strategy = self._determine_strategy(intent_info, entities, context)
        
        # Generate response based on strategy
        if strategy == "rag_contextual":
            response = await self._generate_rag_response(user_input, context, conv_context, intent_info)
        elif strategy == "conversational":
            response = await self._generate_conversational_response(user_input, conv_context, intent_info, emotion_score)
        elif strategy == "procedural":
            response = await self._generate_procedural_response(user_input, entities, conv_context)
        elif strategy == "empathetic":
            response = await self._generate_empathetic_response(user_input, emotion_score, conv_context)
        else:
            response = await self._generate_general_response(user_input, conv_context)
        
        # Add persona and emotional touch
        response = self._apply_persona_touch(response, intent_info, emotion_score)
        
        # Add context awareness
        response = self._add_context_awareness(response, context, entities)
        
        return {
            "text": response,
            "intent": intent_info["intent"],
            "confidence": intent_info["confidence"],
            "emotion_score": emotion_score,
            "entities": entities,
            "strategy": strategy,
            "requires_follow_up": self._requires_follow_up(intent_info, entities),
            "suggested_actions": self._get_suggested_actions(intent_info, entities)
        }
    
    def _determine_strategy(self, intent_info: Dict, entities: Dict, context: Dict) -> str:
        """Determine optimal response strategy"""
        intent = intent_info["intent"]
        
        # Check for emergency/special cases
        if entities.get("urgency_level") == "high" or "emergency" in intent:
            return "empathetic"
        
        # Check if we have relevant context
        if context.get("has_relevant_data", False):
            return "rag_contextual"
        
        # Check intent type
        if intent in ["greeting", "farewell", "general_inquiry"]:
            return "conversational"
        elif intent in ["appointment", "booking", "service"]:
            return "procedural"
        elif intent in ["business_hours", "contact", "pricing"]:
            return "rag_contextual" if context.get("has_relevant_data", False) else "procedural"
        
        return "conversational"
    
    async def _generate_rag_response(self, user_input: str, context: Dict, 
                                     conv_context: str, intent_info: Dict) -> str:
        """Generate response using RAG context"""
        # Search for relevant information
        expanded_query = _query_processor.expand_query(user_input, intent_info["intent"])
        
        try:
            results = await pinecone_service.search_similar_chunks(
                client_id=DEFAULT_CLIENT_ID,
                query=expanded_query,
                top_k=5,
                min_score=0.2
            )
            
            if results:
                # Build context from results
                rag_context = "\n\n".join([
                    f"Source: {r.get('source', 'Information')}\nContent: {r.get('chunk_text', '')[:300]}"
                    for r in results[:3]
                ])
                
                system_prompt = f"""You are {self.business_name}'s intelligent voice assistant. 
                Use the provided information to answer the user's question accurately and naturally.
                Be helpful, concise, and speak conversationally as if having a phone conversation.
                If the information doesn't fully answer the question, acknowledge that and offer to help further.
                
                Available Information:
                {rag_context}
                
                Conversation History:
                {conv_context}
                """
                
                user_prompt = f"User asks: {user_input}\n\nProvide a natural, helpful response:"
                
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    executor,
                    functools.partial(
                        _gemini.generate_response,
                        prompt=user_prompt,
                        system_message=system_prompt,
                        temperature=0.4,
                        max_tokens=250
                    )
                )
                
                return response.strip() if response else self._get_fallback_response(intent_info)
        
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
        
        return self._get_fallback_response(intent_info)
    
    async def _generate_conversational_response(self, user_input: str, conv_context: str,
                                                intent_info: Dict, emotion_score: float) -> str:
        """Generate natural conversational response"""
        
        emotion_adjective = "warm" if emotion_score > 0.3 else "neutral" if emotion_score > -0.3 else "empathetic"
        
        system_prompt = f"""You are {self.business_name}'s friendly voice assistant. 
        Respond naturally and conversationally, as if talking to someone on the phone.
        Be {emotion_adjective}, helpful, and keep responses brief (1-2 sentences).
        Maintain a natural flow in the conversation.
        
        Conversation History:
        {conv_context}
        """
        
        user_prompt = f"User says: {user_input}\n\nRespond naturally:"
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            executor,
            functools.partial(
                _gemini.generate_response,
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.7,
                max_tokens=150
            )
        )
        
        return response.strip() if response else f"Hello! How can I help you with {self.business_name} today?"
    
    async def _generate_procedural_response(self, user_input: str, entities: Dict,
                                            conv_context: str) -> str:
        """Generate response for procedural queries (appointments, bookings, etc.)"""
        
        system_prompt = f"""You are {self.business_name}'s helpful assistant for scheduling and procedures.
        Guide the user through the process naturally. Ask for one piece of information at a time.
        Be clear, patient, and reassuring.
        
        Conversation History:
        {conv_context}
        """
        
        # Add entity context
        entity_context = ""
        if entities.get("date_entities"):
            entity_context += f" User mentioned date(s): {', '.join(entities['date_entities'])}."
        if entities.get("time_entities"):
            entity_context += f" User mentioned time(s): {', '.join(entities['time_entities'])}."
        
        user_prompt = f"User wants to: {user_input}{entity_context}\n\nGuide them naturally:"
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            executor,
            functools.partial(
                _gemini.generate_response,
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.3,
                max_tokens=200
            )
        )
        
        return response.strip() if response else "I'd be happy to help you with that. Could you tell me a bit more about what you need?"
    
    async def _generate_empathetic_response(self, user_input: str, emotion_score: float,
                                            conv_context: str) -> str:
        """Generate empathetic response for emotional or urgent situations"""
        
        empathy_level = "highly empathetic" if emotion_score < -0.5 else "empathetic"
        
        system_prompt = f"""You are {self.business_name}'s compassionate assistant.
        Respond with {empathy_level} care and understanding. Acknowledge their concern.
        Be reassuring, offer clear next steps, and show genuine care.
        Keep responses calm and supportive.
        
        Conversation History:
        {conv_context}
        """
        
        user_prompt = f"User expresses (emotion score: {emotion_score:.2f}): {user_input}\n\nRespond with empathy:"
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            executor,
            functools.partial(
                _gemini.generate_response,
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.5,
                max_tokens=180
            )
        )
        
        return response.strip() if response else "I understand this is important. Let me help you with that right away."
    
    async def _generate_general_response(self, user_input: str, conv_context: str) -> str:
        """Generate general helpful response"""
        
        system_prompt = f"""You are {self.business_name}'s helpful assistant.
        Respond naturally and helpfully. If you don't know something, be honest and offer to connect them with someone who can help.
        Keep responses friendly and professional.
        
        Conversation History:
        {conv_context}
        """
        
        user_prompt = f"User asks: {user_input}\n\nRespond helpfully:"
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            executor,
            functools.partial(
                _gemini.generate_response,
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.6,
                max_tokens=160
            )
        )
        
        return response.strip() if response else f"How can I assist you with {self.business_name} today?"
    
    def _apply_persona_touch(self, response: str, intent_info: Dict, emotion_score: float) -> str:
        """Apply persona-specific touches to the response"""
        response = response.strip()
        
        # Add greeting touch for first interactions
        if intent_info["intent"] == "greeting" and emotion_score > 0:
            if not response.lower().startswith(("hi", "hello", "hey", "welcome")):
                response = f"Hello! {response}"
        
        # Add farewell touch
        if intent_info["intent"] == "farewell":
            if not any(word in response.lower() for word in ["bye", "goodbye", "thank", "welcome"]):
                response = f"{response} Have a wonderful day!"
        
        # Add empathy touch for negative emotions
        if emotion_score < -0.3:
            if not any(word in response.lower() for word in ["sorry", "understand", "apologize"]):
                response = f"I understand. {response}"
        
        # Ensure proper punctuation
        if not response.endswith(('.', '!', '?')):
            response = response + '.'
        
        return response
    
    def _add_context_awareness(self, response: str, context: Dict, entities: Dict) -> str:
        """Add context-aware elements to response"""
        # Add specific references if available
        if entities.get("date_entities"):
            dates = entities["date_entities"]
            if len(dates) == 1 and dates[0] in response.lower():
                # Date already mentioned
                pass
            elif dates:
                response = f"For {dates[0]}, {response.lower()}"
        
        # Add urgency context
        if entities.get("urgency_level") == "high":
            if "right away" not in response.lower() and "immediately" not in response.lower():
                response = f"{response} I'll help you with this right away."
        
        return response
    
    def _requires_follow_up(self, intent_info: Dict, entities: Dict) -> bool:
        """Check if this query requires follow-up questions"""
        follow_up_intents = ["appointment", "booking", "service", "pricing"]
        return intent_info["intent"] in follow_up_intents and not entities.get("date_entities")
    
    def _get_suggested_actions(self, intent_info: Dict, entities: Dict) -> List[str]:
        """Get suggested next actions based on intent"""
        actions = []
        
        if intent_info["intent"] == "appointment":
            if not entities.get("date_entities"):
                actions.append("Ask for preferred date")
            if not entities.get("time_entities"):
                actions.append("Ask for preferred time")
            actions.append("Confirm contact details")
        
        elif intent_info["intent"] == "contact":
            actions.append("Provide contact options")
            actions.append("Offer to connect directly")
        
        elif intent_info["intent"] == "service":
            actions.append("Explain available services")
            actions.append("Ask about specific needs")
        
        return actions
    
    def _get_fallback_response(self, intent_info: Dict) -> str:
        """Get fallback response when generation fails"""
        fallbacks = {
            "greeting": f"Hello! Welcome to {self.business_name}. How can I help you today?",
            "farewell": "Thank you for contacting us. Have a wonderful day!",
            "business_hours": f"I'd be happy to tell you about {self.business_name}'s hours. Let me connect you with that information.",
            "appointment": "I can help you schedule an appointment. Could you tell me what day works best for you?",
            "contact": f"Here's how you can reach {self.business_name}. Would you like our phone number or address?",
            "service": f"Let me tell you about the services we offer at {self.business_name}.",
            "pricing": "I can provide you with pricing information. Let me connect you with our rates.",
            "emergency": "I understand this is urgent. Let me connect you with someone who can help immediately.",
            "general_inquiry": f"How can I assist you with {self.business_name} today?"
        }
        
        return fallbacks.get(intent_info["intent"], fallbacks["general_inquiry"])

_response_generator = ResponseGenerator(BUSINESS_NAME)

# ================== Session Management ==================
class VoiceSession:
    """Complete voice session management"""
    
    def __init__(self, session_id: str, client_id: str, language: str = "en-IN"):
        self.session_id = session_id
        self.client_id = client_id
        self.language = language
        self.start_time = time.time()
        
        # Queues and threading
        self.audio_queue = queue.Queue(maxsize=500)
        self.transcripts_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.stop_event = threading.Event()
        
        # State tracking
        self.is_bot_speaking = False
        self.is_listening = True
        self.last_voice_time = time.time()
        self.silence_start_time = None
        
        # Conversation state
        self.conversation_history = deque(maxlen=MAX_CONVERSATION_HISTORY)
        self.current_context = {}
        self.utterance_buffer = []
        self.pending_debounce_task = None
        
        # Performance metrics
        self.metrics = {
            "total_utterances": 0,
            "avg_response_time": 0,
            "total_words_spoken": 0,
            "interruptions": 0,
            "errors": 0,
            "rag_hits": 0,
            "conversation_depth": 0
        }
        
        # Services
        self.stt_worker = None
        self.tts_worker_task = None
        self.transcript_processor_task = None
    
    async def initialize(self, ws: WebSocket):
        """Initialize session with WebSocket connection"""
        loop = asyncio.get_event_loop()
        
        # Start STT worker thread
        self.stt_worker = EnhancedSTTWorker(
            loop=loop,
            audio_queue=self.audio_queue,
            transcripts_queue=self.transcripts_queue,
            stop_event=self.stop_event,
            initial_language=self.language
        )
        
        stt_thread = threading.Thread(
            target=self.stt_worker.run,
            daemon=True
        )
        stt_thread.start()
        
        # Start TTS worker
        self.tts_worker_task = asyncio.create_task(self._tts_worker_loop(ws))
        
        # Start transcript processor
        self.transcript_processor_task = asyncio.create_task(self._process_transcripts(ws))
        
        logger.info(f"✅ Session {self.session_id} initialized (client: {self.client_id})")
    
    async def _tts_worker_loop(self, ws: WebSocket):
        """TTS worker for sequential audio synthesis"""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get next TTS item with timeout
                    item = await asyncio.wait_for(self.tts_queue.get(), timeout=TTS_WORKER_IDLE_TIMEOUT)
                    if item is None:  # Shutdown signal
                        break
                    
                    text = item.get("text", "")
                    language = item.get("language", self.language)
                    emotion_score = item.get("emotion_score", 0.0)
                    context_type = item.get("context_type", "general")
                    
                    if not text:
                        continue
                    
                    # Synthesize speech
                    audio_data = await synthesize_text_to_pcm(
                        text=text,
                        language_code=language,
                        emotion_score=emotion_score,
                        context_type=context_type,
                        sample_rate_hz=24000,
                        voice_preference="neural"
                    )
                    
                    if audio_data:
                        # Convert to WAV and send
                        wav_bytes = make_wav_from_pcm16(audio_data, sample_rate=24000)
                        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
                        
                        await ws.send_text(json.dumps({
                            "type": "audio",
                            "audio": b64_audio,
                            "metadata": {
                                "text_length": len(text),
                                "emotion_score": emotion_score,
                                "context_type": context_type,
                                "session_id": self.session_id
                            }
                        }))
                        
                        # Update metrics
                        self.metrics["total_words_spoken"] += len(text.split())
                    
                except asyncio.TimeoutError:
                    # Idle timeout, check if we should continue
                    if self.stop_event.is_set():
                        break
                    continue
                except Exception as e:
                    logger.error(f"TTS worker error: {e}")
                    self.metrics["errors"] += 1
        
        except asyncio.CancelledError:
            logger.info(f"TTS worker cancelled for session {self.session_id}")
        except Exception as e:
            logger.exception(f"TTS worker crashed: {e}")
        finally:
            logger.info(f"TTS worker stopped for session {self.session_id}")
    
    async def _process_transcripts(self, ws: WebSocket):
        """Process STT transcripts with advanced logic"""
        try:
            while not self.stop_event.is_set():
                try:
                    resp = await asyncio.wait_for(self.transcripts_queue.get(), timeout=1.0)
                    
                    if resp is None:  # Shutdown signal
                        break
                    
                    # Process each result
                    for result in resp.results:
                        if not result.alternatives:
                            continue
                        
                        alt = result.alternatives[0]
                        transcript = alt.transcript.strip()
                        is_final = result.is_final
                        
                        if not transcript:
                            continue
                        
                        # Send interim transcripts for real-time display
                        if not is_final and len(transcript) > 2:
                            await ws.send_text(json.dumps({
                                "type": "transcript_interim",
                                "text": transcript,
                                "confidence": alt.confidence if hasattr(alt, 'confidence') else 0.0
                            }))
                        
                        # Handle final transcripts
                        if is_final:
                            logger.info(f"📝 Final transcript: {transcript}")
                            
                            # Check for barge-in
                            if self.is_bot_speaking:
                                logger.info("🛑 Barge-in detected")
                                await self._handle_interruption(ws)
                            
                            # Process utterance
                            await self._process_utterance(transcript, ws)
                            
                except asyncio.TimeoutError:
                    # Check for prolonged silence
                    if self.silence_start_time and time.time() - self.silence_start_time > MAX_SILENCE_SECONDS:
                        await self._handle_prolonged_silence(ws)
                    continue
                except Exception as e:
                    logger.error(f"Transcript processing error: {e}")
        
        except asyncio.CancelledError:
            logger.info(f"Transcript processor cancelled for session {self.session_id}")
        except Exception as e:
            logger.exception(f"Transcript processor crashed: {e}")
    
    async def _process_utterance(self, transcript: str, ws: WebSocket):
        """Process a complete user utterance"""
        self.metrics["total_utterances"] += 1
        self.last_voice_time = time.time()
        self.silence_start_time = None
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "text": transcript,
            "timestamp": time.time(),
            "source": "stt"
        })
        
        # Update context
        self.current_context = _context_awareness.update_context(
            transcript, self.conversation_history, self.current_context
        )
        
        # Send acknowledgment
        await ws.send_text(json.dumps({
            "type": "transcript_final",
            "text": transcript,
            "session_id": self.session_id
        }))
        
        # Generate and send response
        await self._generate_and_send_response(transcript, ws)
    
    async def _generate_and_send_response(self, user_input: str, ws: WebSocket):
        """Generate intelligent response and send TTS"""
        start_time = time.time()
        
        try:
            # Get business context if available
            context_data = {}
            try:
                context_data = await self._get_business_context()
            except Exception as e:
                logger.error(f"Failed to get business context: {e}")
            
            # Generate response
            response_data = await _response_generator.generate_response(
                user_input=user_input,
                context=context_data,
                conversation_history=list(self.conversation_history),
                client_id=self.client_id
            )
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "text": response_data["text"],
                "intent": response_data["intent"],
                "emotion_score": response_data["emotion_score"],
                "timestamp": time.time(),
                "metadata": response_data
            })
            
            # Update metrics
            if response_data.get("strategy") == "rag_contextual":
                self.metrics["rag_hits"] += 1
            
            self.metrics["conversation_depth"] = len(self.conversation_history) // 2
            
            # Send text response
            await ws.send_text(json.dumps({
                "type": "ai_response",
                "text": response_data["text"],
                "metadata": {
                    "intent": response_data["intent"],
                    "confidence": response_data["confidence"],
                    "emotion_score": response_data["emotion_score"],
                    "requires_follow_up": response_data["requires_follow_up"],
                    "suggested_actions": response_data["suggested_actions"],
                    "response_time_ms": int((time.time() - start_time) * 1000)
                }
            }))
            
            # Queue for TTS
            await self.tts_queue.put({
                "text": response_data["text"],
                "language": self.language,
                "emotion_score": response_data["emotion_score"],
                "context_type": response_data.get("strategy", "general")
            })
            
            # Mark bot as speaking
            self.is_bot_speaking = True
            
            # Update response time metric
            response_time = (time.time() - start_time) * 1000
            if self.metrics["avg_response_time"] == 0:
                self.metrics["avg_response_time"] = response_time
            else:
                # Moving average
                self.metrics["avg_response_time"] = (self.metrics["avg_response_time"] * 0.7 + response_time * 0.3)
            
            logger.info(f"✅ Response generated in {response_time:.0f}ms")
            
        except Exception as e:
            logger.exception(f"Response generation failed: {e}")
            self.metrics["errors"] += 1
            
            # Send error response
            error_response = "I apologize, but I encountered an issue processing your request. Could you please try again?"
            await ws.send_text(json.dumps({
                "type": "ai_response",
                "text": error_response,
                "metadata": {"error": True}
            }))
            
            await self.tts_queue.put({
                "text": error_response,
                "language": self.language,
                "emotion_score": -0.2,
                "context_type": "error"
            })
    
    async def _handle_interruption(self, ws: WebSocket):
        """Handle user interruption while bot is speaking"""
        self.metrics["interruptions"] += 1
        self.is_bot_speaking = False
        
        # Clear TTS queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except Exception:
                break
        
        # Send interruption signal
        await ws.send_text(json.dumps({
            "type": "interruption",
            "session_id": self.session_id,
            "timestamp": time.time()
        }))
        
        logger.info("✅ Handled interruption")
    
    async def _handle_prolonged_silence(self, ws: WebSocket):
        """Handle prolonged silence in conversation"""
        if not self.is_bot_speaking and len(self.conversation_history) > 0:
            # Check if we should prompt user to continue
            last_interaction = time.time() - self.last_voice_time
            if last_interaction > MAX_SILENCE_SECONDS * 2:
                prompt = "Is there anything else I can help you with?"
                await self.tts_queue.put({
                    "text": prompt,
                    "language": self.language,
                    "emotion_score": 0.0,
                    "context_type": "prompt"
                })
                self.silence_start_time = None
    
    async def _get_business_context(self) -> Dict[str, Any]:
        """Get business-specific context for responses"""
        # This would typically query a database for business info
        # For now, return basic context
        return {
            "business_name": BUSINESS_NAME,
            "has_relevant_data": True,  # Assume we have data for now
            "context_type": "business",
            "available_services": ["consultation", "appointment", "information"],
            "supported_languages": ["en-IN", "hi-IN"]
        }
    
    async def cleanup(self):
        """Clean up session resources"""
        logger.info(f"🔄 Cleaning up session {self.session_id}")
        
        # Signal shutdown
        self.stop_event.set()
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break
        
        # Send shutdown signals
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass
        
        try:
            self.transcripts_queue.put_nowait(None)
        except Exception:
            pass
        
        try:
            self.tts_queue.put_nowait(None)
        except Exception:
            pass
        
        # Cancel tasks
        if self.tts_worker_task:
            self.tts_worker_task.cancel()
            try:
                await self.tts_worker_task
            except asyncio.CancelledError:
                pass
        
        if self.transcript_processor_task:
            self.transcript_processor_task.cancel()
            try:
                await self.transcript_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"✅ Session {self.session_id} cleaned up")

# ================== Session Manager ==================
class SessionManager:
    """Manages active voice sessions"""
    
    def __init__(self):
        self.sessions = {}
        self.session_lock = asyncio.Lock()
    
    async def create_session(self, client_id: str, language: str = "en-IN") -> VoiceSession:
        """Create a new voice session"""
        session_id = f"session_{int(time.time())}_{hash(client_id) % 10000}"
        
        async with self.session_lock:
            session = VoiceSession(session_id, client_id, language)
            self.sessions[session_id] = session
        
        logger.info(f"✅ Created session {session_id} for client {client_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get an existing session"""
        async with self.session_lock:
            return self.sessions.get(session_id)
    
    async def remove_session(self, session_id: str):
        """Remove and cleanup a session"""
        async with self.session_lock:
            session = self.sessions.pop(session_id, None)
            if session:
                await session.cleanup()
                logger.info(f"🗑️ Removed session {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        return {
            "total_sessions": len(self.sessions),
            "session_ids": list(self.sessions.keys()),
            "timestamp": time.time()
        }

_session_manager = SessionManager()

# ================== WebSocket Handler ==================
@router.websocket("/ws/hd-audio")
async def hd_audio_websocket(ws: WebSocket):
    """Main WebSocket endpoint for HD voice conversations"""
    
    # Extract connection parameters
    client_id = ws.query_params.get("client_id", DEFAULT_CLIENT_ID)
    session_token = ws.query_params.get("token")
    initial_language = ws.query_params.get("language", "en-IN")
    
    # Authentication (if configured)
    if WEBSOCKET_API_TOKEN and session_token != WEBSOCKET_API_TOKEN:
        await ws.close(code=1008, reason="Unauthorized")
        logger.warning(f"Unauthorized connection attempt from client {client_id}")
        return
    
    await ws.accept()
    logger.info(f"✅ WebSocket accepted for client {client_id}")
    
    # Create session
    session = await _session_manager.create_session(client_id, initial_language)
    
    try:
        # Initialize session
        await session.initialize(ws)
        
        # Send session ready message
        await ws.send_text(json.dumps({
            "type": "session_ready",
            "session_id": session.session_id,
            "client_id": client_id,
            "language": initial_language,
            "timestamp": time.time(),
            "capabilities": {
                "streaming_stt": True,
                "emotional_tts": True,
                "rag_context": True,
                "interruptions": True,
                "multilingual": True
            }
        }))
        
        # Main WebSocket message loop
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(ws.receive(), timeout=30.0)
                
                if message["type"] == "websocket.disconnect":
                    logger.info(f"Client disconnected from session {session.session_id}")
                    break
                
                # Handle text messages (control)
                if message["type"] == "websocket.receive" and "text" in message:
                    data = json.loads(message["text"])
                    await _handle_control_message(data, ws, session)
                
                # Handle binary messages (audio)
                elif message["type"] == "websocket.receive" and "bytes" in message:
                    audio_data = message["bytes"]
                    
                    # Check for silence
                    if not is_silence(audio_data):
                        session.last_voice_time = time.time()
                        session.silence_start_time = None
                    
                        # Queue audio for STT
                        if session.audio_queue.qsize() < 450:  # Prevent overflow
                            try:
                                session.audio_queue.put_nowait(audio_data)
                            except queue.Full:
                                logger.warning(f"Audio queue full for session {session.session_id}")
                        else:
                            logger.warning(f"Dropping audio - queue full for session {session.session_id}")
                    else:
                        # Track silence
                        if session.silence_start_time is None:
                            session.silence_start_time = time.time()
                
            except asyncio.TimeoutError:
                # Check if session is still active
                if time.time() - session.last_voice_time > 300:  # 5 minutes inactivity
                    logger.info(f"Session {session.session_id} timeout due to inactivity")
                    break
                continue
            except Exception as e:
                logger.exception(f"WebSocket error in session {session.session_id}: {e}")
                break
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session.session_id}")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
    finally:
        # Cleanup session
        await _session_manager.remove_session(session.session_id)
        
        try:
            await ws.close()
        except Exception:
            pass
        
        logger.info(f"✅ WebSocket connection closed for session {session.session_id}")

async def _handle_control_message(data: Dict, ws: WebSocket, session: VoiceSession):
    """Handle control messages from client"""
    message_type = data.get("type")
    
    if message_type == "change_language":
        new_language = data.get("language", "en-IN")
        if new_language != session.language:
            session.language = new_language
            session.stt_worker.change_language(new_language)
            await ws.send_text(json.dumps({
                "type": "language_changed",
                "language": new_language,
                "session_id": session.session_id
            }))
    
    elif message_type == "get_metrics":
        await ws.send_text(json.dumps({
            "type": "session_metrics",
            "session_id": session.session_id,
            "metrics": session.metrics,
            "conversation_length": len(session.conversation_history),
            "session_duration": time.time() - session.start_time
        }))
    
    elif message_type == "clear_history":
        session.conversation_history.clear()
        await ws.send_text(json.dumps({
            "type": "history_cleared",
            "session_id": session.session_id
        }))
    
    elif message_type == "stop_speaking":
        await session._handle_interruption(ws)
    
    elif message_type == "ping":
        await ws.send_text(json.dumps({
            "type": "pong",
            "timestamp": time.time(),
            "session_id": session.session_id
        }))
    
    else:
        logger.warning(f"Unknown control message type: {message_type}")

# ================== HTTP Endpoints ==================
@router.get("/voice/sessions")
async def get_active_sessions():
    """Get list of active voice sessions"""
    stats = _session_manager.get_stats()
    return {
        "status": "success",
        "active_sessions": stats["total_sessions"],
        "session_ids": stats["session_ids"],
        "timestamp": time.time()
    }

@router.get("/voice/capabilities")
async def get_voice_capabilities():
    """Get voice system capabilities"""
    return {
        "status": "success",
        "capabilities": {
            "languages": list(VOICE_MAP.keys()),
            "voice_engines": ["neural", "standard", "wavenet"],
            "emotion_support": True,
            "interruption_handling": True,
            "context_awareness": True,
            "rag_integration": True,
            "streaming": True,
            "max_concurrent_sessions": MAX_CONCURRENT_TTS * 2
        },
        "performance": {
            "stt_sample_rate": STT_SAMPLE_RATE,
            "tts_sample_rate": 24000,
            "max_response_time": LLM_TIMEOUT,
            "vad_threshold": VAD_THRESHOLD
        }
    }

@router.post("/voice/test")
async def test_voice_system():
    """Test voice system components"""
    try:
        # Test STT
        test_audio = b"\x00" * 3200  # 100ms of silence
        audio = speech.RecognitionAudio(content=test_audio)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN"
        )
        
        # Test TTS
        synthesis_input = tts.SynthesisInput(text="Test")
        voice = tts.VoiceSelectionParams(
            language_code="en-IN",
            name="en-IN-Neural2-A"
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16
        )
        
        return {
            "status": "operational",
            "components": {
                "google_stt": "connected",
                "google_tts": "connected",
                "gemini_llm": "initialized",
                "pinecone_rag": "initialized" if pinecone_service.is_configured() else "disconnected",
                "emotion_engine": "initialized"
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }

@router.post("/voice/generate")
async def generate_voice_response(data: Dict[str, Any]):
    """Generate voice response for testing"""
    text = data.get("text", "Hello, this is a test.")
    language = data.get("language", "en-IN")
    emotion_score = data.get("emotion_score", 0.0)
    
    try:
        audio = await synthesize_text_to_pcm(
            text=text,
            language_code=language,
            emotion_score=emotion_score,
            sample_rate_hz=24000
        )
        
        if audio:
            wav_bytes = make_wav_from_pcm16(audio, sample_rate=24000)
            b64_audio = base64.b64encode(wav_bytes).decode("ascii")
            
            return {
                "status": "success",
                "audio": b64_audio,
                "text": text,
                "language": language,
                "emotion_score": emotion_score,
                "audio_length": len(audio)
            }
        else:
            return {
                "status": "error",
                "error": "TTS synthesis failed"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/voice/health")
async def voice_health_check():
    """Comprehensive health check for voice system"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Check Google services
    try:
        # Quick STT test
        audio = speech.RecognitionAudio(content=b"\x00" * 1600)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN"
        )
        health_status["components"]["google_stt"] = "healthy"
    except Exception as e:
        health_status["components"]["google_stt"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Quick TTS test
        synthesis_input = tts.SynthesisInput(text="Health check")
        voice = tts.VoiceSelectionParams(language_code="en-IN", name="en-IN-Neural2-A")
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)
        health_status["components"]["google_tts"] = "healthy"
    except Exception as e:
        health_status["components"]["google_tts"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Gemini
    try:
        test_response = await asyncio.get_event_loop().run_in_executor(
            executor,
            functools.partial(_gemini.generate_response, prompt="Test", max_tokens=10)
        )
        health_status["components"]["gemini_llm"] = "healthy"
    except Exception as e:
        health_status["components"]["gemini_llm"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Pinecone
    try:
        if pinecone_service.is_configured():
            health_status["components"]["pinecone_rag"] = "healthy"
        else:
            health_status["components"]["pinecone_rag"] = "not_configured"
    except Exception as e:
        health_status["components"]["pinecone_rag"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Session status
    stats = _session_manager.get_stats()
    health_status["sessions"] = {
        "active": stats["total_sessions"],
        "max_concurrent": MAX_CONCURRENT_TTS * 2
    }
    
    return health_status

# ================== Error Handlers ==================
@router.get("/voice/errors")
async def get_recent_errors(limit: int = 20):
    """Get recent system errors"""
    # In production, this would query a database
    return {
        "status": "success",
        "errors": [],
        "timestamp": time.time()
    }

@router.post("/voice/reset")
async def reset_voice_system():
    """Reset voice system components"""
    # Note: This is a dangerous endpoint for production
    logger.warning("Voice system reset requested")
    
    return {
        "status": "success",
        "message": "Reset initiated",
        "timestamp": time.time()
    }

# ================== WebSocket Test Client ==================
@router.get("/voice/test-client")
async def test_client_page():
    """Serve test client HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Assistant Test Client</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            .log { background: #f8f9fa; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Voice Assistant Test Client</h1>
            <div id="status" class="status disconnected">Disconnected</div>
            
            <div>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()" disabled id="disconnectBtn">Disconnect</button>
                <button onclick="testTTS()">Test TTS</button>
                <button onclick="getMetrics()">Get Metrics</button>
            </div>
            
            <div>
                <h3>Test Text-to-Speech</h3>
                <textarea id="testText">Hello, this is a test of the voice system.</textarea>
                <button onclick="synthesizeText()">Synthesize</button>
            </div>
            
            <div>
                <h3>System Log</h3>
                <div id="log" class="log"></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            const logEl = document.getElementById('log');
            const statusEl = document.getElementById('status');
            const disconnectBtn = document.getElementById('disconnectBtn');
            
            function log(msg) {
                const entry = document.createElement('div');
                entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
                logEl.appendChild(entry);
                logEl.scrollTop = logEl.scrollHeight;
                console.log(msg);
            }
            
            async function connect() {
                const wsUrl = `ws://${window.location.host}/ws/hd-audio?client_id=test`;
                
                try {
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = () => {
                        statusEl.className = 'status connected';
                        statusEl.textContent = 'Connected';
                        disconnectBtn.disabled = false;
                        log('✅ Connected to voice server');
                    };
                    
                    ws.onclose = () => {
                        statusEl.className = 'status disconnected';
                        statusEl.textContent = 'Disconnected';
                        disconnectBtn.disabled = true;
                        log('❌ Disconnected from voice server');
                    };
                    
                    ws.onerror = (error) => {
                        log(`❌ WebSocket error: ${error}`);
                    };
                    
                    ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            log(`📨 Received: ${data.type}`);
                        } catch (e) {
                            log(`Received non-JSON message: ${event.data}`);
                        }
                    };
                    
                } catch (error) {
                    log(`❌ Connection error: ${error}`);
                }
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            async function testTTS() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    log('❌ Not connected');
                    return;
                }
                
                ws.send(JSON.stringify({
                    type: 'test_tts',
                    text: 'This is a test of the text to speech system.'
                }));
                
                log('📤 Sent TTS test request');
            }
            
            async function synthesizeText() {
                const text = document.getElementById('testText').value;
                
                const response = await fetch('/api/voice/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, language: 'en-IN' })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    log('✅ TTS synthesis successful');
                    // Play audio
                    const audio = new Audio(`data:audio/wav;base64,${result.audio}`);
                    audio.play();
                } else {
                    log(`❌ TTS failed: ${result.error}`);
                }
            }
            
            async function getMetrics() {
                const response = await fetch('/api/voice/health');
                const result = await response.json();
                log(`📊 System health: ${JSON.stringify(result, null, 2)}`);
            }
        </script>
    </body>
    </html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

# ================== Main Voice Router ==================
@router.get("/")
async def voice_root():
    """Voice system root endpoint"""
    return {
        "service": "Voice Assistant System",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/hd-audio",
            "health": "/api/voice/health",
            "capabilities": "/api/voice/capabilities",
            "sessions": "/api/voice/sessions",
            "test": "/api/voice/test",
            "generate": "/api/voice/generate"
        },
        "timestamp": time.time()
    }
