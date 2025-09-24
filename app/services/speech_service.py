# app/services/speech_service.py

from google.cloud import texttospeech
from google.api_core.exceptions import GoogleAPICallError, RetryError
import os
import logging
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SpeechService:
    def __init__(self):
        """Initialize Google Cloud Text-to-Speech client with enhanced configuration"""
        try:
            # Initialize the TTS client
            self.client = texttospeech.TextToSpeechClient()
            
            # Premium Wavenet voices for natural sound
            self.wavenet_voices = {
                # Indian English - Most natural for clinic conversations
                "en-IN": {
                    "female": "en-IN-Wavenet-A",  # Polite, clear (Recommended)
                    "male": "en-IN-Wavenet-B",    # Professional, authoritative
                    "neutral": "en-IN-Wavenet-C"  # Warm, friendly
                },
                # Hindi voices
                "hi-IN": {
                    "female": "hi-IN-Wavenet-A",
                    "male": "hi-IN-Wavenet-B",
                    "neutral": "hi-IN-Wavenet-C"
                },
                # South Indian languages
                "ta-IN": {"female": "ta-IN-Wavenet-A", "neutral": "ta-IN-Wavenet-A"},
                "te-IN": {"female": "te-IN-Wavenet-A", "neutral": "te-IN-Wavenet-A"},
                "kn-IN": {"female": "kn-IN-Wavenet-A", "neutral": "kn-IN-Wavenet-A"},
                "ml-IN": {"female": "ml-IN-Wavenet-A", "neutral": "ml-IN-Wavenet-A"},
                # International variants
                "en-US": {
                    "female": "en-US-Wavenet-F",  # Very natural US female
                    "male": "en-US-Wavenet-A",    # Professional US male
                    "neutral": "en-US-Wavenet-C"
                },
                "en-GB": {
                    "female": "en-GB-Wavenet-A",
                    "male": "en-GB-Wavenet-B",    # British professional
                    "neutral": "en-GB-Wavenet-C"
                }
            }
            
            # Default audio configurations for different use cases
            self.audio_configs = {
                "telephone": texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MULAW,
                    sample_rate_hertz=8000,
                    speaking_rate=1.0,
                    pitch=0.0,
                    volume_gain_db=0.0
                ),
                "standard": texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=24000,
                    speaking_rate=1.0,
                    pitch=0.0,
                    volume_gain_db=0.0
                ),
                "premium": texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    sample_rate_hertz=48000,
                    speaking_rate=1.0,
                    pitch=0.0,
                    volume_gain_db=0.0
                )
            }
            
            logger.info("SpeechService initialized successfully with Wavenet voices")
            
        except Exception as e:
            logger.error(f"Failed to initialize SpeechService: {str(e)}")
            raise

    def text_to_speech(self, text: str, language_code: str = "en-IN", 
                      gender: str = "neutral", audio_type: str = "telephone",
                      speaking_rate: float = None, pitch: float = None) -> bytes:
        """
        Convert text to speech using Google Cloud TTS with Wavenet voices
        
        Args:
            text: Input text to convert to speech
            language_code: Language code (e.g., 'en-IN', 'hi-IN')
            gender: Voice gender ('male', 'female', 'neutral')
            audio_type: Audio configuration type ('telephone', 'standard', 'premium')
            speaking_rate: Optional speaking rate (0.25 to 4.0)
            pitch: Optional pitch adjustment (-20.0 to 20.0)
        
        Returns:
            bytes: Audio content in specified format
        """
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Normalize language code
            language_code = language_code.lower()
            
            # Determine if text contains SSML
            is_ssml = text.strip().startswith('<speak>')
            
            # Select appropriate voice
            voice = self._select_voice(language_code, gender)
            
            # Get audio configuration
            audio_config = self._get_audio_config(audio_type, speaking_rate, pitch)
            
            # Set synthesis input
            if is_ssml:
                synthesis_input = texttospeech.SynthesisInput(ssml=text)
                logger.debug("Using SSML input for TTS")
            else:
                synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Perform text-to-speech synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            logger.info(f"Successfully generated speech for {len(text)} characters "
                       f"in {language_code} with {gender} voice")
            
            return response.audio_content
            
        except GoogleAPICallError as e:
            logger.error(f"Google API error in text_to_speech: {str(e)}")
            raise
        except RetryError as e:
            logger.error(f"Retry error in text_to_speech: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in text_to_speech: {str(e)}")
            raise

    def _select_voice(self, language_code: str, gender: str) -> texttospeech.VoiceSelectionParams:
        """Select the best available Wavenet voice for the given parameters"""
        try:
            # Check if we have Wavenet voices for this language
            if language_code in self.wavenet_voices:
                language_voices = self.wavenet_voices[language_code]
                
                # Try to get the requested gender, fallback to neutral, then female
                voice_name = (language_voices.get(gender) or 
                            language_voices.get("neutral") or 
                            language_voices.get("female"))
                
                if voice_name:
                    return texttospeech.VoiceSelectionParams(
                        language_code=language_code,
                        name=voice_name
                    )
            
            # Fallback: Use standard voice with SSML gender
            ssml_gender = {
                "male": texttospeech.SsmlVoiceGender.MALE,
                "female": texttospeech.SsmlVoiceGender.FEMALE,
                "neutral": texttospeech.SsmlVoiceGender.NEUTRAL
            }.get(gender, texttospeech.SsmlVoiceGender.NEUTRAL)
            
            return texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=ssml_gender
            )
            
        except Exception as e:
            logger.error(f"Error selecting voice: {str(e)}")
            # Ultimate fallback
            return texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )

    def _get_audio_config(self, audio_type: str, speaking_rate: float = None, 
                         pitch: float = None) -> texttospeech.AudioConfig:
        """Get audio configuration with optional overrides"""
        try:
            # Get base configuration
            base_config = self.audio_configs.get(audio_type, self.audio_configs["telephone"])
            
            # Create a new config with optional overrides
            config_kwargs = {
                "audio_encoding": base_config.audio_encoding,
                "sample_rate_hertz": base_config.sample_rate_hertz,
                "speaking_rate": speaking_rate if speaking_rate is not None else base_config.speaking_rate,
                "pitch": pitch if pitch is not None else base_config.pitch,
                "volume_gain_db": base_config.volume_gain_db,
            }
            
            return texttospeech.AudioConfig(**config_kwargs)
            
        except Exception as e:
            logger.error(f"Error getting audio config: {str(e)}")
            return self.audio_configs["telephone"]

    def generate_ssml_text(self, text: str, pause_ms: int = 300, 
                          emphasis: bool = False, rate: float = None) -> str:
        """
        Generate SSML text with natural pauses and emphasis for more natural speech
        
        Args:
            text: Input text
            pause_ms: Pause duration in milliseconds
            emphasis: Whether to add emphasis to important words
            rate: Speaking rate adjustment
        
        Returns:
            str: SSML formatted text
        """
        try:
            # Basic SSML wrapper
            ssml_parts = ['<speak>']
            
            # Add optional speaking rate
            if rate is not None:
                ssml_parts.append(f'<prosody rate="{rate}">')
            
            # Process text for natural speech patterns
            sentences = text.split('. ')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    # Add sentence with pause
                    ssml_parts.append(sentence.strip())
                    if i < len(sentences) - 1 and pause_ms > 0:
                        ssml_parts.append(f'<break time="{pause_ms}ms"/>')
            
            # Close prosody tag if opened
            if rate is not None:
                ssml_parts.append('</prosody>')
            
            ssml_parts.append('</speak>')
            
            return ' '.join(ssml_parts)
            
        except Exception as e:
            logger.error(f"Error generating SSML: {str(e)}")
            # Fallback to basic SSML
            return f'<speak>{text}</speak>'

    def generate_clinic_greeting(self, business_name: str, language: str = "en-IN") -> str:
        """
        Generate a natural-sounding clinic greeting with SSML enhancements
        """
        if language == "en-IN":
            return f"""<speak>
                Hello <break time="300ms"/> and thank you for calling {business_name}. 
                <break time="400ms"/>
                My name is Priya, and I'm your virtual assistant. 
                <break time="300ms"/>
                How can I help you today?
            </speak>"""
        elif language == "hi-IN":
            return f"""<speak>
                नमस्ते <break time="300ms"/> {business_name} में आपका स्वागत है।
                <break time="400ms"/>
                मेरा नाम प्रिया है, और मैं आपकी virtual assistant हूँ।
                <break time="300ms"/>
                मैं आपकी किस प्रकार सहायता कर सकती हूँ?
            </speak>"""
        else:
            return self.generate_ssml_text(f"Hello, thank you for calling {business_name}. How can I help you today?")

    def get_supported_voices(self, language_filter: str = None) -> Dict[str, Any]:
        """
        Get list of supported Wavenet voices
        """
        try:
            if language_filter:
                filtered_voices = {}
                for lang_code, voices in self.wavenet_voices.items():
                    if language_filter.lower() in lang_code:
                        filtered_voices[lang_code] = voices
                return filtered_voices
            else:
                return self.wavenet_voices.copy()
                
        except Exception as e:
            logger.error(f"Error getting supported voices: {str(e)}")
            return {}

    def validate_audio_content(self, audio_content: bytes) -> bool:
        """
        Validate that audio content is properly generated
        """
        try:
            if not audio_content:
                return False
            
            # Check minimum size (very small audio likely indicates an error)
            if len(audio_content) < 100:
                return False
            
            # Check for common audio headers (basic validation)
            if audio_content[:4] in [b'RIFF', b'\x2e\x73\x6e\x64', b'\xff\xfb']:
                return True
            
            # For μ-law, we expect specific patterns
            if len(audio_content) > 100:
                return True  # Basic length check passed
                
            return False
            
        except Exception as e:
            logger.error(f"Error validating audio content: {str(e)}")
            return False

    def get_audio_info(self, audio_content: bytes) -> Dict[str, Any]:
        """
        Get information about generated audio content
        """
        try:
            return {
                "size_bytes": len(audio_content),
                "duration_estimate_sec": len(audio_content) / 8000,  # Rough estimate for μ-law
                "is_valid": self.validate_audio_content(audio_content),
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {"error": str(e)}

    def batch_text_to_speech(self, texts: list, language_code: str = "en-IN", 
                           gender: str = "neutral") -> Dict[str, bytes]:
        """
        Convert multiple texts to speech in batch
        """
        results = {}
        try:
            for i, text in enumerate(texts):
                try:
                    audio_content = self.text_to_speech(text, language_code, gender)
                    results[f"audio_{i}"] = audio_content
                    logger.info(f"Generated batch audio {i+1}/{len(texts)}")
                except Exception as e:
                    logger.error(f"Failed to generate audio for text {i}: {str(e)}")
                    results[f"audio_{i}"] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch text-to-speech: {str(e)}")
            return {}

    def speech_to_text(self, audio_content: bytes, language_code: str = "en-IN") -> str:
        """
        Convert speech to text using Google Cloud Speech-to-Text
        This is kept for backward compatibility
        """
        try:
            from google.cloud import speech
            
            client = speech.SpeechClient()
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="phone_call",  # Use phone call model for better accuracy
                use_enhanced=True    # Use enhanced model for better results
            )
            
            audio = speech.RecognitionAudio(content=audio_content)
            
            response = client.recognize(config=config, audio=audio)
            
            if response.results:
                return response.results[0].alternatives[0].transcript
            return ""
            
        except ImportError:
            logger.error("Google Speech-to-Text client not available")
            return ""
        except Exception as e:
            logger.error(f"Error in speech-to-text: {str(e)}")
            return ""

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the speech service
        """
        try:
            # Test with a simple phrase
            test_text = "Health check successful."
            audio_content = self.text_to_speech(test_text, "en-IN", "neutral")
            
            return {
                "status": "healthy",
                "service": "Google Cloud Text-to-Speech",
                "wavenet_voices": "available",
                "audio_generation": "working",
                "supported_languages": len(self.wavenet_voices),
                "test_audio_size": len(audio_content),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "Google Cloud Text-to-Speech",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Utility functions for enhanced functionality
def create_natural_pause(text: str, pause_after_sentences: bool = True, 
                        pause_after_commas: bool = False) -> str:
    """
    Add natural pauses to text for more realistic speech
    """
    if pause_after_sentences:
        # Add pauses after sentence-ending punctuation
        text = text.replace('. ', '. <break time="300ms"/> ')
        text = text.replace('? ', '? <break time="400ms"/> ')
        text = text.replace('! ', '! <break time="350ms"/> ')
    
    if pause_after_commas:
        # Add shorter pauses after commas
        text = text.replace(', ', ', <break time="150ms"/> ')
    
    return f'<speak>{text}</speak>'


def emphasize_important_words(text: str, words_to_emphasize: list) -> str:
    """
    Add emphasis to important words in the text
    """
    for word in words_to_emphasize:
        if word in text:
            text = text.replace(word, f'<emphasis level="strong">{word}</emphasis>')
    
    return f'<speak>{text}</speak>'


# Create a global instance for easy access
speech_service = SpeechService()

# Example usage:
if __name__ == "__main__":
    # Test the service
    service = SpeechService()
    
    # Test natural clinic greeting
    greeting = service.generate_clinic_greeting("Dr. Sharma's Clinic")
    audio = service.text_to_speech(greeting, "en-IN", "female")
    
    print(f"Generated audio: {len(audio)} bytes")
    print("SpeechService is working correctly!")
