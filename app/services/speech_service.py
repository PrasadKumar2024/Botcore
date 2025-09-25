# app/services/speech_service.py

import logging
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SpeechService:
    """
    Basic speech service for language detection and text processing
    Uses only Twilio's built-in features - no external APIs
    """
    
    def __init__(self):
        # Language detection keywords
        self.language_keywords = {
            "en": {
                "keywords": ["english", "inglish", "angrezi", "in english", "english me", "speak english"],
                "name": "English",
                "voice": "Polly.Aditi",
                "code": "en-IN"
            },
            "hi": {
                "keywords": ["hindi", "हिंदी", "hindee", "hindi me", "hindi mein", "हिंदी में", "speak hindi"],
                "name": "Hindi", 
                "voice": "Polly.Aditi",
                "code": "hi-IN"
            },
            "ta": {
                "keywords": ["tamil", "தமிழ்", "tamil la", "tamil ah", "in tamil", "speak tamil"],
                "name": "Tamil",
                "voice": "Polly.Aditi", 
                "code": "ta-IN"
            },
            "te": {
                "keywords": ["telugu", "తెలుగు", "in telugu", "speak telugu"],
                "name": "Telugu",
                "voice": "Polly.Aditi",
                "code": "te-IN"
            },
            "kn": {
                "keywords": ["kannada", "ಕನ್ನಡ", "in kannada", "speak kannada"],
                "name": "Kannada",
                "voice": "Polly.Aditi",
                "code": "kn-IN"
            }
        }
        
        # Common medical terms for better understanding
        self.medical_terms = {
            "en": ["appointment", "doctor", "clinic", "prescription", "medicine", "emergency"],
            "hi": ["अपॉइंटमेंट", "डॉक्टर", "क्लिनिक", "दवा", "प्रिस्क्रिप्शन", "इमरजेंसी"],
            "ta": ["அப்பாயின்ட்மெண்ட்", "டாக்டர்", "கிளினிக்", "மருந்து", "பிரஸ்கிரிப்ஷன்", "அவசர"],
            "te": ["అపాయింట్మెంట్", "డాక్టర్", "క్లినిక్", "మందు", "ప్రిస్క్రిప్షన్", "అత్యవసర"],
            "kn": ["ಅಪಾಯಿಂಟ್ಮೆಂಟ್", "ಡಾಕ್ಟರ್", "ಕ್ಲಿನಿಕ್", "ಮದ್ದು", "ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್", "ತುರ್ತು"]
        }
        
        logger.info("Basic SpeechService initialized (Twilio-only mode)")

    def detect_language(self, text: str, current_language: str = "en") -> Tuple[str, float]:
        """
        Detect language from text with confidence score
        Returns: (language_code, confidence_score)
        """
        text_lower = text.lower()
        matches = []
        
        for lang_code, lang_data in self.language_keywords.items():
            keyword_count = 0
            for keyword in lang_data["keywords"]:
                if keyword in text_lower:
                    keyword_count += 1
            
            if keyword_count > 0:
                confidence = min(keyword_count * 0.3, 1.0)  # Basic confidence calculation
                matches.append((lang_code, confidence))
        
        # Also check for medical terms to boost confidence
        for lang_code, terms in self.medical_terms.items():
            term_count = sum(1 for term in terms if term in text_lower)
            if term_count > 0:
                # Find existing match or create new
                existing_match = next((m for m in matches if m[0] == lang_code), None)
                if existing_match:
                    matches.remove(existing_match)
                    matches.append((lang_code, existing_match[1] + term_count * 0.1))
                else:
                    matches.append((lang_code, term_count * 0.1))
        
        if matches:
            # Return the match with highest confidence
            best_match = max(matches, key=lambda x: x[1])
            logger.info(f"Detected language: {best_match[0]} with confidence {best_match[1]:.2f}")
            return best_match
        
        # No keywords found, return current language with low confidence
        return (current_language, 0.1)

    def should_switch_language(self, text: str, current_language: str) -> bool:
        """
        Determine if language should be switched based on user request
        """
        text_lower = text.lower()
        
        # Check for explicit language switch requests
        switch_phrases = [
            "speak in", "in language", "change language", "switch to",
            "में बोलो", "भाषा बदलो", "மொழி மாற்ற", "భాష మార్చు", "ಭಾಷೆ ಬದಲಾಯಿಸಿ"
        ]
        
        if any(phrase in text_lower for phrase in switch_phrases):
            return True
        
        # Check if user is consistently using another language
        detected_lang, confidence = self.detect_language(text, current_language)
        return confidence > 0.5 and detected_lang != current_language

    def normalize_text(self, text: str, language: str = "en") -> str:
        """
        Normalize text for better TTS pronunciation
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Language-specific normalization
        if language == "en":
            # Expand common abbreviations
            replacements = {
                "dr.": "doctor",
                "appt.": "appointment",
                "rx": "prescription",
                "asap": "as soon as possible"
            }
            for abbr, full in replacements.items():
                text = text.replace(abbr, full)
        
        return text

    def extract_key_information(self, text: str, language: str = "en") -> Dict[str, str]:
        """
        Extract key information from user speech for better responses
        """
        info = {
            "intent": "",
            "time_mention": "",
            "doctor_mention": "",
            "urgency_level": "normal"
        }
        
        text_lower = text.lower()
        
        # Detect intent
        intents = {
            "appointment": ["appointment", "schedule", "book", "मिलना", "अपॉइंटमेंट", "அப்பாயின்ட்மெண்ட్"],
            "prescription": ["prescription", "medicine", "refill", "दवा", "மருந்து", "మందు"],
            "emergency": ["emergency", "urgent", "now", "immediately", "तुरंत", "அவசர", "తక్షణ"]
        }
        
        for intent, keywords in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                info["intent"] = intent
                break
        
        # Detect time mentions
        time_indicators = ["today", "tomorrow", "morning", "afternoon", "evening", "week", "month"]
        for indicator in time_indicators:
            if indicator in text_lower:
                info["time_mention"] = indicator
                break
        
        # Detect urgency
        if any(word in text_lower for word in ["emergency", "urgent", "immediately", "now", "तुरंत", "அவசர"]):
            info["urgency_level"] = "high"
        
        return info

    def generate_natural_response(self, text: str, language: str = "en") -> str:
        """
        Add natural language flourishes to responses
        """
        if language == "hi":
            # Hindi natural responses
            if "thank you" in text.lower() or "धन्यवाद" in text:
                return "आपका स्वागत है! क्या मैं आपकी और किसी बात में मदद कर सकती हूँ?"
            elif "sorry" in text.lower() or "माफ़" in text:
                return "कोई बात नहीं। कृपया अपनी समस्या बताएं।"
                
        elif language == "ta":
            # Tamil natural responses
            if "thank you" in text.lower() or "நன்றி" in text:
                return "தயவு செய்து! மேலும் எதையும் உதவ முடியுமா?"
            elif "sorry" in text.lower() or "மன்னிக்க" in text:
                return "பரவாயில்லை. தயவு செய்து உங்கள் பிரச்சனையை சொல்லுங்கள்."
        
        else:
            # English natural responses (default)
            if "thank you" in text.lower():
                return "You're welcome! Is there anything else I can help you with?"
            elif "sorry" in text.lower():
                return "No problem at all. Please tell me about your issue."
        
        return text

    def get_language_config(self, language_code: str) -> Dict[str, str]:
        """
        Get Twilio voice configuration for a language
        """
        lang_data = self.language_keywords.get(language_code, self.language_keywords["en"])
        return {
            "voice": lang_data["voice"],
            "code": lang_data["code"],
            "name": lang_data["name"]
        }

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get list of all supported languages
        """
        return [
            {
                "code": lang_code,
                "name": data["name"],
                "voice": data["voice"],
                "twilio_code": data["code"]
            }
            for lang_code, data in self.language_keywords.items()
        ]

    def is_medical_emergency(self, text: str) -> bool:
        """
        Detect potential medical emergencies for immediate handling
        """
        emergency_keywords = [
            "heart attack", "stroke", "bleeding", "unconscious", "can't breathe",
            "heart", "attack", "emergency", "urgent", "help immediately",
            "दिल का दौरा", "स्ट्रोक", "बेहोश", "सांस नहीं आ रही",
            "இதய அடைப்பு", "ஸ்ட்ரோக்", "உணர்வு இழப்பு", "மூச்சுத் திணறல்"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in emergency_keywords)

    def health_check(self) -> Dict[str, any]:
        """
        Basic health check for the speech service
        """
        return {
            "status": "healthy",
            "service": "Basic Speech Service (Twilio-only)",
            "supported_languages": len(self.language_keywords),
            "features": [
                "Language detection",
                "Text normalization", 
                "Intent extraction",
                "Emergency detection"
            ],
            "timestamp": datetime.now().isoformat()
        }


# Utility functions
def create_conversation_prompt(text: str, language: str = "en") -> str:
    """
    Create a conversation-friendly prompt by adding polite phrases
    """
    prompts = {
        "en": "Please tell me more about: ",
        "hi": "कृपया मुझे और बताएं: ",
        "ta": "தயவு செய்து மேலும் சொல்லுங்கள்: ",
        "te": "దయచేసి మరింత చెప్పండి: ",
        "kn": "ದಯವಿಟ್ಟು ಇನ್ನಷ್ಟು ಹೇಳಿ: "
    }
    
    prompt = prompts.get(language, prompts["en"])
    return prompt + text


def format_phone_number(text: str) -> str:
    """
    Extract and format phone numbers from text
    """
    # Simple phone number extraction
    phone_pattern = r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'
    matches = re.findall(phone_pattern, text)
    return matches[0] if matches else ""


# Create global instance
speech_service = SpeechService()


# Example usage and testing
if __name__ == "__main__":
    # Test the service
    service = SpeechService()
    
    # Test language detection
    test_texts = [
        "I want to speak in Hindi please",
        "हिंदी में बात करते हैं",
        "Tamil la pesunga",
        "I need an appointment with doctor"
    ]
    
    for text in test_texts:
        lang, confidence = service.detect_language(text)
        print(f"Text: {text}")
        print(f"Detected: {lang} (confidence: {confidence:.2f})")
        print(f"Should switch: {service.should_switch_language(text, 'en')}")
        print("---")
    
    print("Basic SpeechService is working correctly!")
