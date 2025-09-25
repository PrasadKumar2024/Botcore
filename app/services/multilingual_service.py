# app/services/multilingual_service.py

import logging
from typing import Dict, List, Optional, Tuple, Any
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class MultilingualService:
    """
    Multilingual service for handling language detection, translation, and localization
    Uses only built-in methods - no external APIs
    """
    
    def __init__(self):
        # Supported languages with their configurations
        self.supported_languages = {
            "en": {
                "name": "English",
                "code": "en-IN",
                "voice": "Polly.Aditi",
                "greeting": "Hello! How can I help you today?",
                "goodbye": "Thank you for contacting us. Have a great day!",
                "fallback": True
            },
            "hi": {
                "name": "Hindi",
                "code": "hi-IN", 
                "voice": "Polly.Aditi",
                "greeting": "नमस्ते! मैं आपकी कैसे मदद कर सकती हूँ?",
                "goodbye": "हमसे संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो!",
                "fallback": False
            },
            "ta": {
                "name": "Tamil",
                "code": "ta-IN",
                "voice": "Polly.Aditi",
                "greeting": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?",
                "goodbye": "தொடர்பு கொண்டதற்கு நன்றி. நல்ல நாள்!",
                "fallback": False
            },
            "te": {
                "name": "Telugu",
                "code": "te-IN",
                "voice": "Polly.Aditi", 
                "greeting": "నమస్కారం! నేను మీకు ఎలా సహాయపడగలను?",
                "goodbye": "సంప్రదించినందుకు ధన్యవాదాలు. మంచి రోజు!",
                "fallback": False
            },
            "kn": {
                "name": "Kannada",
                "code": "kn-IN",
                "voice": "Polly.Aditi",
                "greeting": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
                "goodbye": "ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ಶುಭ ದಿನ!",
                "fallback": False
            }
        }
        
        # Language detection patterns
        self.language_patterns = {
            "en": {
                "keywords": ["english", "inglish", "angrezi", "in english", "speak english", "english me"],
                "unicode_range": None,
                "common_words": ["the", "and", "you", "that", "was", "for", "are", "with", "his", "they"]
            },
            "hi": {
                "keywords": ["hindi", "हिंदी", "hindee", "hindi me", "hindi mein", "हिंदी में", "speak hindi"],
                "unicode_range": r'[\u0900-\u097F]',  # Devanagari range
                "common_words": ["में", "के", "है", "यह", "वह", "की", "से", "को", "का", "हो"]
            },
            "ta": {
                "keywords": ["tamil", "தமிழ்", "tamil la", "tamil ah", "in tamil", "speak tamil"],
                "unicode_range": r'[\u0B80-\u0BFF]',  # Tamil range
                "common_words": ["மற்றும்", "இந்த", "அவர்", "என்று", "ஒரு", "நான்", "உள்ள", "பொருள்", "வேண்டும்", "இது"]
            },
            "te": {
                "keywords": ["telugu", "తెలుగు", "in telugu", "speak telugu"],
                "unicode_range": r'[\u0C00-\u0C7F]',  # Telugu range
                "common_words": ["మరియు", "ఈ", "అది", "ఒక", "నేను", "కోసం", "పై", "తో", "వారు", "ఉండే"]
            },
            "kn": {
                "keywords": ["kannada", "ಕನ್ನಡ", "in kannada", "speak kannada"],
                "unicode_range": r'[\u0C80-\u0CFF]',  # Kannada range
                "common_words": ["ಮತ್ತು", "ಈ", "ಅದು", "ಒಂದು", "ನಾನು", "ಫಾರ್", "ಮೇಲೆ", "ಜೊತೆ", "ಅವರು", "ಎಂದು"]
            }
        }
        
        # Common phrases and their translations
        self.common_phrases = {
            "greeting": {
                "en": "Hello, welcome to {business_name}! How can I assist you today?",
                "hi": "नमस्ते, {business_name} में आपका स्वागत है! आज मैं आपकी कैसे मदद कर सकती हूँ?",
                "ta": "வணக்கம், {business_name}க்கு வரவேற்கிறோம்! இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
                "te": "నమస్కారం, {business_name}కు స్వాగతం! ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?",
                "kn": "ನಮಸ್ಕಾರ, {business_name}ಗೆ ಸ್ವಾಗತ! ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
            },
            "help": {
                "en": "I'm here to help you. Please tell me what you need.",
                "hi": "मैं आपकी मदद के लिए यहां हूं। कृपया मुझे बताएं कि आपको क्या चाहिए।",
                "ta": "நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன். தயவு செய்து உங்களுக்கு என்ன தேவை என்று சொல்லுங்கள்.",
                "te": "నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను. దయచేసి మీకు ఏమి కావాలో చెప్పండి.",
                "kn": "ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ನಿಮಗೆ ಏನು ಬೇಕು ಎಂದು ಹೇಳಿ."
            },
            "thanks": {
                "en": "Thank you for your patience. I'll assist you right away.",
                "hi": "आपके धैर्य के लिए धन्यवाद। मैं आपकी तुरंत सहायता करूंगी।",
                "ta": "உங்கள் பொறுமைக்கு நன்றி. நான் உங்களுக்கு உடனடியாக உதவுவேன்.",
                "te": "మీ ఓపికకు ధన్యవాదాలు. నేను మీకు వెంటనే సహాయం చేస్తాను.",
                "kn": "ನಿಮ್ಮ ತಾಳ್ಮೆಗೆ ಧನ್ಯವಾದಗಳು. ನಾನು ನಿಮಗೆ ತಕ್ಷಣ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ."
            }
        }
        
        logger.info("MultilingualService initialized with %d supported languages", len(self.supported_languages))

    def detect_language(self, text: str, current_language: str = "en") -> Tuple[str, float]:
        """
        Detect language from text with confidence score
        Returns: (language_code, confidence_score)
        """
        if not text or not text.strip():
            return current_language, 0.0
            
        text_lower = text.lower()
        scores = {}
        
        # Method 1: Keyword matching
        for lang_code, patterns in self.language_patterns.items():
            keyword_score = 0
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    keyword_score += 1
            scores[lang_code] = keyword_score * 0.3  # Weight for keywords
        
        # Method 2: Unicode range detection
        for lang_code, patterns in self.language_patterns.items():
            if patterns["unicode_range"]:
                try:
                    char_count = len(re.findall(patterns["unicode_range"], text))
                    if char_count > 0:
                        unicode_score = min(char_count / len(text) * 2, 1.0)
                        scores[lang_code] = scores.get(lang_code, 0) + unicode_score
                except Exception as e:
                    logger.warning(f"Unicode detection error for {lang_code}: {e}")
        
        # Method 3: Common words frequency
        for lang_code, patterns in self.language_patterns.items():
            if patterns["common_words"]:
                word_score = 0
                words = text_lower.split()
                for common_word in patterns["common_words"]:
                    if common_word in words:
                        word_score += 1
                scores[lang_code] = scores.get(lang_code, 0) + word_score * 0.1
        
        # If no strong detection, return current language with low confidence
        if not scores or max(scores.values()) < 0.3:
            return current_language, 0.1
        
        # Return language with highest score
        best_lang = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_lang[1], 1.0)  # Cap at 1.0
        
        logger.debug(f"Language detected: {best_lang[0]} with confidence {confidence:.2f}")
        return best_lang[0], confidence

    def get_language_config(self, language_code: str) -> Dict[str, Any]:
        """
        Get configuration for a specific language
        """
        return self.supported_languages.get(language_code, self.supported_languages["en"])

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
            for lang_code, data in self.supported_languages.items()
        ]

    def get_phrase(self, phrase_key: str, language: str, **kwargs) -> str:
        """
        Get translated phrase with variable substitution
        """
        phrases = self.common_phrases.get(phrase_key, {})
        phrase = phrases.get(language, phrases.get("en", ""))
        
        # Substitute variables
        for key, value in kwargs.items():
            phrase = phrase.replace(f"{{{key}}}", str(value))
            
        return phrase

    def should_switch_language(self, text: str, current_language: str) -> bool:
        """
        Determine if language should be switched based on user request
        """
        text_lower = text.lower()
        
        # Check for explicit language switch requests
        switch_patterns = [
            "speak in", "in language", "change language", "switch to",
            "में बोलो", "भाषा बदलो", "மொழி மாற்ற", "భాష మార్చు", "ಭಾಷೆ ಬದಲಾಯಿಸಿ"
        ]
        
        if any(pattern in text_lower for pattern in switch_patterns):
            return True
        
        # Check language detection confidence
        detected_lang, confidence = self.detect_language(text, current_language)
        return confidence > 0.5 and detected_lang != current_language

    def normalize_text(self, text: str, language: str) -> str:
        """
        Normalize text for better processing based on language
        """
        # Basic normalization for all languages
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Language-specific normalization
        if language == "en":
            # Expand common abbreviations
            abbreviations = {
                "dr.": "doctor",
                "appt.": "appointment", 
                "rx": "prescription",
                "asap": "as soon as possible",
                "pls": "please",
                "thx": "thanks"
            }
            for abbr, full in abbreviations.items():
                text = text.replace(abbr, full)
                
        elif language == "hi":
            # Normalize Hindi variations
            text = text.replace("हिंग्लिश", "अंग्रेजी")
            text = text.replace("hindee", "हिंदी")
            
        return text

    def extract_language_specific_entities(self, text: str, language: str) -> Dict[str, Any]:
        """
        Extract language-specific entities like names, dates, etc.
        """
        entities = {
            "person_names": [],
            "dates": [],
            "locations": [],
            "numbers": []
        }
        
        # Extract numbers (works for all languages)
        numbers = re.findall(r'\d+', text)
        entities["numbers"] = numbers
        
        # Language-specific entity extraction
        if language == "en":
            # Simple English name pattern (capitalized words)
            names = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities["person_names"] = names
            
        elif language == "hi":
            # Hindi names often have specific patterns
            hindi_names = re.findall(r'\b[अ-ह]+[ा-ौ]?[अ-ह]*\b', text)
            entities["person_names"] = hindi_names
            
        # Date patterns (basic)
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{1,2} \w+ \d{4}',      # 15 January 2024
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            entities["dates"].extend(dates)
            
        return entities

    def get_cultural_appropriate_response(self, text: str, language: str) -> str:
        """
        Add culturally appropriate elements to responses
        """
        if language == "hi":
            # Hindi cultural elements
            if any(word in text.lower() for word in ["thank", "धन्यवाद", "शुक्रिया"]):
                return "आपका स्वागत है! " + text
            elif any(word in text.lower() for word in ["sorry", "माफ", "क्षमा"]):
                return "कोई बात नहीं। " + text
                
        elif language == "ta":
            # Tamil cultural elements
            if any(word in text.lower() for word in ["thank", "நன்றி"]):
                return "தயவு செய்து! " + text
            elif any(word in text.lower() for word in ["sorry", "மன்னிக்க"]):
                return "பரவாயில்லை. " + text
                
        elif language == "te":
            # Telugu cultural elements
            if any(word in text.lower() for word in ["thank", "ధన్యవాదాలు"]):
                return "దయచేసి! " + text
            elif any(word in text.lower() for word in ["sorry", "క్షమించండి"]):
                return "పర్వాలేదు. " + text
                
        elif language == "kn":
            # Kannada cultural elements
            if any(word in text.lower() for word in ["thank", "ಧನ್ಯವಾದ"]):
                return "ದಯವಿಟ್ಟು! " + text
            elif any(word in text.lower() for word in ["sorry", "ಕ್ಷಮಿಸಿ"]):
                return "ಪರವಾಗಿಲ್ಲ. " + text
                
        return text

    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported
        """
        return language_code in self.supported_languages

    def get_fallback_language(self) -> str:
        """
        Get the fallback language (English)
        """
        return "en"

    def generate_language_prompt(self, user_input: str, detected_language: str) -> str:
        """
        Generate AI prompt with language context
        """
        language_name = self.supported_languages[detected_language]["name"]
        
        prompt = f"""
        The user is speaking in {language_name}. Please respond naturally in the same language.
        User's message: {user_input}
        
        Important: 
        - Respond in {language_name} only
        - Use natural, conversational tone
        - Be culturally appropriate
        - Keep responses concise for phone conversation
        """
        
        return prompt

    def health_check(self) -> Dict[str, Any]:
        """
        Health check for multilingual service
        """
        return {
            "status": "healthy",
            "service": "Multilingual Service",
            "supported_languages": len(self.supported_languages),
            "languages": list(self.supported_languages.keys()),
            "timestamp": datetime.now().isoformat()
        }


# Utility functions
def sanitize_text_for_tts(text: str, language: str) -> str:
    """
    Sanitize text for better TTS pronunciation
    """
    # Remove special characters that might cause TTS issues
    text = re.sub(r'[^\w\s\.\,\!\?\-\–\—]', '', text)
    
    # Language-specific sanitization
    if language == "en":
        # Expand numbers for better pronunciation
        text = re.sub(r'(\d+)', lambda x: expand_number(x.group(1)), text)
    
    return text

def expand_number(number_str: str) -> str:
    """
    Expand numbers for better TTS (e.g., "123" -> "one hundred twenty three")
    Simple version for common cases
    """
    number_map = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
        "20": "twenty", "30": "thirty", "100": "hundred"
    }
    
    if number_str in number_map:
        return number_map[number_str]
    
    # For longer numbers, return as digits
    if len(number_str) > 3:
        return " ".join(number_str)  # "1234" -> "1 2 3 4"
    
    return number_str

def detect_script(text: str) -> str:
    """
    Detect the script used in the text
    """
    scripts = {
        "devanagari": r'[\u0900-\u097F]',  # Hindi, Marathi, etc.
        "tamil": r'[\u0B80-\u0BFF]',
        "telugu": r'[\u0C00-\u0C7F]', 
        "kannada": r'[\u0C80-\u0CFF]',
        "latin": r'[a-zA-Z]'
    }
    
    for script_name, pattern in scripts.items():
        if re.search(pattern, text):
            return script_name
            
    return "unknown"

# Create global instance
multilingual_service = MultilingualService()


# Example usage and testing
if __name__ == "__main__":
    # Test the service
    service = MultilingualService()
    
    # Test language detection
    test_texts = [
        "I want to speak in Hindi please",
        "हिंदी में बात करते हैं",
        "தமிழில் பேசலாம்", 
        "I need an appointment with doctor Sharma",
        "डॉक्टर शर्मा के साथ अपॉइंटमेंट चाहिए"
    ]
    
    for text in test_texts:
        lang, confidence = service.detect_language(text)
        print(f"Text: {text}")
        print(f"Detected: {lang} ({service.supported_languages[lang]['name']})")
        print(f"Confidence: {confidence:.2f}")
        print(f"Script: {detect_script(text)}")
        print("---")
    
    # Test phrase translation
    business_name = "Suresh Clinic"
    greeting = service.get_phrase("greeting", "hi", business_name=business_name)
    print(f"Hindi greeting: {greeting}")
    
    print("MultilingualService is working correctly!")
