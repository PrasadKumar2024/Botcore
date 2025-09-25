from typing import Optional
import logging
from twilio.rest import Client
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class SpeechService:
    def __init__(self):
        self.twilio_client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    
    def text_to_speech(self, text: str) -> str:
        """
        Convert text to speech using Twilio's basic text-to-speech
        Returns TwiML <Say> verb for voice responses
        """
        try:
            # Basic text cleaning for TTS
            cleaned_text = self._clean_text_for_speech(text)
            
            # Generate TwiML for speech
            twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="woman">{cleaned_text}</Say></Response>'
            return twiml
            
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            # Fallback response
            return '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="woman">I apologize, but I encountered an error. Please try again.</Say></Response>'
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean text for better speech synthesis
        """
        if not text:
            return "I don't have an answer for that question. Please try asking something else."
        
        # Remove special characters that might cause TTS issues
        import re
        cleaned = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Limit length to avoid very long responses
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."
            
        return cleaned
    
    def generate_voice_response(self, question: str, answer: str) -> str:
        """
        Generate complete voice response TwiML for a Q&A
        """
        try:
            # Clean and prepare the answer for speech
            speech_answer = self._clean_text_for_speech(answer)
            
            twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="woman">You asked: {question}</Say>
    <Pause length="1"/>
    <Say voice="woman">{speech_answer}</Say>
    <Pause length="2"/>
    <Say voice="woman">What else would you like to know?</Say>
</Response>'''
            return twiml
            
        except Exception as e:
            logger.error(f"Error generating voice response: {e}")
            return self.text_to_speech("I apologize, but I encountered an error processing your request.")
    
    def generate_welcome_message(self, business_name: str) -> str:
        """
        Generate welcome message for voice calls
        """
        welcome_text = f"Hello! Thank you for calling {business_name}. How can I help you today?"
        return self.text_to_speech(welcome_text)
    
    def generate_goodbye_message(self) -> str:
        """
        Generate goodbye message for voice calls
        """
        goodbye_text = "Thank you for calling. Have a great day!"
        return self.text_to_speech(goodbye_text)
    
    def generate_error_response(self) -> str:
        """
        Generate error response for voice calls
        """
        error_text = "I'm sorry, I'm having trouble understanding. Please try calling again later."
        return self.text_to_speech(error_text)
