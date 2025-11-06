# app/services/__init__.py


'''from .client_service import ClientService
from .document_service import DocumentService
from .subscription_service import SubscriptionService
from .twilio_service import TwilioService '''
from .gemini_service import GeminiService
from .pinecone_service import PineconeService
#from .speech_service import SpeechService
#from .multilingual_service import MultilingualService

# Export all service classes for easy importing
__all__ = [
    #"ClientService",
  #  "DocumentService", 
   # "SubscriptionService",
  #  "TwilioService",
    "GeminiService",
    "PineconeService", 
 #   "SpeechService",
   # "MultilingualService"
]

# Optional: You can also create service factory functions if needed
''' def get_client_service():
    """Factory function for ClientService"""
    return ClientService()

def get_document_service():
    """Factory function for DocumentService"""
    return DocumentService()

def get_subscription_service():
    """Factory function for SubscriptionService"""
    return SubscriptionService()

def get_twilio_service():
    """Factory function for TwilioService"""
    return TwilioService() '''

def get_gemini_service():
    """Factory function for GeminiService"""
    return GeminiService()

def get_pinecone_service():
    """Factory function for PineconeService"""
    return PineconeService()
'''
def get_speech_service():
    """Factory function for SpeechService"""
    return SpeechService()

def get_multilingual_service():
    """Factory function for MultilingualService"""
    return MultilingualService() '''
