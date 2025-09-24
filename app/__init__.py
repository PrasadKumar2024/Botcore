"""
OwnBot - AI-Powered Chatbot Management Platform

A comprehensive platform for managing WhatsApp, Voice, and Web chatbots
with PDF-based knowledge systems and subscription management.
"""

__version__ = "1.0.0"
__author__ = "OwnBot Team"
__description__ = "AI-powered chatbot management platform"

# Package initialization
from app.config import settings
from app.database import Base, engine, SessionLocal
from app.utils import date_utils, file_utils

# Import models to ensure they are registered with SQLAlchemy
from app.models import Client, Document, PhoneNumber, Subscription, WhatsAppProfile

# Import services for easy access
from app.services import (
    client_service,
    document_service, 
    subscription_service,
    twilio_service,
    gemini_service,
    pinecone_service
)

# Import routes
from app.routes import (
    clients,
    documents,
    subscriptions,
    numbers,
    chat,
    voice
)

# Make important classes available at package level for easier imports
__all__ = [
    # Core components
    'settings',
    'Base',
    'engine', 
    'SessionLocal',
    
    # Models
    'Client',
    'Document',
    'PhoneNumber',
    'Subscription',
    'WhatsAppProfile',
    
    # Services
    'client_service',
    'document_service',
    'subscription_service',
    'twilio_service', 
    'gemini_service',
    'pinecone_service',
    
    # Routes
    'clients',
    'documents',
    'subscriptions',
    'numbers',
    'chat',
    'voice',
    
    # Utilities
    'date_utils',
    'file_utils'
]

# Package metadata
package_metadata = {
    "name": "OwnBot",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "Multi-bot management (WhatsApp, Voice, Web Chat)",
        "PDF-based knowledge system",
        "Subscription billing",
        "Twilio integration",
        "Gemini AI integration",
        "Pinecone vector storage"
    ]
}

def get_package_info():
    """Return package information"""
    return package_metadata

# Initialize package-level logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

print(f"OwnBot {__version__} initialized successfully!")
