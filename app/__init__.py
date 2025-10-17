"""
OwnBot - AI-Powered Chatbot Management Platform
"""

__version__ = "1.0.0"
__author__ = "OwnBot Team"
__description__ = "AI-powered chatbot management platform"

# Package initialization - MINIMAL VERSION TO FIX IMPORT ERRORS
from app.config import settings
from app.database import Base, engine, SessionLocal

# Import models to ensure they are registered with SQLAlchemy
from app.models import Client, Document, PhoneNumber, Subscription, WhatsAppProfile

# TEMPORARILY COMMENT OUT ALL SERVICES AND ROUTES
# We'll add these back one by one after the app starts

print(f"OwnBot {__version__} initialized successfully!")
