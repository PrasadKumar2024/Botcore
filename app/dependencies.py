from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Generator
import logging

from app.database import SessionLocal
from app.config import settings

logger = logging.getLogger(__name__)

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function that provides a database session.
    Used by all routes that need database access.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )
    finally:
        db.close()

def get_current_client(client_id: int, db: Session = Depends(get_db)):
    """
    Dependency to get and validate a client by ID.
    Used by routes that require a specific client.
    """
    from app.models import Client
    from app.services.client_service import get_client_by_id
    
    client = get_client_by_id(db, client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client with ID {client_id} not found"
        )
    return client

def validate_subscription(client_id: int, bot_type: str, db: Session = Depends(get_db)):
    """
    Dependency to validate if a client has an active subscription for a specific bot type.
    Used by chat and voice endpoints.
    """
    from app.models import Subscription
    from app.utils.date_utils import check_subscription_active
    
    subscription = db.query(Subscription).filter(
        Subscription.client_id == client_id,
        Subscription.bot_type == bot_type
    ).first()
    
    if not subscription or not check_subscription_active(subscription):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"{bot_type.capitalize()} bot subscription is not active"
        )
    return subscription

def get_twilio_service():
    """
    Dependency to get Twilio service instance.
    """
    from app.services.twilio_service import twilio_service
    return twilio_service

def get_gemini_service():
    """
    Dependency to get Gemini service instance.
    """
    from app.services.gemini_service import gemini_service
    return gemini_service

def get_pinecone_service():
    """
    Dependency to get Pinecone service instance.
    """
    from app.services.pinecone_service import pinecone_service
    return pinecone_service

def get_subscription_service():
    """
    Dependency to get Subscription service instance.
    """
    from app.services.subscription_service import subscription_service
    return subscription_service

def get_document_service():
    """
    Dependency to get Document service instance.
    """
    from app.services.document_service import document_service
    return document_service

# Configuration dependencies
def get_settings():
    """
    Dependency to get application settings.
    """
    return settings

# Security dependencies (for future authentication)
def verify_api_key(api_key: str = None):
    """
    Dependency for API key authentication (if needed in future).
    """
    if not api_key or api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# File upload validation dependency
async def validate_file_upload(file_content: bytes, filename: str):
    """
    Dependency to validate uploaded files.
    """
    from app.utils.file_utils import validate_pdf_file
    
    validation_result = await validate_pdf_file(file_content, filename)
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_result["error"]
        )
    return validation_result

# Client creation validation
def validate_client_creation(client_data: dict):
    """
    Dependency to validate client creation data.
    """
    if not client_data.get("name") or len(client_data["name"].strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client name must be at least 2 characters long"
        )
    
    valid_business_types = ["restaurant", "gym", "clinic", "retail", "other"]
    if client_data.get("business_type") not in valid_business_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Business type must be one of: {', '.join(valid_business_types)}"
        )
    
    return client_data

# Subscription validation
def validate_subscription_data(months: int, bot_type: str):
    """
    Dependency to validate subscription data.
    """
    valid_months = [1, 2, 3, 6, 12]
    if months not in valid_months:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Months must be one of: {valid_months}"
        )
    
    valid_bot_types = ["whatsapp", "voice", "web"]
    if bot_type not in valid_bot_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bot type must be one of: {', '.join(valid_bot_types)}"
        )
    
    return {"months": months, "bot_type": bot_type}

# Phone number purchase validation
def validate_phone_number_purchase(country: str):
    """
    Dependency to validate phone number purchase data.
    """
    # Example list of supported countries - expand as needed
    supported_countries = ["US", "IN", "GB", "CA", "AU"]
    
    if country.upper() not in supported_countries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Country not supported. Supported countries: {', '.join(supported_countries)}"
        )
    
    return country.upper()

# Error handling dependency
async def global_exception_handler(request, exc):
    """
    Global exception handler dependency.
    """
    logger.error(f"Global exception handler: {str(exc)}")
    if isinstance(exc, HTTPException):
        return exc
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )
