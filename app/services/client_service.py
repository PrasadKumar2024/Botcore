# app/services/client_service.py
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import uuid

from app.models import Client, Subscription, PhoneNumber, Document
from app.schemas import ClientCreate, ClientUpdate, SubscriptionCreate
#from app.services.twilio_service import release_phone_number

# Set up logging
logger = logging.getLogger(__name__)

def generate_embed_code(client_id: uuid.UUID) -> Dict[str, str]:
    """
    Generate unique embed code and chatbot URL for a client
    """
    unique_id = str(uuid.uuid4())
    
    # Generate embed code - using your actual domain from Render
    embed_code = f'<script src="https://botcore-z6j0.onrender.com/static/js/chat-widget.js?client_id={unique_id}"></script>'
    
    # Generate chatbot URL
    chatbot_url = f"https://botcore-z6j0.onrender.com/chat/{unique_id}"
    
    return {
        "embed_code": embed_code,
        "chatbot_url": chatbot_url,
        "unique_id": unique_id
    }

def create_client(db: Session, client_data: ClientCreate) -> Client:
    """
    Create a new client in the database with auto-generated embed code
    """
    try:
        # Create client instance with only the fields that exist in the model
        db_client = Client(
            name=client_data.name,
            business_name=client_data.business_name,
            business_type=client_data.business_type,
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Add to database first to get the client ID
        db.add(db_client)
        db.commit()
        db.refresh(db_client)
        
        # Generate embed code and update client
        embed_data = generate_embed_code(db_client.id)
        db_client.embed_code = embed_data["embed_code"]
        db_client.chatbot_url = embed_data["chatbot_url"]
        db_client.unique_id = embed_data["unique_id"]
        db_client.web_chat_active = False
        db_client.web_chat_start_date = None
        db_client.web_chat_expiry_date = None
        
        # Commit the embed code updates
        db.commit()
        db.refresh(db_client)
        
        logger.info(f"Created new client with embed code: {db_client.id} - {db_client.business_name}")
        return db_client
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating client: {str(e)}")
        raise

def get_client_by_id(db: Session, client_id: uuid.UUID) -> Optional[Client]:
    """
    Get a client by ID with all related data
    """
    return db.query(Client).filter(Client.id == client_id).first()

def get_all_clients(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    active_only: bool = False,
    search: Optional[str] = None
) -> List[Client]:
    """
    Get all clients with optional filtering and search
    """
    query = db.query(Client)
    
    if active_only:
        query = query.filter(Client.status == "active")
    
    if search:
        search_filter = or_(
            Client.name.ilike(f"%{search}%"),
            Client.business_name.ilike(f"%{search}%"),
        )
        query = query.filter(search_filter)
    
    return query.offset(skip).limit(limit).all()

def update_client_details(db: Session, client_id: uuid.UUID, client_data: ClientUpdate) -> Optional[Client]:
    """
    Update client information
    """
    try:
        db_client = db.query(Client).filter(Client.id == client_id).first()
        if not db_client:
            return None
        
        # Update only provided fields
        update_data = client_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(db_client, field):
                setattr(db_client, field, value)
        
        db_client.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_client)
        
        logger.info(f"Updated client: {client_id}")
        return db_client
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating client {client_id}: {str(e)}")
        raise

def delete_client(db: Session, client_id: uuid.UUID) -> bool:
    """
    Delete a client and associated resources
    """
    try:
        db_client = db.query(Client).filter(Client.id == client_id).first()
        if not db_client:
            return False
        
        # First, release all phone numbers associated with this client
        phone_numbers = db.query(PhoneNumber).filter(PhoneNumber.client_id == client_id).all()
        for phone in phone_numbers:
            try:
                release_phone_number(phone.twilio_sid)
            except Exception as e:
                logger.warning(f"Failed to release Twilio number {phone.twilio_sid}: {str(e)}")
        
        # Delete client (this will cascade to related records based on model relationships)
        db.delete(db_client)
        db.commit()
        
        logger.info(f"Deleted client: {client_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting client {client_id}: {str(e)}")
        raise

def get_client_subscriptions(db: Session, client_id: uuid.UUID) -> Optional[List[Subscription]]:
    """
    Get all subscriptions for a client
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None
    
    return client.subscriptions

def create_client_subscription(db: Session, client_id: uuid.UUID, subscription_data: SubscriptionCreate) -> Optional[Subscription]:
    """
    Create a new subscription for a client
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        # Calculate start and expiry dates based on months
        now = datetime.utcnow()
        expiry_date = now + relativedelta(months=+subscription_data.months)
        
        # Create subscription
        db_subscription = Subscription(
            client_id=client_id,
            bot_type=subscription_data.bot_type,
            start_date=now,
            expiry_date=expiry_date,
            is_active=True,
            created_at=now,
            updated_at=now
        )
        
        db.add(db_subscription)
        db.commit()
        db.refresh(db_subscription)
        
        logger.info(f"Created subscription for client: {client_id}")
        return db_subscription
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating subscription for client {client_id}: {str(e)}")
        raise

def get_client_phone_numbers(db: Session, client_id: uuid.UUID) -> Optional[List[PhoneNumber]]:
    """
    Get all phone numbers assigned to a client
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None
    
    return client.phone_numbers

def add_phone_number_to_client(db: Session, client_id: uuid.UUID, twilio_sid: str, number: str, country: str) -> Optional[PhoneNumber]:
    """
    Add a phone number to a client in the database
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        # Create phone number record
        db_phone = PhoneNumber(
            client_id=client_id,
            twilio_sid=twilio_sid,
            number=number,
            country=country,
            is_active=True,
            purchased_at=datetime.utcnow()
        )
        
        db.add(db_phone)
        db.commit()
        db.refresh(db_phone)
        
        logger.info(f"Added phone number {number} to client: {client_id}")
        return db_phone
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding phone number to client {client_id}: {str(e)}")
        raise

def remove_phone_number_from_client(db: Session, client_id: uuid.UUID, phone_sid: str) -> bool:
    """
    Remove a phone number from a client in the database
    """
    try:
        phone = db.query(PhoneNumber).filter(
            PhoneNumber.client_id == client_id, 
            PhoneNumber.twilio_sid == phone_sid
        ).first()
        
        if not phone:
            return False
        
        db.delete(phone)
        db.commit()
        
        logger.info(f"Removed phone number {phone_sid} from client: {client_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing phone number {phone_sid} from client {client_id}: {str(e)}")
        raise

def get_client_stats(db: Session, client_id: uuid.UUID) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a client
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None
    
    # Count documents
    documents_count = db.query(Document).filter(Document.client_id == client_id).count()
    
    # Count active phone numbers
    phone_numbers_count = db.query(PhoneNumber).filter(
        PhoneNumber.client_id == client_id, 
        PhoneNumber.is_active == True
    ).count()
    
    # Check active subscription
    has_active_subscription = any(
        sub.is_active and (sub.expiry_date is None or sub.expiry_date >= datetime.utcnow())
        for sub in client.subscriptions
    )
    
    # Get latest activity (simplified - you might want to track this properly)
    latest_activity = client.updated_at
    
    return {
        "client_id": client_id,
        "documents_count": documents_count,
        "phone_numbers_count": phone_numbers_count,
        "has_active_subscription": has_active_subscription,
        "latest_activity": latest_activity,
        "status": client.status
    }

def get_client_by_phone_number(db: Session, phone_number: str) -> Optional[Client]:
    """
    Find a client by their phone number
    """
    phone = db.query(PhoneNumber).filter(PhoneNumber.number == phone_number).first()
    if not phone:
        return None
    
    return phone.client
'''
def deactivate_client(db: Session, client_id: uuid.UUID) -> Optional[Client]:
    """
    Deactivate a client (soft delete)
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        client.status = "inactive"
        client.updated_at = datetime.utcnow()
        
        # Also deactivate all phone numbers
        for phone in client.phone_numbers:
            phone.is_active = False
            phone.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(client)
        
        logger.info(f"Deactivated client: {client_id}")
        return client
   except Exception as e:
        db.rollback()
        logger.error(f"Error deactivating client: {str(e)}")
        raise '''

def regenerate_embed_code(db: Session, client_id: uuid.UUID) -> Optional[Client]:
    """
    Regenerate embed code for an existing client
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        # Generate new embed code
        embed_data = generate_embed_code(client_id)
        client.embed_code = embed_data["embed_code"]
        client.chatbot_url = embed_data["chatbot_url"]
        client.unique_id = embed_data["unique_id"]
        client.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(client)
        
        logger.info(f"Regenerated embed code for client: {client_id}")
        return client
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error regenerating embed code for client {client_id}: {str(e)}")
        raise

def activate_web_chat(db: Session, client_id: uuid.UUID) -> Optional[Client]:
    """
    Activate web chat bot for a client
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        client.web_chat_active = True
        client.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(client)
        
        logger.info(f"Activated web chat for client: {client_id}")
        return client
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error activating web chat for client {client_id}: {str(e)}")
        raise

def deactivate_web_chat(db: Session, client_id: uuid.UUID) -> Optional[Client]:
    """
    Deactivate web chat bot for a client
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        client.web_chat_active = False
        client.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(client)
        
        logger.info(f"Deactivated web chat for client: {client_id}")
        return client
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deactivating web chat for client {client_id}: {str(e)}")
        raise

def set_web_chat_subscription_dates(db: Session, client_id: uuid.UUID, start_date: datetime, expiry_date: datetime) -> Optional[Client]:
    """
    Set web chat subscription dates for a client
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        client.web_chat_start_date = start_date
        client.web_chat_expiry_date = expiry_date
        client.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(client)
        
        logger.info(f"Set web chat subscription dates for client: {client_id}")
        return client
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error setting web chat subscription dates for client {client_id}: {str(e)}")
        raise   
    
