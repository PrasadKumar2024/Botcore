# app/services/client_service.py
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging

from app.models import Client, Subscription, PhoneNumber, Document
from app.schemas import ClientCreate, ClientUpdate, SubscriptionCreate, PhoneNumberCreate
from app.services.twilio_service import purchase_phone_number, release_phone_number

# Set up logging
logger = logging.getLogger(__name__)

def create_client(db: Session, client_data: ClientCreate) -> Client:
    """
    Create a new client in the database
    """
    try:
        # Create client instance
        db_client = Client(
            name=client_data.name,
            email=client_data.email,
            business_name=client_data.business_name,
            industry=client_data.industry,
            website=client_data.website,
            timezone=client_data.timezone,
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Add to database
        db.add(db_client)
        db.commit()
        db.refresh(db_client)
        
        logger.info(f"Created new client: {db_client.id} - {db_client.business_name}")
        return db_client
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating client: {str(e)}")
        raise

def get_client_by_id(db: Session, client_id: int) -> Optional[Client]:
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
            Client.email.ilike(f"%{search}%"),
            Client.industry.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    return query.offset(skip).limit(limit).all()

def update_client_details(db: Session, client_id: int, client_data: ClientUpdate) -> Optional[Client]:
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

def delete_client(db: Session, client_id: int) -> bool:
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

def get_client_subscriptions(db: Session, client_id: int) -> Optional[List[Subscription]]:
    """
    Get all subscriptions for a client
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None
    
    return client.subscriptions

def create_client_subscription(db: Session, client_id: int, subscription_data: SubscriptionCreate) -> Optional[Subscription]:
    """
    Create a new subscription for a client
    """
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        # Create subscription
        db_subscription = Subscription(
            client_id=client_id,
            plan_type=subscription_data.plan_type,
            start_date=subscription_data.start_date or date.today(),
            end_date=subscription_data.end_date,
            is_active=subscription_data.is_active,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
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

def get_client_phone_numbers(db: Session, client_id: int) -> Optional[List[PhoneNumber]]:
    """
    Get all phone numbers assigned to a client
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None
    
    return client.phone_numbers

def add_phone_number_to_client(db: Session, client_id: int, twilio_sid: str, phone_data: PhoneNumberCreate) -> Optional[PhoneNumber]:
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
            phone_number=phone_data.phone_number,
            friendly_name=phone_data.friendly_name,
            number_type=phone_data.number_type,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(db_phone)
        db.commit()
        db.refresh(db_phone)
        
        logger.info(f"Added phone number {phone_data.phone_number} to client: {client_id}")
        return db_phone
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding phone number to client {client_id}: {str(e)}")
        raise

def remove_phone_number_from_client(db: Session, client_id: int, phone_sid: str) -> bool:
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

def get_client_stats(db: Session, client_id: int) -> Optional[Dict[str, Any]]:
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
        sub.is_active and (sub.end_date is None or sub.end_date >= date.today())
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
    phone = db.query(PhoneNumber).filter(PhoneNumber.phone_number == phone_number).first()
    if not phone:
        return None
    
    return phone.client

def deactivate_client(db: Session, client_id: int) -> Optional[Client]:
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
        logger.error(f"Error deactivating client {client_id}: {str(e)}")
        raise
