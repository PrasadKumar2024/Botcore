from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Enum, Text
from sqlalchemy.dialects.sqlite import UUID
import uuid
from datetime import datetime
from app.database import Base
import enum

# Enums for business types and status
class BusinessType(enum.Enum):
    RESTAURANT = "restaurant"
    GYM = "gym"
    CLINIC = "clinic"
    RETAIL = "retail"
    OTHER = "other"

class ClientStatus(enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"

class BotType(enum.Enum):
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    WEB = "web"

class Client(Base):
    __tablename__ = "clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)  # Client's name
    business_name = Column(String(200), nullable=False)  # Business name
    business_type = Column(Enum(BusinessType), nullable=False)
    status = Column(Enum(ClientStatus), default=ClientStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="client", cascade="all, delete-orphan")
    phone_numbers = relationship("PhoneNumber", back_populates="client", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="client", cascade="all, delete-orphan")
    whatsapp_profile = relationship("WhatsAppProfile", back_populates="client", uselist=False, cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    filename = Column(String(255), nullable=False)  # Original filename
    stored_filename = Column(String(255), nullable=False)  # Unique filename on server
    file_path = Column(String(500), nullable=False)  # Path to stored file
    file_size = Column(Integer, nullable=False)  # File size in bytes
    processed = Column(Boolean, default=False)  # Whether PDF text extracted
    processing_error = Column(Text, nullable=True)  # Error if processing failed
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    client = relationship("Client", back_populates="documents")

class PhoneNumber(Base):
    __tablename__ = "phone_numbers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    number = Column(String(20), nullable=False, unique=True)  # E.164 format: +1234567890
    country = Column(String(100), nullable=False)  # Country name
    twilio_sid = Column(String(50), nullable=False, unique=True)  # Twilio SID
    is_active = Column(Boolean, default=True)
    purchased_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    client = relationship("Client", back_populates="phone_numbers")

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    bot_type = Column(Enum(BotType), nullable=False)  # whatsapp, voice, or web
    start_date = Column(DateTime, nullable=True)  # Null initially
    expiry_date = Column(DateTime, nullable=True)  # Null initially
    is_active = Column(Boolean, default=False)  # Inactive until months added
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    client = relationship("Client", back_populates="subscriptions")

class WhatsAppProfile(Base):
    __tablename__ = "whatsapp_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    business_name = Column(String(200), nullable=False)
    address = Column(Text, nullable=True)
    logo_url = Column(String(500), nullable=True)  # Path to uploaded logo
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    client = relationship("Client", back_populates="whatsapp_profile")

class MessageLog(Base):
    __tablename__ = "message_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    channel = Column(String(20), nullable=False)  # whatsapp, voice, web
    message_text = Column(Text, nullable=False)  # User's message
    response_text = Column(Text, nullable=True)  # AI's response
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Index for faster querying
    __table_args__ = (
        Index('ix_message_logs_client_id_timestamp', 'client_id', 'timestamp'),
    )
