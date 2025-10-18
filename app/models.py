
from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Enum, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
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
    
    # Bot activation status
    bot_active = Column(Boolean, default=False)  # Overall bot status
    
    # Relationships
    documents = relationship("Document", back_populates="client", cascade="all, delete-orphan")
    phone_numbers = relationship("PhoneNumber", back_populates="client", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="client", cascade="all, delete-orphan")
    whatsapp_profile = relationship("WhatsAppProfile", back_populates="client", uselist=False, cascade="all, delete-orphan")
    knowledge_chunks = relationship("KnowledgeChunk", back_populates="client", cascade="all, delete-orphan")  # ADDED
    message_logs = relationship("MessageLog", back_populates="client", cascade="all, delete-orphan")  # ADDED
    bot_settings = relationship("BotSettings", back_populates="client", cascade="all, delete-orphan")  # ADDED

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
    processed_at = Column(DateTime, nullable=True)  # When document was processed
    
    # Relationships
    client = relationship("Client", back_populates="documents")
    knowledge_chunks = relationship("KnowledgeChunk", back_populates="document", cascade="all, delete-orphan")  # ADDED

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
    
    # Bot activation status for individual bot types
    bot_activated = Column(Boolean, default=False)  # Whether this specific bot is activated
    
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
    
    # Relationship
    client = relationship("Client", back_populates="message_logs")  # ADDED
    
    # Index for faster querying
    __table_args__ = (
        Index('ix_message_logs_client_id_timestamp', 'client_id', 'timestamp'),
    )

# Knowledge Base Chunks for AI Processing
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)  # Extracted text chunk
    chunk_index = Column(Integer, nullable=False)  # Order of chunk in document
    vector_id = Column(String(255), nullable=True)  # Pinecone vector ID
    chunk_metadata = Column(Text, nullable=True)  # ADDED: JSON string for chunk metadata (filename, page, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships - UPDATED
    client = relationship("Client", back_populates="knowledge_chunks")
    document = relationship("Document", back_populates="knowledge_chunks")
    
    # Index for faster querying
    __table_args__ = (
        Index('ix_knowledge_chunks_client_id', 'client_id'),
        Index('ix_knowledge_chunks_document_id', 'document_id'),
    )

# Bot Settings and Configuration
class BotSettings(Base):
    __tablename__ = "bot_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    bot_type = Column(Enum(BotType), nullable=False)
    settings = Column(Text, nullable=True)  # JSON string for bot-specific settings
    is_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    client = relationship("Client", back_populates="bot_settings")  # UPDATED
    
    # Unique constraint: one settings per bot type per client
    __table_args__ = (
        Index('ix_bot_settings_client_bot_type', 'client_id', 'bot_type', unique=True),
    )
