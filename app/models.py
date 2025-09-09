
# app/models.py - SQLAlchemy data models for OwnBot

from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Enum
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

class ChannelType(enum.Enum):
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    WEB = "web"

class Client(Base):
    __tablename__ = "clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_name = Column(String, nullable=False)
    business_type = Column(Enum(BusinessType), nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(ClientStatus), default=ClientStatus.ACTIVE)

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    filename = Column(String, nullable=False)
    processed = Column(Boolean, default=False)
    upload_date = Column(DateTime, default=datetime.utcnow)

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    channel_type = Column(Enum(ChannelType), nullable=False)
    start_date = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime)
    months_purchased = Column(Integer, default=0)
    active = Column(Boolean, default=True)

class MessageLog(Base):
    __tablename__ = "message_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    channel = Column(String, nullable=False)
    message = Column(String, nullable=False)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
