# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
import uuid

# Enums for business types and status
class BusinessType(str, Enum):
    RESTAURANT = "restaurant"
    GYM = "gym"
    CLINIC = "clinic"
    RETAIL = "retail"
    OTHER = "other"

class ClientStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"

class BotType(str, Enum):
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    WEB = "web"

# Client schemas
class ClientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, example="Suresh Kumar")
    business_name: str = Field(..., min_length=1, max_length=200, example="Suresh Store")
    business_type: BusinessType = Field(..., example=BusinessType.RETAIL)

class ClientCreate(ClientBase):
    pass

class ClientUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    business_name: Optional[str] = Field(None, min_length=1, max_length=200)
    business_type: Optional[BusinessType] = Field(None)
    status: Optional[ClientStatus] = Field(None)

class Client(ClientBase):
    id: uuid.UUID
    status: ClientStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Document schemas
class DocumentBase(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255, example="menu.pdf")
    file_size: int = Field(..., gt=0, example=1024000)

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: uuid.UUID
    client_id: uuid.UUID
    stored_filename: str
    file_path: str
    processed: bool
    processing_error: Optional[str] = None
    uploaded_at: datetime

    class Config:
        from_attributes = True

# Phone number schemas
class PhoneNumberBase(BaseModel):
    number: str = Field(..., min_length=10, max_length=20, example="+1234567890")
    country: str = Field(..., min_length=2, max_length=100, example="India")

class PhoneNumberCreate(PhoneNumberBase):
    pass

class PhoneNumber(PhoneNumberBase):
    id: uuid.UUID
    client_id: uuid.UUID
    twilio_sid: str
    is_active: bool
    purchased_at: datetime

    class Config:
        from_attributes = True

# Subscription schemas - simplified for month-based system
class SubscriptionBase(BaseModel):
    bot_type: BotType = Field(..., example=BotType.WHATSAPP)
    months: int = Field(..., ge=1, le=12, example=3)  # Months to add (1-12)

class SubscriptionCreate(SubscriptionBase):
    pass

class Subscription(SubscriptionBase):
    id: uuid.UUID
    client_id: uuid.UUID
    start_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# WhatsApp profile schemas
class WhatsAppProfileBase(BaseModel):
    business_name: str = Field(..., min_length=1, max_length=200, example="Suresh Store")
    address: Optional[str] = Field(None, min_length=1, max_length=500, example="123 Main Street")
    logo_url: Optional[str] = Field(None, min_length=1, max_length=500, example="/static/logos/suresh-store.png")

class WhatsAppProfileCreate(WhatsAppProfileBase):
    pass

class WhatsAppProfileUpdate(WhatsAppProfileBase):
    pass

class WhatsAppProfile(WhatsAppProfileBase):
    id: uuid.UUID
    client_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Chat and voice message schemas
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, example="What are your business hours?")
    session_id: Optional[str] = Field(None, example="session_12345")

class ChatResponse(BaseModel):
    response: str = Field(..., example="Our business hours are 9 AM to 5 PM, Monday to Friday.")
    session_id: str = Field(..., example="session_12345")
    sources: Optional[List[str]] = Field(None, example=["FAQ Document Page 5"])

class VoiceCallRequest(BaseModel):
    from_number: str = Field(..., min_length=10, max_length=20, example="+19876543210")
    to_number: str = Field(..., min_length=10, max_length=20, example="+1234567890")

class VoiceCallResponse(BaseModel):
    call_sid: str = Field(..., example="CAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    status: str = Field(..., example="initiated")
    message: Optional[str] = Field(None, example="Call initiated successfully")

# Statistics and reporting schemas
class ClientStats(BaseModel):
    total_messages: int = Field(0, example=150)
    active_chats: int = Field(0, example=5)
    voice_calls: int = Field(0, example=12)
    documents_processed: int = Field(0, example=8)
    last_activity: Optional[datetime] = Field(None)

class UsageReport(BaseModel):
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    most_common_questions: List[str]
