
# app/schemas.py
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum

# Enums for predefined choices
class SubscriptionPlan(str, Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"

class PhoneNumberType(str, Enum):
    VOICE = "voice"
    SMS = "sms"
    WHATSAPP = "whatsapp"

class ClientStatus(str, Enum):
    ACTIVE = "active"
    TRIAL = "trial"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

# Base schemas
class ClientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, example="John's Law Firm")
    email: EmailStr = Field(..., example="contact@johnslaw.com")
    business_name: str = Field(..., min_length=1, max_length=200, example="John's Law Firm LLC")
    industry: str = Field(..., min_length=1, max_length=100, example="Legal Services")
    website: Optional[str] = Field(None, example="https://johnslaw.com")
    timezone: str = Field(default="UTC", example="America/New_York")

class SubscriptionBase(BaseModel):
    plan_type: SubscriptionPlan = Field(..., example=SubscriptionPlan.PROFESSIONAL)
    start_date: date = Field(default_factory=date.today)
    end_date: Optional[date] = Field(None)
    is_active: bool = Field(default=True)

class PhoneNumberBase(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20, example="+1234567890")
    friendly_name: str = Field(..., min_length=1, max_length=50, example="John's Law Main Line")
    number_type: PhoneNumberType = Field(default=PhoneNumberType.VOICE, example=PhoneNumberType.VOICE)
    twilio_sid: Optional[str] = Field(None, example="PNxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

class DocumentBase(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255, example="terms-of-service.pdf")
    original_filename: str = Field(..., min_length=1, max_length=255, example="Terms of Service v2.pdf")
    file_size: int = Field(..., gt=0, example=1024000)
    pages: int = Field(..., gt=0, example=15)

# Create schemas (for POST requests)
class ClientCreate(ClientBase):
    password: str = Field(..., min_length=8, example="securepassword123")
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        # Add more password strength validation if needed
        return v

class SubscriptionCreate(SubscriptionBase):
    plan_type: SubscriptionPlan
    start_date: Optional[date] = None  # Let the server set default

class PhoneNumberCreate(PhoneNumberBase):
    phone_number: str
    friendly_name: str

class DocumentCreate(DocumentBase):
    client_id: int = Field(..., gt=0, example=1)
    uploader_id: int = Field(..., gt=0, example=1)

# Update schemas (for PUT/PATCH requests)
class ClientUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = Field(None)
    business_name: Optional[str] = Field(None, min_length=1, max_length=200)
    industry: Optional[str] = Field(None, min_length=1, max_length=100)
    website: Optional[str] = Field(None)
    timezone: Optional[str] = Field(None)
    status: Optional[ClientStatus] = Field(None)

class SubscriptionUpdate(BaseModel):
    plan_type: Optional[SubscriptionPlan] = Field(None)
    end_date: Optional[date] = Field(None)
    is_active: Optional[bool] = Field(None)

# Response schemas (for GET responses)
class PhoneNumber(PhoneNumberBase):
    id: int
    client_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Subscription(SubscriptionBase):
    id: int
    client_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Document(DocumentBase):
    id: int
    client_id: int
    uploader_id: int
    created_at: datetime
    updated_at: datetime
    processed: bool = Field(default=False)
    processing_error: Optional[str] = Field(None)

    class Config:
        from_attributes = True

class Client(ClientBase):
    id: int
    status: ClientStatus = Field(default=ClientStatus.ACTIVE)
    created_at: datetime
    updated_at: datetime
    has_active_subscription: bool = Field(default=False)

    class Config:
        from_attributes = True

# Detailed response schemas with relationships
class ClientWithDetails(Client):
    subscriptions: List[Subscription] = Field(default_factory=list)
    phone_numbers: List[PhoneNumber] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)

# Chat and Voice related schemas
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, example="What are your business hours?")
    session_id: Optional[str] = Field(None, example="session_12345")
    client_id: int = Field(..., gt=0, example=1)

class ChatResponse(BaseModel):
    response: str = Field(..., example="Our business hours are 9 AM to 5 PM, Monday to Friday.")
    session_id: str = Field(..., example="session_12345")
    sources: Optional[List[str]] = Field(None, example=["FAQ Document Page 5"])

class VoiceCallRequest(BaseModel):
    from_number: str = Field(..., min_length=10, max_length=20, example="+19876543210")
    to_number: str = Field(..., min_length=10, max_length=20, example="+1234567890")
    client_id: int = Field(..., gt=0, example=1)

class VoiceCallResponse(BaseModel):
    call_sid: str = Field(..., example="CAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    status: str = Field(..., example="initiated")
    message: Optional[str] = Field(None, example="Call initiated successfully")

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    client_id: Optional[int] = None
    username: Optional[str] = None

# Statistics and reporting schemas
class ClientStats(BaseModel):
    total_messages: int = Field(0, example=150)
    active_chats: int = Field(0, example=5)
    voice_calls: int = Field(0, example=12)
    documents_processed: int = Field(0, example=8)
    last_activity: Optional[datetime] = Field(None)

class UsageReport(BaseModel):
    period_start: date
    period_end: date
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    most_common_questions: List[Dict[str, Any]]
