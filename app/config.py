from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "ownbot-index"
    
    # Gemini
    GEMINI_API_KEY: str
    
    # Twilio
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    
    # JWT
    JWT_SECRET: str
    
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()

# Test if all required variables are present
try:
    settings = get_settings()
    print("✅ All environment variables loaded successfully")
except Exception as e:
    print(f"❌ Error loading environment variables: {e}")
