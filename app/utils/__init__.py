# app/utils/__init__.py

from .file_utils import FileUtils
from .date_utils import DateUtils
from .audio_utils import AudioUtils
from .language_utils import LanguageUtils

# Note: payment_utils.py is intentionally excluded as per your requirement
# since we don't use payment processing in the dashboard

# Export all utility classes for easy importing
__all__ = [
    "FileUtils",
    "DateUtils", 
    "AudioUtils",
    "LanguageUtils"
]

# Optional: Factory functions for dependency injection
def get_file_utils():
    """Factory function for FileUtils"""
    return FileUtils()

def get_date_utils():
    """Factory function for DateUtils"""
    return DateUtils()

def get_audio_utils():
    """Factory function for AudioUtils"""
    return AudioUtils()

def get_language_utils():
    """Factory function for LanguageUtils"""
    return LanguageUtils()

# Utility functions that might be commonly used across the application
def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """Validate file extension against allowed list"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_days_until_expiry(expiry_date) -> int:
    """Calculate days remaining until expiry date"""
    from datetime import date
    if not expiry_date:
        return 0
    today = date.today()
    delta = expiry_date - today
    return delta.days if delta.days > 0 else 0

def format_phone_number(phone_number: str) -> str:
    """Format phone number for display"""
    if not phone_number:
        return ""
    # Remove any non-digit characters
    cleaned = ''.join(filter(str.isdigit, phone_number))
    # Format based on length (basic international formatting)
    if len(cleaned) == 10:
        return f"+1{cleaned}"  # US/Canada default
    elif len(cleaned) > 10:
        return f"+{cleaned}"
    return cleaned
