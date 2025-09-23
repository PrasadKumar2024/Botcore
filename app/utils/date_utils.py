from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def add_months_to_date(start_date: Optional[datetime], months: int) -> datetime:
    """
    Add months to a given date. If no start_date provided, uses current date.
    
    Args:
        start_date: The starting date (None for current date)
        months: Number of months to add (1, 2, 3, 6, 12)
    
    Returns:
        New date after adding months
    """
    if start_date is None:
        start_date = datetime.utcnow()
    
    try:
        new_date = start_date + relativedelta(months=months)
        return new_date
    except Exception as e:
        logger.error(f"Error adding months to date: {str(e)}")
        # Fallback: approximate 30 days per month
        return start_date + timedelta(days=30 * months)

def calculate_expiry_date(start_date: Optional[datetime], months: int, current_expiry: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate expiry date based on subscription logic:
    - First payment: Start Date = Today, Expiry Date = Today + months
    - Renewal: Start Date remains, Expiry Date = Current Expiry + months
    
    Args:
        start_date: Current start date (None for new subscription)
        months: Number of months to add
        current_expiry: Current expiry date (for renewals)
    
    Returns:
        Dictionary with new_start_date and new_expiry_date
    """
    now = datetime.utcnow()
    
    # First payment logic
    if start_date is None or current_expiry is None:
        new_start_date = now
        new_expiry_date = add_months_to_date(now, months)
        logger.info(f"First payment: Start={new_start_date.date()}, Expiry={new_expiry_date.date()}")
    
    # Renewal logic
    else:
        new_start_date = start_date  # Start date remains unchanged
        new_expiry_date = add_months_to_date(current_expiry, months)
        logger.info(f"Renewal: Start remains {new_start_date.date()}, New Expiry={new_expiry_date.date()}")
    
    return {
        "start_date": new_start_date,
        "expiry_date": new_expiry_date,
        "months_added": months
    }

def check_subscription_active(expiry_date: datetime) -> bool:
    """
    Check if subscription is active by comparing expiry date with current date.
    
    Args:
        expiry_date: The subscription expiry date
    
    Returns:
        True if active, False if expired
    """
    now = datetime.utcnow()
    is_active = expiry_date > now
    
    if not is_active:
        logger.info(f"Subscription expired on {expiry_date.date()}")
    
    return is_active

def get_subscription_status(start_date: Optional[datetime], expiry_date: Optional[datetime]) -> Dict[str, Any]:
    """
    Get comprehensive subscription status.
    
    Args:
        start_date: Subscription start date
        expiry_date: Subscription expiry date
    
    Returns:
        Dictionary with status details
    """
    now = datetime.utcnow()
    
    if start_date is None or expiry_date is None:
        return {
            "status": "INACTIVE",
            "active": False,
            "start_date": None,
            "expiry_date": None,
            "days_remaining": 0,
            "expired": True
        }
    
    is_active = check_subscription_active(expiry_date)
    days_remaining = (expiry_date - now).days if expiry_date > now else 0
    
    status = "ACTIVE" if is_active else "EXPIRED"
    
    return {
        "status": status,
        "active": is_active,
        "start_date": start_date,
        "expiry_date": expiry_date,
        "days_remaining": max(0, days_remaining),
        "expired": not is_active
    }

def format_date_display(date: Optional[datetime]) -> str:
    """
    Format date for display in UI.
    
    Args:
        date: Date to format
    
    Returns:
        Formatted date string or "--/--/----" if None
    """
    if date is None:
        return "--/--/----"
    
    return date.strftime("%d/%m/%Y")

def get_days_until_expiry(expiry_date: datetime) -> int:
    """
    Calculate days until expiry.
    
    Args:
        expiry_date: Expiry date
    
    Returns:
        Number of days until expiry (negative if expired)
    """
    now = datetime.utcnow()
    return (expiry_date - now).days

def should_send_expiry_reminder(expiry_date: datetime, days_before: int = 7) -> bool:
    """
    Check if expiry reminder should be sent.
    
    Args:
        expiry_date: Expiry date
        days_before: Days before expiry to send reminder
    
    Returns:
        True if reminder should be sent
    """
    days_until = get_days_until_expiry(expiry_date)
    return 0 < days_until <= days_before

def validate_month_selection(months: int) -> bool:
    """
    Validate if month selection is valid.
    
    Args:
        months: Number of months selected
    
    Returns:
        True if valid
    """
    valid_months = [1, 2, 3, 6, 12]
    return months in valid_months

def get_subscription_duration_months(start_date: datetime, expiry_date: datetime) -> int:
    """
    Calculate total subscription duration in months.
    
    Args:
        start_date: Start date
        expiry_date: Expiry date
    
    Returns:
        Duration in months (rounded)
    """
    if start_date is None or expiry_date is None:
        return 0
    
    duration = relativedelta(expiry_date, start_date)
    total_months = duration.years * 12 + duration.months
    
    # Add partial month if needed
    if duration.days > 0:
        total_months += 1
    
    return total_months

# Utility functions for date comparisons
def is_date_in_future(date: datetime) -> bool:
    """Check if date is in the future."""
    return date > datetime.utcnow()

def is_date_in_past(date: datetime) -> bool:
    """Check if date is in the past."""
    return date < datetime.utcnow()

def get_current_datetime() -> datetime:
    """Get current UTC datetime."""
    return datetime.utcnow()

def format_datetime_for_db(date: datetime) -> str:
    """Format datetime for database storage."""
    return date.isoformat()

def parse_datetime_from_db(date_str: str) -> datetime:
    """Parse datetime from database string."""
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        logger.error(f"Failed to parse datetime: {date_str}")
        return datetime.utcnow() 
