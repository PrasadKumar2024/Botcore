import os
import requests
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")

AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

def check_slot(date: str, time: str) -> bool:
    """
    Checks if a slot is available based ONLY on Date and Time.
    Returns: True (Available) or False (Booked/Error).
    """
    # Formula: Find records where date=date AND time=time AND status='booked'
    formula = f"AND({{date}}='{date}', {{time}}='{time}', {{status}}='booked')"
    
    try:
        response = requests.get(
            AIRTABLE_URL, 
            headers=HEADERS, 
            params={"filterByFormula": formula}
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        
        # If 0 records found, the slot is empty (Available)
        return len(records) == 0
    except Exception as e:
        print(f"❌ Service Error (Check): {e}")
        return False # Assume booked on error to be safe

def book_slot(name: str, date: str, time: str, phone: str):
    """
    Books the slot with ALL user details.
    """
    payload = {
        "fields": {
            "name": name,       # User's Name
            "date": date,       # Appointment Date
            "time": time,       # Appointment Time
            "phone": phone,     # User's Phone
            "status": "booked", # Hardcoded status
            "source": "ai"      # Hardcoded source
        }
    }

    try:
        response = requests.post(AIRTABLE_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        print(f"✅ Service Success: Booked for {name}")
    except Exception as e:
        print(f"❌ Service Error (Book): {e}")
        raise e # Re-raise error so main.py knows it failed
