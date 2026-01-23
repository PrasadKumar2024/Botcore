import os
import requests
from dotenv import load_dotenv

# Load env variables safely
load_dotenv()

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")

AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

def check_slot_service(date: str, time: str) -> bool:
    """
    Checks Airtable. Returns True (Python Boolean) if available.
    """
    # Formula: date=date AND time=time AND status='booked'
    formula = f"AND({{date}}='{date}', {{time}}='{time}', {{status}}='booked')"
    params = {"filterByFormula": formula}

    try:
        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        records = response.json().get("records", [])
        # If 0 records found, it means no one has booked it yet -> Available
        return len(records) == 0
    except Exception as e:
        print(f"Service Error (Check): {e}")
        return False # Fail safe

def book_slot_service(date: str, time: str, phone: str = "Anonymous"):
    """
    Writes to Airtable. Handles 'Anonymous' if no phone provided.
    """
    payload = {
        "fields": {
            "date": date,
            "time": time,
            "status": "booked",
            "source": "ai",
            "phone": phone  # Logic handles the default "Anonymous"
        }
    }
    response = requests.post(AIRTABLE_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
