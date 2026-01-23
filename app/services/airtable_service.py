import os
import requests
from dotenv import load_dotenv

load_dotenv()

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
    Checks if a slot is available. Returns True/False.
    """
    formula = f"AND({{date}}='{date}', {{time}}='{time}')"
    params = {"filterByFormula": formula}

    try:
        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        records = response.json().get("records", [])
        return len(records) == 0
    except Exception as e:
        print(f"Error checking slot: {e}")
        return False 

def book_slot(date: str, time: str, phone: str = "Anonymous"):
    """
    Books the slot in Airtable.
    """
    payload = {
        "fields": {
            "date": date,
            "time": time,
            
        }
    }
    response = requests.post(AIRTABLE_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
