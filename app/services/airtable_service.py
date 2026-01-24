import os
import requests
from dotenv import load_dotenv
from datetime import datetime


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
    try:
        # Convert from '2024-06-12' to '12/06/2024' format
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        airtable_date = date_obj.strftime("%d/%m/%Y")
        
        formula = f"AND({{date}}='{airtable_date}', {{time}}='{time}')"
        params = {"filterByFormula": formula}
        
        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        records = response.json().get("records", [])
        return len(records) == 0
    except Exception as e:
        print(f"Error checking slot: {e}")
        return False

def book_slot(date: str, time: str, phone: str | None):
    try:
        # Convert date format
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        airtable_date = date_obj.strftime("%d/%m/%Y")
        
        payload = {
            "fields": {
                "date": airtable_date,
                "time": time,
            }
        }
        response = requests.post(AIRTABLE_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Booking error: {e}")
        raise
