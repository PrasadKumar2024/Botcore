import os
import requests

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")

AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

def check_slot(date: str, time: str) -> bool:
    formula = f"AND({{date}}='{date}', {{time}}='{time}', {{status}}='booked')"

    params = {
        "filterByFormula": formula
    }

    response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
    response.raise_for_status()

    records = response.json().get("records", [])
    return len(records) == 0  # True if available


def book_slot(date: str, time: str, phone: str | None):
    payload = {
        "fields": {
            "date": date,
            "time": time,
            "status": "booked",
            "source": "ai",
            "phone": phone or "unknown"
        }
    }

    response = requests.post(AIRTABLE_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
