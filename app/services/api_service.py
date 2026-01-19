import requests
import logging
from typing import Optional

# SETUP LOGGING
logger = logging.getLogger(__name__)

# CONFIGURATION
AIRTABLE_BASE_ID = "appz9qfmC5R20hB1d"
AIRTABLE_TABLE_NAME = "Management"
AIRTABLE_TOKEN = "patWqPObGDr5PjPwd.387ffcf217e1fe9a73abbef24dba18d74a23345c3a3eef0deafa6a47216f9082"

def fetch_live_data(query_type: str, specific_name: Optional[str] = None) -> str:
    """
    Production-grade connector for Airtable.
    Includes: Timeouts, Error Codes, and Fallback responses.
    """
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json"
    }

    params = {}
    
    # LOGIC: SMART FILTERING
    if specific_name:
        # Sanitize input to prevent injection-like issues
        clean_name = specific_name.strip().replace("'", "")
        params["filterByFormula"] = f"{{Name}} = '{clean_name}'"
        logger.info(f"📡 API Request: Checking specific person '{clean_name}'")
    else:
        # General Scan (Limit 10 for performance)
        params["maxRecords"] = 10
        logger.info(f"📡 API Request: Scanning general roster (Limit 10)")

    try:
        # ⚠️ CRITICAL: Timeout prevents the bot from hanging if Airtable is slow
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        # Check for non-200 status codes (e.g., 401 Unauthorized, 429 Rate Limit)
        if response.status_code != 200:
            logger.error(f"❌ API Failed: Status {response.status_code} | Body: {response.text}")
            return "System Error: Unable to access the roster database at this moment."

        data = response.json()
        records = data.get("records", [])

        if not records:
            logger.info("📡 API Result: No records found.")
            return "DATABASE RESPONSE: No staff members matching that criteria were found."

        # SMART SUMMARY GENERATION
        summary = []
        for r in records:
            fields = r.get("fields", {})
            name = fields.get("Name", "Unknown Staff")
            status = fields.get("Status", "Unknown Status")
            timings = fields.get("TIMINGS", "No specific time set")
            
            summary.append(f"• {name}: {status} ({timings})")

        result_text = "LIVE DATABASE ROSTER:\n" + "\n".join(summary)
        logger.info(f"✅ API Success: Retrieved {len(records)} records.")
        return result_text

    except requests.exceptions.Timeout:
        logger.error("❌ API Error: Request timed out ( >5s ).")
        return "System Notice: The database connection timed out. Please try again."
    except requests.exceptions.ConnectionError:
        logger.error("❌ API Error: Connection refused or no internet.")
        return "System Notice: Network connection failed."
    except Exception as e:
        logger.error(f"❌ API Critical Error: {str(e)}")
        return "System Notice: An unexpected database error occurred."
