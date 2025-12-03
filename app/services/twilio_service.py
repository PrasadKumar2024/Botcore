# ownbot/app/services/twilio_service.py
"""
Twilio helpers: parse sender, send WhatsApp messages, build TwiML, call internal chat API.
Synchronous implementations so they can be used in FastAPI BackgroundTasks.
"""

import os
import re
import logging
from typing import List, Optional

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import httpx

# ENV
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. "whatsapp:+14155238886"
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE")              # e.g. "https://botcore-0n2z.onrender.com"

# lazy client
_tw_client: Optional[Client] = None


def _get_twilio_client() -> Client:
    global _tw_client
    if _tw_client is None:
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            raise RuntimeError("Twilio credentials missing: set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
        _tw_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _tw_client


def parse_sender_number(twilio_from: str) -> str:
    """
    Convert 'whatsapp:+9199...' -> '+9199...'
    """
    if not twilio_from:
        return ""
    return re.sub(r'^(whatsapp:)?', '', twilio_from).strip()


def to_whatsapp_format(number: str) -> str:
    """
    Ensure number format 'whatsapp:+<country><number>'
    """
    if not number:
        raise ValueError("Empty number")
    n = number.strip()
    if n.startswith("whatsapp:"):
        return n
    if n.startswith("+"):
        return f"whatsapp:{n}"
    return f"whatsapp:+{n}"


def send_whatsapp_message(to_number: str, body: str, media_urls: Optional[List[str]] = None) -> str:
    """
    Send WhatsApp via Twilio REST. to_number can be 'whatsapp:+91...' or '+91...'
    Returns message SID.
    """
    client = _get_twilio_client()
    to = to_number if to_number.startswith("whatsapp:") else to_whatsapp_format(to_number)
    payload = {"from_": TWILIO_WHATSAPP_FROM, "to": to, "body": body}
    if media_urls:
        payload["media_url"] = media_urls
    logging.info("Twilio send: from=%s to=%s len(body)=%d media=%s", TWILIO_WHATSAPP_FROM, to, len(body), bool(media_urls))
    msg = client.messages.create(**payload)
    logging.info("Twilio sent SID=%s", msg.sid)
    return msg.sid


def build_twiml_response(text: str) -> str:
    """
    Build TwiML XML string for immediate webhook response.
    """
    resp = MessagingResponse()
    resp.message(text)
    return str(resp)
    
def call_internal_chat_api_sync(client_id: str, message: str, timeout: float = 25.0) -> Optional[str]:
    """
    Synchronously call your WhatsApp chat endpoint /api/whatsapp/chat/{client_id}
    Expects JSON response with 'response' field.
    Returns reply string or None on failure.
    """
    if not LOCAL_API_BASE:
        logging.warning("LOCAL_API_BASE not configured; cannot call internal chat API.")
        return None
    
    # Use the new WhatsApp-specific endpoint
    url = f"{LOCAL_API_BASE.rstrip('/')}/api/whatsapp/chat/{client_id}"
    
    try:
        with httpx.Client(timeout=timeout) as http:
            r = http.post(url, json={"message": message})
            
            if r.status_code != 200:
                logging.warning("WhatsApp chat API returned status %s: %s", r.status_code, r.text)
                return None
            
            j = r.json()
            
            # Check status field
            if j.get("status") == "error":
                logging.error("WhatsApp chat API error: %s", j.get("response"))
                return None
            
            # Return the response
            return j.get("response")
            
    except Exception as e:
        logging.exception("Error calling WhatsApp chat API: %s", e)
        return None
