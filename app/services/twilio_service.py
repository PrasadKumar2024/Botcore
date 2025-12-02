"""
ownbot/app/services/twilio_service.py

Full helper service for Twilio WhatsApp integration.
Adapt the `get_answer_from_kb` call to point to your actual knowledge-base/chat function.
"""

import os
import logging
import re
from typing import List, Optional

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# Optional: validator (used for verifying Twilio signature)
try:
    from twilio.request_validator import RequestValidator
except Exception:
    RequestValidator = None  # not available, skip validation if not installed

# Environment variables (set on Render)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# For sandbox use: 'whatsapp:+14155238886' typically; replace when you have prod number.
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# Initialize Twilio REST client lazily
_twilio_client = None


def _get_twilio_client():
    global _twilio_client
    if _twilio_client is None:
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            raise RuntimeError("Twilio credentials not set (TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN).")
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


# ----------------------------
# Utility helpers
# ----------------------------
def parse_sender_number(twilio_from: str) -> str:
    """
    Normalize Twilio 'From' value (e.g. 'whatsapp:+9199...') -> '+9199...'
    Use this as client_id lookup key.
    """
    if not twilio_from:
        return ""
    return re.sub(r'^(whatsapp:)?', '', twilio_from).strip()


def to_whatsapp_format(phone: str) -> str:
    """
    Ensure the number to send has 'whatsapp:' prefix for Twilio REST .create call.
    Accepts '+9199...' or 'whatsapp:+9199...' and returns 'whatsapp:+9199...'
    """
    if not phone:
        raise ValueError("phone is empty")
    phone = phone.strip()
    if phone.startswith("whatsapp:"):
        return phone
    if phone.startswith("+"):
        return f"whatsapp:{phone}"
    # If user stored without +, try to be helpful (not ideal for prod)
    return f"whatsapp:+{phone}"


# ----------------------------
# Sending helpers
# ----------------------------
def send_whatsapp_message(to_number: str, body: str, media_urls: Optional[List[str]] = None) -> str:
    """
    Send a WhatsApp message using Twilio REST API.
    - to_number: '+9199...' or 'whatsapp:+9199...'
    - body: text message
    - media_urls: list of absolute URLs (images/pdf) to attach (optional)
    Returns message SID.
    """
    client = _get_twilio_client()
    to = to_number if to_number.startswith("whatsapp:") else to_whatsapp_format(to_number)
    payload = {
        "from_": TWILIO_WHATSAPP_NUMBER,
        "to": to,
        "body": body
    }
    if media_urls:
        payload["media_url"] = media_urls

    logging.info(f"Sending WhatsApp msg from={TWILIO_WHATSAPP_NUMBER} to={to} len_body={len(body)} media={bool(media_urls)}")
    msg = client.messages.create(**payload)
    logging.info(f"Sent message SID={msg.sid}")
    return msg.sid


# ----------------------------
# TwiML response helper (for webhooks)
# ----------------------------
def build_twiml_response(text: str) -> str:
    """
    Return TwiML XML string to use as webhook response.
    Use this in your FastAPI/Flask endpoint as:
        return Response(content=build_twiml_response(answer), media_type="application/xml")
    """
    resp = MessagingResponse()
    resp.message(text)
    return str(resp)


# ----------------------------
# Twilio request validation (optional but recommended)
# ----------------------------
def verify_twilio_request(url: str, params: dict, headers: dict) -> bool:
    """
    Validate incoming request signature from Twilio.
    - url: the full public URL for the request (must match what Twilio used)
    - params: dictionary of POST params (form data)
    - headers: request headers mapping (expects 'X-Twilio-Signature')
    Returns True if valid. If RequestValidator not installed, returns True (for dev).
    """
    if RequestValidator is None:
        logging.warning("twilio.request_validator not available; skipping validation (install twilio package >=6.0).")
        return True

    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    signature = headers.get("X-Twilio-Signature", "")
    # Validator expects the full URL and params (for POST, pass form data dict)
    try:
        valid = validator.validate(url, params, signature)
        if not valid:
            logging.warning("Twilio request validation failed.")
        return bool(valid)
    except Exception as e:
        logging.exception("Error validating Twilio request: %s", e)
        return False


# ----------------------------
# High-level inbound handling helper
# ----------------------------
def get_answer_from_kb_stub(query: str, client_id: str) -> str:
    """
    STUB: Replace this with a call to your actual KB/chat function.
    Example replacements:
      - from app.services.document_service import find_best_answer
      - from app.services.chat_service import get_chatbot_reply
    Your real function should take (query, client_id) and return a short reply string.
    """
    # >>> REPLACE THIS CODE <<<
    # return search_pdf_knowledge(query, client_id)
    return "This is a placeholder reply. Replace get_answer_from_kb_stub with your KB call."


def handle_inbound_and_get_reply(form: dict) -> dict:
    """
    High-level helper meant to be called from your webhook.
    `form` should be the parsed form data from Twilio request (e.g. request.form() result).
    Returns a dict: { 'to': 'whatsapp:+...', 'from': 'whatsapp:+1415...', 'body': 'reply text' }
    """
    incoming_text = form.get("Body", "").strip() if form else ""
    twilio_from = form.get("From", "") if form else ""
    twilio_to = form.get("To", "") if form else TWILIO_WHATSAPP_NUMBER

    client_id = parse_sender_number(twilio_from)

    # Call your KB/chat function here (replace the stub)
    try:
        # Replace get_answer_from_kb_stub with your actual function
        reply = get_answer_from_kb_stub(incoming_text, client_id)
    except Exception as e:
        logging.exception("Error running KB search or chat function: %s", e)
        reply = "Sorry, something went wrong while fetching your answer. Please try again."

    # short-circuit: if reply is long, shorten and prompt
    if reply and len(reply) > 700:
        short = reply[:700].rsplit("\n", 1)[0]
        short += "\n\nReply MORE for the rest."
        reply = short

    result = {
        "to": twilio_from,     # Twilio expects "whatsapp:+91..." format when using REST send
        "from": twilio_to,     # e.g. TWILIO_WHATSAPP_NUMBER
        "body": reply
    }
    return result


# ----------------------------
# Example convenience function to both build TwiML and/or send async
# ----------------------------
def respond_via_twiml(reply_text: str) -> str:
    """
    Return TwiML string to immediately reply to webhook (recommended for simple flows).
    """
    return build_twiml_response(reply_text)


def respond_via_rest_send(to_number: str, body: str, media_urls: Optional[List[str]] = None) -> str:
    """
    Use Twilio REST to send a message asynchronously (useful for multi-part flows).
    Returns message SID.
    """
    return send_whatsapp_message(to_number, body, media_urls)


# ----------------------------
# If needed, you can expose a small CLI/test helper
# ----------------------------
if __name__ == "__main__":
    print("twilio_service module loaded. This file is a helper module — import into your app.")
