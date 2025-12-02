# ownbot/app/routes/twilio_webhook.py
from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from app.services.twilio_service import parse_sender_number
from app.routes.chat import handle_chat_query  # reuse your existing chat logic (adjust import if different)
import logging

router = APIRouter()

@router.post("/twilio/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """
    Twilio sends x-www-form-urlencoded POSTs.
    We will parse 'Body' and 'From', call your chat handler, and return TwiML.
    """
    form = await request.form()
    incoming_text = form.get("Body", "").strip()
    from_number = form.get("From", "")  # e.g. "whatsapp:+1415XXXXXXX"

    logging.info(f"Twilio webhook received from={from_number} body={incoming_text}")

    # Convert Twilio sender to normalized client id if you have a helper
    client_id = parse_sender_number(from_number)

    # Call your existing chat logic — MUST return text (string). Adapt as needed.
    # Example: handle_chat_query(client_id, incoming_text) -> reply_text
    reply_text = await handle_chat_query(client_id, incoming_text)

    # Build TwiML response
    resp = MessagingResponse()
    resp.message(reply_text or "Sorry, I didn't understand. Try again or type HELP.")

    xml = str(resp)
    return Response(content=xml, media_type="application/xml")
