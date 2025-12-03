# ownbot/app/routes/twilio_webhook.py
"""
Twilio WhatsApp webhook route.
- Immediately returns a TwiML ack so Twilio doesn't timeout.
- Runs a background job that calls your /api/chat/{client_id} (RAG) and sends the final
  answer via Twilio REST to the user.
Uses the already-built knowledgebase via your /api/chat endpoint.
"""

import logging
import os
from fastapi import APIRouter, Request, BackgroundTasks, Response

from app.services.twilio_service import (
    parse_sender_number,
    build_twiml_response,
    call_internal_chat_api_sync,
    send_whatsapp_message,
)

router = APIRouter()

# Default client id to use if you want all messages to map to a fixed knowledgebase:
DEFAULT_CLIENT_ID = "9b7881dd-3215-4d1e-a533-4857ba29653c"


def _bg_process_and_reply(from_number: str, incoming_text: str):
    """
    Background sync job called by FastAPI BackgroundTasks.
    1) Resolve client_id (we use DEFAULT_CLIENT_ID as authoritative).
    2) Call internal chat API (RAG).
    3) Send final answer via Twilio REST using send_whatsapp_message.
    """
    try:
        # Use provided default client id (explicit requirement). If you prefer per-phone KB,
        # replace DEFAULT_CLIENT_ID with parse_sender_number(from_number)
        client_id = DEFAULT_CLIENT_ID

        # Call internal chat endpoint (synchronous)
        reply = call_internal_chat_api_sync(client_id=client_id, message=incoming_text)
        if not reply:
            # fallback helpful message
            reply = "Sorry, I couldn't find an answer in the knowledgebase right now."

        # Send the reply back to WhatsApp
        send_whatsapp_message(to_number=from_number, body=reply)
        logging.info("Background reply sent to %s", from_number)
    except Exception as e:
        logging.exception("Background job failed: %s", e)


@router.post("/twilio/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Entry point for Twilio webhook. Accepts x-www-form-urlencoded POST from Twilio.
    """
    form = await request.form()
    incoming_text = (form.get("Body") or "").strip()
    from_number = form.get("From") or ""  # e.g. "whatsapp:+9199..."
    logging.info("TWILIO WEBHOOK HIT from=%s body=%s", from_number, incoming_text)

    # Immediate TwiML ack so Twilio doesn't time out or mark 11200
    ack_text = "Processing your question — I'll reply shortly."
    twiml = build_twiml_response(ack_text)

    # Enqueue background job to fetch RAG answer and send via Twilio REST
    background_tasks.add_task(_bg_process_and_reply, from_number, incoming_text)

    # Return TwiML response
    return Response(content=twiml, media_type="application/xml")
