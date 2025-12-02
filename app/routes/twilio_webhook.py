# ownbot/app/routes/twilio_webhook.py
import os
import logging
from fastapi import APIRouter, Request, BackgroundTasks, Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
import httpx
from app.services.twilio_service import parse_sender_number

router = APIRouter()

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. "whatsapp:+1415..."
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE")  # e.g. https://botcore-0n2z.onrender.com

twilio_client = None
if TWILIO_SID and TWILIO_TOKEN:
    try:
        twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
    except Exception as e:
        logging.error("Twilio client init failed: %s", e)

async def _bg_handle_and_reply(from_number: str, incoming_text: str):
    """
    Background job:
      1) call internal chat API: /api/chat/{client_id}
      2) send reply back via Twilio REST API (WhatsApp)
    """
    try:
        client_id = parse_sender_number(from_number)
    except Exception:
        client_id = None

    # call internal chat endpoint
    reply_text = None
    try:
        if LOCAL_API_BASE and client_id:
            url = f"{LOCAL_API_BASE}/api/chat/{client_id}"
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(url, json={"message": incoming_text})
            if resp.status_code == 200:
                data = resp.json()
                reply_text = data.get("response") or data.get("answer") or data.get("message")
        else:
            logging.warning("LOCAL_API_BASE or client_id missing; skipping internal chat call.")
    except Exception as e:
        logging.exception("Error calling internal chat API: %s", e)

    # fallback simple echo if no reply_text
    if not reply_text:
        reply_text = f"pong: {incoming_text}"

    # send via Twilio REST
    try:
        if twilio_client and TWILIO_WHATSAPP_FROM:
            twilio_client.messages.create(
                body=reply_text,
                from_=TWILIO_WHATSAPP_FROM,
                to=from_number
            )
            logging.info("Sent WhatsApp reply to %s", from_number)
        else:
            logging.warning("Twilio client or TWILIO_WHATSAPP_FROM not configured.")
    except Exception as e:
        logging.exception("Failed sending Twilio WhatsApp message: %s", e)


@router.post("/twilio/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    # Twilio posts form-data x-www-form-urlencoded
    form = await request.form()
    incoming_text = (form.get("Body") or "").strip()
    from_number = form.get("From") or ""  # "whatsapp:+91..."

    logging.info("Webhook received from=%s body=%s", from_number, incoming_text)

    # Respond immediately so Twilio sees 200 (prevents retries/timeouts)
    resp = MessagingResponse()
    resp.message("Processing your request...")  # optional immediate ack

    # Start background job that will call chat API and send final reply
    background_tasks.add_task(_bg_handle_and_reply, from_number, incoming_text)

    return Response(content=str(resp), media_type="application/xml")
