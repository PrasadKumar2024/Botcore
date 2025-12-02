# ownbot/app/routes/twilio_webhook.py
from fastapi import APIRouter, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from app.services.twilio_service import parse_sender_number
import logging

router = APIRouter()

@router.post("/twilio/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    incoming_text = form.get("Body", "").strip()
    from_number = form.get("From", "")

    logging.info(f"Twilio webhook received from={from_number} body={incoming_text}")

    client_id = parse_sender_number(from_number)

    # delayed import to avoid circular import at startup
    try:
        from app.routes.chat import handle_chat_query
    except Exception as e:
        logging.warning("Could not import handle_chat_query (fallback to pong): %s", e)
        reply_text = "pong: " + incoming_text   # TEMP test reply
    else:
        # use real handler (await if it's async)
        maybe = handle_chat_query(client_id, incoming_text)
        if hasattr(maybe, "__await__"):
            reply_text = await maybe
        else:
            reply_text = maybe

    resp = MessagingResponse()
    resp.message(reply_text or "Sorry, I didn't understand. Try again.")
    return Response(content=str(resp), media_type="application/xml")
