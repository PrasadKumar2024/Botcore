"""
Twilio Voice Routes - Initiates WebSocket connection
"""
import os
from fastapi import APIRouter, Response
from twilio.rest import Client
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Configuration
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+12542846845")
YOUR_PERSONAL_NUMBER = os.getenv("YOUR_PERSONAL_NUMBER", "+919938349076")
RENDER_PUBLIC_URL = os.getenv("RENDER_PUBLIC_URL", "botcore-z6j0.onrender.com")

@router.get("/test-call-me")
async def trigger_outbound_call():
    """Trigger outbound call for testing"""
    if not TWILIO_SID or not TWILIO_TOKEN:
        return {"status": "error", "message": "Twilio credentials missing"}
    
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        call = client.calls.create(
            to=YOUR_PERSONAL_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url=f"https://{RENDER_PUBLIC_URL}/twilio/voice/incoming",
            method="POST"
        )
        logger.info(f"📞 Calling {YOUR_PERSONAL_NUMBER}")
        return {
            "status": "success",
            "message": f"Calling {YOUR_PERSONAL_NUMBER}",
            "call_sid": call.sid
        }
    except Exception as e:
        logger.exception(f"Failed to initiate call: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/incoming", response_class=Response)
async def voice_incoming_webhook():
    """
    Initial webhook - Returns TwiML to connect call to WebSocket stream
    """
    ws_url = f"wss://{RENDER_PUBLIC_URL}/media-stream"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    
    logger.info(f"🔌 Connecting call to WebSocket: {ws_url}")
    return Response(content=twiml, media_type="application/xml")
