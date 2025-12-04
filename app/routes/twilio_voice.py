from fastapi import APIRouter, Form, Request, Response
import httpx
import os
from typing import Optional
import logging
from twilio.rest import Client # <--- Added this import

router = APIRouter()

# --- CONFIGURATION ---
DEFAULT_KB_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "https://botcore-0n2z.onrender.com").rstrip("/")

# Twilio Credentials (Loaded from your Render Environment)
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# This is the US Number you bought (Sender)
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+12542846845") 
# This is YOUR Indian Number (Receiver) - Verify this in Twilio Console!
YOUR_PERSONAL_NUMBER = "+919876543210" # <--- 🔴 CHANGE THIS TO YOUR NUMBER!

# --- HELPER FUNCTIONS ---
def call_internal_chat_api_sync(client_id: str, message: str, timeout: float = 20.0) -> Optional[str]:
    if not LOCAL_API_BASE:
        logging.warning("LOCAL_API_BASE not configured")
        return None
    
    # We use the /api/chat endpoint which you confirmed works for WhatsApp
    url = f"{LOCAL_API_BASE}/api/chat/{client_id}"
    
    try:
        logging.info(f"🎤 Voice Bot calling Chat API: {url} with query: {message}")
        with httpx.Client(timeout=timeout) as http:
            # We pass a dummy session_id to satisfy the API requirements
            r = http.post(url, json={"message": message, "session_id": "voice_test_123"})
            
            if r.status_code != 200:
                logging.warning("internal chat API status %s: %s", r.status_code, r.text)
                return None
            
            j = r.json()
            if j.get("status") == "error":
                logging.warning("internal chat API error response: %s", j.get("response"))
                return None
                
            return j.get("response")
    except Exception as e:
        logging.exception("Error calling internal chat API: %s", e)
        return None

# --- ROUTES ---

# 1. THE MAGIC LINK (Trigger the call)
@router.get("/test-call-me")
async def trigger_outbound_call():
    """
    Hit this URL to make the bot call YOU.
    """
    if not TWILIO_SID or not TWILIO_TOKEN:
        return {"status": "error", "message": "Twilio Credentials missing in Env Vars"}

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        
        # This tells Twilio: "Call Suresh, and when he answers, fetch the Webhook logic"
        call = client.calls.create(
            to=YOUR_PERSONAL_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{LOCAL_API_BASE}/twilio/voice/webhook", # The bot logic URL
            method="POST"
        )
        return {"status": "success", "message": f"Calling {YOUR_PERSONAL_NUMBER}...", "call_sid": call.sid}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 2. THE WEBHOOK (Twilio hits this when you answer)
@router.post("/webhook", response_class=Response)
async def voice_incoming_webhook():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say voice="alice">Hello! I am ready. Ask me a question about the clinic.</Say>
        <Gather input="speech" action="/twilio/voice/webhook/handle_speech" timeout="5" speechTimeout="auto">
        </Gather>
        <Say>I didn't hear anything. Goodbye.</Say>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

# 3. HANDLE SPEECH (Process your question)
@router.post("/webhook/handle_speech", response_class=Response)
async def voice_handle_speech(Request: Request, SpeechResult: Optional[str] = Form(None)):
    user_text = (SpeechResult or "").strip()
    logging.info(f"🎤 User said: {user_text}")

    if not user_text:
        twiml = """<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, no input.</Say></Response>"""
        return Response(content=twiml, media_type="application/xml")

    # Get answer from PDF
    answer = call_internal_chat_api_sync(DEFAULT_KB_CLIENT_ID, user_text)
    
    if not answer:
        speak_text = "I am sorry, I could not find that information."
    else:
        speak_text = answer[:700] # Limit length

    # Sanitize text for XML
    speak_text = speak_text.replace("&", "and").replace("<", "").replace(">", "")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say voice="alice">{speak_text}</Say>
        <Pause length="1"/>
        <Say>Ask another question.</Say>
        <Gather input="speech" action="/twilio/voice/webhook/handle_speech" timeout="5" speechTimeout="auto">
        </Gather>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")
