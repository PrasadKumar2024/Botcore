from fastapi import APIRouter, Form, Request, Response
import httpx
import os
from typing import Optional
import logging

router = APIRouter()

# ✅ 1. Keep your Hardcoded Client ID for testing
DEFAULT_KB_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "https://botcore-0n2z.onrender.com").rstrip("/")

# ✅ 2. FIXED Helper: Point to the CORRECT Chat API
def call_internal_chat_api_sync(client_id: str, message: str, timeout: float = 20.0) -> Optional[str]:
    if not LOCAL_API_BASE:
        logging.warning("LOCAL_API_BASE not configured")
        return None
    
    # FIX: Changed from /api/whatsapp/chat to /api/chat (matches main.py)
    url = f"{LOCAL_API_BASE}/api/chat/{client_id}"
    
    try:
        logging.info(f"🎤 Voice Bot calling Chat API: {url} with query: {message}")
        with httpx.Client(timeout=timeout) as http:
            # We must pass session_id to match the Chat API expectation
            r = http.post(url, json={"message": message, "session_id": "voice_call_test"})
            
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

# ✅ 3. Incoming Call Webhook
@router.post("/webhook", response_class=Response)
async def voice_incoming_webhook():
    # FIX: Added full path /twilio/voice/... to the action
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say voice="alice">Hello! Thanks for calling Bright Care. Please ask your question after the beep.</Say>
        <Gather input="speech" action="/twilio/voice/webhook/handle_speech" timeout="5" speechTimeout="auto">
        </Gather>
        <Say>We didn't catch that. Please call again. Goodbye.</Say>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

# ✅ 4. Handle Speech Result
@router.post("/webhook/handle_speech", response_class=Response)
async def voice_handle_speech(Request: Request,
                              SpeechResult: Optional[str] = Form(None)):
    
    user_text = (SpeechResult or "").strip()
    logging.info(f"🎤 User said: {user_text}")

    if not user_text:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say>Sorry, I didn't hear anything. Goodbye.</Say>
        </Response>
        """
        return Response(content=twiml, media_type="application/xml")

    # Call your internal RAG/chat endpoint
    answer = call_internal_chat_api_sync(DEFAULT_KB_CLIENT_ID, user_text)
    
    if not answer:
        # Fallback message
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say>I am sorry, I am having trouble accessing the records right now.</Say>
        </Response>
        """
    else:
        # Limit length because Twilio <Say> has limits and it's boring to listen to long text
        speak_text = answer
        if len(speak_text) > 800:
            speak_text = speak_text[:780] + " ... please check our website for more details."
        
        # Escape special XML characters just in case
        speak_text = speak_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say voice="alice">{speak_text}</Say>
            <Pause length="1"/>
            <Say>Do you have another question?</Say>
            <Gather input="speech" action="/twilio/voice/webhook/handle_speech" timeout="5" speechTimeout="auto">
            </Gather>
        </Response>
        """
    
    return Response(content=twiml, media_type="application/xml")
