from fastapi import APIRouter, Form, Request, Response
import httpx
import os
from typing import Optional
import logging

router = APIRouter()

# Use same client id KB you gave
DEFAULT_KB_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "").rstrip("/")

# Helper: call internal chat API (synchronous HTTP call)
def call_internal_chat_api_sync(client_id: str, message: str, timeout: float = 20.0) -> Optional[str]:
    if not LOCAL_API_BASE:
        logging.warning("LOCAL_API_BASE not configured")
        return None
    url = f"{LOCAL_API_BASE}/api/whatsapp/chat/{client_id}"
    try:
        with httpx.Client(timeout=timeout) as http:
            r = http.post(url, json={"message": message})
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

# Incoming call — ask for speech (Twilio will POST speech result to /webhook/handle_speech)
@router.post("/webhook", response_class=Response)
async def voice_incoming_webhook():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Processing your question — speak after the beep. Then remain silent when finished.</Say>
  <Gather input="speech dtmf" timeout="5" speechTimeout="auto" action="/twilio/voice/webhook/handle_speech" method="POST">
    <Say>Please ask your question now.</Say>
  </Gather>
  <Say>We didn't get any input. Goodbye.</Say>
  <Hangup/>
</Response>
"""
    return Response(content=twiml, media_type="application/xml")

# Twilio posts the speech result here (SpeechResult form field)
@router.post("/webhook/handle_speech", response_class=Response)
async def voice_handle_speech(Request: Request,
                              From: Optional[str] = Form(None),
                              To: Optional[str] = Form(None),
                              CallSid: Optional[str] = Form(None),
                              SpeechResult: Optional[str] = Form(None)):
    user_text = (SpeechResult or "").strip()
    if not user_text:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, I didn't hear anything. Goodbye.</Say>
  <Hangup/>
</Response>
"""
        return Response(content=twiml, media_type="application/xml")

    # call your internal RAG/chat endpoint
    answer = call_internal_chat_api_sync(DEFAULT_KB_CLIENT_ID, user_text)

    if not answer:
        # fallback message
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, I couldn't find an answer right now. Please try again later.</Say>
  <Hangup/>
</Response>
"""
    else:
        # speak the answer
        # limit length somewhat (Twilio has limits); keep it short
        speak_text = answer
        if len(speak_text) > 800:
            speak_text = speak_text[:780] + " ... (truncated)"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{speak_text}</Say>
  <Hangup/>
</Response>
"""
    return Response(content=twiml, media_type="application/xml")
