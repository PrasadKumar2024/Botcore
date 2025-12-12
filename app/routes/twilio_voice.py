from fastapi import APIRouter, Form, Request, Response
import httpx
import os
from typing import Optional
import logging
from twilio.rest import Client
from app.database import SessionLocal
from app.services.gemini_service import GeminiService_gemini = GeminiService()

router = APIRouter()

# Configuration
DEFAULT_KB_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "https://botcore-0n2z.onrender.com").rstrip("/")

# Twilio Credentials
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+12542846845")
YOUR_PERSONAL_NUMBER = "+919938349076"

def call_internal_chat_api_sync(client_id: str, message: str, timeout: float = 45.0) -> Optional[str]:
    """Call internal chat API and return response"""
    if not LOCAL_API_BASE:
        logging.warning("LOCAL_API_BASE not configured")
        return None
    
    url = f"{LOCAL_API_BASE}/api/chat/{client_id}"
    
    try:
        logging.info(f"Voice Bot calling Chat API: {url} with query: {message}")
        with httpx.Client(timeout=timeout) as http:
            r = http.post(url, json={"message": message, "session_id": f"voice_{client_id}"})
            
            if r.status_code != 200:
                logging.warning(f"Chat API returned status {r.status_code}: {r.text}")
                return None
            
            j = r.json()
            if j.get("status") == "error":
                logging.warning(f"Chat API error: {j.get('response')}")
                return None
            
            return j.get("response")
    except Exception as e:
        logging.exception(f"Error calling internal chat API: {e}")
        return None

# Route 1: Trigger outbound call
@router.get("/test-call-me")
async def trigger_outbound_call():
    """Trigger an outbound call to your number"""
    if not TWILIO_SID or not TWILIO_TOKEN:
        return {"status": "error", "message": "Twilio credentials missing"}
    
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        call = client.calls.create(
            to=YOUR_PERSONAL_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{LOCAL_API_BASE}/twilio/voice/incoming",
            method="POST"
        )
        return {
            "status": "success",
            "message": f"Calling {YOUR_PERSONAL_NUMBER}",
            "call_sid": call.sid
        }
    except Exception as e:
        logging.exception(f"Failed to initiate call: {e}")
        return {"status": "error", "message": str(e)}

# Route 2: Initial webhook when call is answered
async def call_internal_chat_local(client_id: str, message: str) -> Optional[str]:
    """Local RAG + Gemini call (no httpx)."""
    db = SessionLocal()
    try:
        import time
        t0 = time.time()
        # import inside to avoid circular imports
        from app.services.pinecone_service import pinecone_service
        # fast semantic search (tune top_k)
        results = await pinecone_service.search_similar_chunks(client_id=str(client_id), query=message, top_k=2, min_score=0.0)
        context = "\n\n".join([r["chunk_text"] for r in results]) if results else ""
        t1 = time.time(); logging.info(f"[timing] context {round(t1-t0,2)}s")
        t2 = time.time()
        ai_response = await _gemini.generate_response(
            user_message=message,
            context=context,
            client_info={"business_name":"voice","business_type":"voice"},
            conversation_history=[]
        )
        t3 = time.time(); logging.info(f"[timing] gemini {round(t3-t2,2)}s total {round(t3-t0,2)}s")
        return ai_response
    except Exception as e:
        logging.exception("Local chat call failed: %s", e)
        return None
    finally:
        try: db.close()
        except: pass
            
@router.post("/incoming", response_class=Response)
async def voice_incoming_webhook():
    """Handle incoming call - provide initial greeting"""
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Hello! I am your BrightCare Mini Health Service assistant. You can ask me about our timings, location, or services.</Say>
    <Gather input="speech" action="/twilio/voice/process-speech" timeout="5" speechTimeout="auto">
        <Say voice="alice">Please ask your question.</Say>
    </Gather>
    <Say voice="alice">I did not hear anything. Goodbye.</Say>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# Route 3: Process speech input
@router.post("/process-speech", response_class=Response)
async def voice_handle_speech(
    Request: Request,
    SpeechResult: Optional[str] = Form(None)
):
    """Process user speech and return AI response"""
    user_text = (SpeechResult or "").strip()
    logging.info(f"User said: {user_text}")
    
    if not user_text:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Sorry, I did not catch that. Please try again.</Say>
    <Gather input="speech" action="/twilio/voice/process-speech" timeout="5" speechTimeout="auto">
        <Say voice="alice">What would you like to know?</Say>
    </Gather>
</Response>"""
        return Response(content=twiml, media_type="application/xml")
    
    # Get answer from knowledge base
    answer = await call_internal_chat_local(DEFAULT_KB_CLIENT_ID, user_text)
    
    if not answer:
        speak_text = "I apologize, but I am having trouble finding that information. Please try asking about our timings, location, or services."
    else:
        # Limit response length for voice
        speak_text = answer[:600]
    
    # Sanitize for XML
    speak_text = speak_text.replace("&", "and").replace("<", "").replace(">", "").replace('"', "")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">{speak_text}</Say>
    <Pause length="1"/>
    <Gather input="speech" action="/twilio/voice/process-speech" timeout="5" speechTimeout="auto">
        <Say voice="alice">Do you have another question?</Say>
    </Gather>
    <Say voice="alice">Thank you for calling. Goodbye.</Say>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")
