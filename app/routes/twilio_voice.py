from fastapi import APIRouter, Form, Request, Response
import httpx
import os
from typing import Optional
import logging
from twilio.rest import Client
from app.database import SessionLocal
from app.services.gemini_service import GeminiService

_gemini = GeminiService()

router = APIRouter()
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_KB_CLIENT_ID = os.getenv("DEFAULT_KB_CLIENT_ID", "9b7881dd-3215-4d1e-a533-4857ba29653c")
RENDER_PUBLIC_URL = os.getenv("RENDER_PUBLIC_URL", "botcore-0n2z.onrender.com")
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

# Route 2: Initial webhook when call is answered
async def call_internal_chat_local(client_id: str, message: str) -> Optional[str]:
    """Run local RAG + Gemini in-process (no httpx)."""
    db = SessionLocal()
    try:
        import time
        t0 = time.time()
        
        # Get context from Pinecone
        from app.services.pinecone_service import pinecone_service
        results = await pinecone_service.search_similar_chunks(
            client_id=str(client_id), 
            query=message, 
            top_k=2
        )
        context = "\n\n".join([r.get("chunk_text","") for r in results]) if results else ""
        logging.info(f"[timing] context fetch {round(time.time()-t0,2)}s")
        
        t1 = time.time()
        
        # Build prompt with context
        system_message = "You are a helpful assistant for BrightCare Mini Health Service. Answer questions based on the context provided."
        
        if context:
            full_prompt = f"Context from documents:\n{context}\n\nUser question: {message}\n\nProvide a clear, concise answer based on the context above."
        else:
            full_prompt = f"User question: {message}\n\nProvide a helpful answer about BrightCare Mini Health Service."
        
        # Call Gemini with correct parameters
        ai_response = _gemini.generate_response(
            prompt=full_prompt,
            temperature=0.7,
            max_tokens=600,  # Shorter for voice
            system_message=system_message
        )
        
        logging.info(f"[timing] gemini {round(time.time()-t1,2)}s total {round(time.time()-t0,2)}s")
        return ai_response
        
    except Exception as e:
        logging.exception("Local chat call failed: %s", e)
        return None
    finally:
        try: 
            db.close()
        except: 
            pass
            
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
