from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import Response, PlainTextResponse
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import xml.etree.ElementTree as ET

from app.database import get_db
from app.services import gemini_service, pinecone_service, subscription_service, twilio_service
from app.models import Client, PhoneNumber, Subscription
from app.utils.date_utils import check_subscription_active
from app.dependencies import validate_subscription

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/voice/incoming")
async def handle_incoming_call(request: Request, db: Session = Depends(get_db)):
    """
    Handle incoming voice calls from Twilio
    """
    try:
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        
        logger.info(f"Incoming call from {from_number} to {to_number}")
        
        # Find the client associated with this phone number
        phone_number = db.query(PhoneNumber).filter(PhoneNumber.number == to_number).first()
        if not phone_number:
            logger.error(f"No client found for number: {to_number}")
            return generate_twiml_response("Sorry, this number is not associated with any service.")
        
        client_id = phone_number.client_id
        
        # Check if voice subscription is active using the dependency
        try:
            subscription = await validate_subscription(client_id, "voice", db)
        except HTTPException:
            return generate_twiml_response("Sorry, the voice service is not currently active. Please contact the business owner.")
        
        # Check if this is the initial call or a response to a prompt
        speech_result = form_data.get("SpeechResult")
        call_sid = form_data.get("CallSid")
        
        if speech_result:
            # Process the speech input
            return await process_voice_input(speech_result, client_id, call_sid, db)
        else:
            # Initial greeting
            client = db.query(Client).filter(Client.id == client_id).first()
            business_name = client.name if client else "our business"
            greeting = f"Hello, thank you for calling {business_name}. How can I help you today?"
            return generate_twiml_response(greeting, True)
            
    except Exception as e:
        logger.error(f"Error handling incoming call: {str(e)}")
        return generate_twiml_response("Sorry, we're experiencing technical difficulties. Please try again later.")

async def process_voice_input(speech_text: str, client_id: int, call_sid: str, db: Session):
    """
    Process voice input and generate a response using AI
    """
    try:
        logger.info(f"Processing voice input for client {client_id}: {speech_text}")
        
        # Generate embedding for the query
        query_embedding_result = await gemini_service.generate_embeddings([speech_text])
        
        if not query_embedding_result or len(query_embedding_result) == 0:
            return generate_twiml_response("I'm having trouble understanding your request. Please try again.", True)
        
        # Query Pinecone for relevant context
        query_result = await pinecone_service.query_embeddings(
            query_embedding=query_embedding_result[0],
            client_id=str(client_id),
            top_k=3
        )
        
        # Extract context from query results
        context = []
        if query_result.get("matches"):
            for match in query_result["matches"]:
                if match.get("metadata") and match["metadata"].get("text"):
                    context.append(match["metadata"]["text"])
        
        # Generate response using Gemini
        response_result = await gemini_service.generate_response(
            query=speech_text,
            context=context
        )
        
        if not response_result["success"]:
            return generate_twiml_response("I'm sorry, I couldn't process your request at the moment. Please try again.", True)
        
        # Return the response with option to continue
        return generate_twiml_response(response_result["response"], True)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        return generate_twiml_response("I'm having trouble processing your request. Please try again.", True)

def generate_twiml_response(text: str, expect_response: bool = False):
    """
    Generate TwiML response for Twilio
    """
    response = ET.Element("Response")
    
    if expect_response:
        # Use <Gather> to collect speech input with enhanced settings
        gather = ET.SubElement(response, "Gather")
        gather.set("input", "speech")
        gather.set("action", "/api/voice/incoming")
        gather.set("method", "POST")
        gather.set("speechTimeout", "3")
        gather.set("speechModel", "phone_call")
        gather.set("enhanced", "true")
        gather.set("actionOnEmptyResult", "true")
        
        say = ET.SubElement(gather, "Say")
        say.set("voice", "alice")
        say.set("language", "en-US")
        say.text = text
        
        # Add a pause to allow user to respond
        ET.SubElement(gather, "Pause", length="1")
        
    else:
        # Just say the text and hang up
        say = ET.SubElement(response, "Say")
        say.set("voice", "alice")
        say.set("language", "en-US")
        say.text = text
        ET.SubElement(response, "Hangup")
    
    # Convert to string
    twiml = '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(response, encoding="unicode")
    return Response(content=twiml, media_type="application/xml")

@router.post("/voice/status")
async def handle_call_status(request: Request):
    """
    Handle call status updates from Twilio
    """
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration", "0")
        
        logger.info(f"Call {call_sid} status: {call_status}, duration: {call_duration} seconds")
        
        # You can log call analytics here
        if call_status == "completed":
            logger.info(f"Call completed successfully. Duration: {call_duration} seconds")
        elif call_status == "failed":
            logger.warning(f"Call failed: {call_sid}")
        
        return PlainTextResponse("OK")
    except Exception as e:
        logger.error(f"Error handling call status: {str(e)}")
        return PlainTextResponse("OK")

@router.get("/voice/test/{client_id}")
async def test_voice_functionality(client_id: int, db: Session = Depends(get_db)):
    """
    Test endpoint to verify voice functionality for a client
    """
    try:
        # Check if client exists
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Check voice subscription
        voice_subscription = db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.bot_type == "voice"
        ).first()
        
        # Check if phone number is assigned
        phone_number = db.query(PhoneNumber).filter(PhoneNumber.client_id == client_id).first()
        
        return {
            "client": client.name,
            "voice_subscription_active": check_subscription_active(voice_subscription) if voice_subscription else False,
            "phone_number_assigned": phone_number is not None,
            "phone_number": phone_number.number if phone_number else None,
            "status": "Voice bot is configured correctly" if (voice_subscription and phone_number) else "Voice bot not fully configured"
        }
    except Exception as e:
        logger.error(f"Error testing voice functionality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/health")
async def voice_health_check():
    """
    Health check endpoint for the voice service
    """
    try:
        # Check if required services are available
        gemini_health = await gemini_service.validate_api_key()
        pinecone_health = await pinecone_service.check_health()
        
        health_status = {
            "status": "healthy",
            "services": {
                "gemini": "available" if gemini_health else "unavailable",
                "pinecone": pinecone_health.get("status", "unknown"),
                "twilio": "configured" if twilio_service.is_configured() else "unconfigured"
            },
            "endpoints": {
                "incoming_call": "/api/voice/incoming",
                "call_status": "/api/voice/status",
                "health_check": "/api/voice/health"
            }
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Voice health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/voice/call-test")
async def test_call_functionality():
    """
    Test Twilio call functionality (optional - for manual testing)
    """
    try:
        # This would initiate a test call - implement if needed
        return {"message": "Call test functionality would be implemented here"}
    except Exception as e:
        logger.error(f"Error testing call functionality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
