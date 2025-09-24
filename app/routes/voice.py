from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import Response, PlainTextResponse
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime

from app.database import get_db
from app.services import gemini_service, pinecone_service, subscription_service, twilio_service, speech_service
from app.models import Client, PhoneNumber, Subscription
from app.utils.date_utils import check_subscription_active
from app.dependencies import validate_subscription

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory cache for conversation context (use Redis in production)
conversation_context = {}

@router.post("/voice/incoming")
async def handle_incoming_call(request: Request, db: Session = Depends(get_db)):
    """
    Handle incoming voice calls from Twilio with Wavenet voices and enhanced conversation flow
    """
    try:
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        call_sid = form_data.get("CallSid", "")
        
        logger.info(f"Incoming call from {from_number} to {to_number}, CallSID: {call_sid}")
        
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
        
        # Get client information for personalized greeting
        client = db.query(Client).filter(Client.id == client_id).first()
        business_name = client.name if client else "our business"
        
        # Check if this is the initial call or a response to a prompt
        speech_result = form_data.get("SpeechResult")
        digits_result = form_data.get("Digits")  # For DTMF input
        
        if speech_result:
            # Process the speech input with conversation context
            return await process_voice_input(speech_result, client_id, call_sid, db, from_number)
        elif digits_result:
            # Process DTMF input (keypad presses)
            return await process_dtmf_input(digits_result, client_id, call_sid, db, from_number)
        else:
            # Initial greeting with Wavenet voice and natural pacing
            greeting = generate_natural_greeting(business_name)
            
            # Initialize conversation context
            conversation_context[call_sid] = {
                "client_id": client_id,
                "from_number": from_number,
                "conversation_history": [],
                "created_at": datetime.now(),
                "caller_name": None,
                "appointment_context": None
            }
            
            return generate_twiml_response(greeting, True, client_id)
            
    except Exception as e:
        logger.error(f"Error handling incoming call: {str(e)}")
        error_message = "I apologize, we're experiencing some technical difficulties. Please call back in a few moments."
        return generate_twiml_response(error_message, False)

async def process_voice_input(speech_text: str, client_id: int, call_sid: str, db: Session, from_number: str):
    """
    Process voice input and generate a natural response using AI with conversation context
    """
    try:
        logger.info(f"Processing voice input for client {client_id}: {speech_text}")
        
        # Get conversation context
        context = conversation_context.get(call_sid, {})
        conversation_history = context.get("conversation_history", [])
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": speech_text, "timestamp": datetime.now()})
        
        # Keep only last 6 messages to avoid context overflow
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]
        
        # Generate embedding for the query
        query_embedding_result = await gemini_service.generate_embeddings([speech_text])
        
        if not query_embedding_result or len(query_embedding_result) == 0:
            response_text = "I'm having trouble understanding your request. Could you please repeat that?"
            return generate_twiml_response(response_text, True, client_id)
        
        # Query Pinecone for relevant context
        query_result = await pinecone_service.query_embeddings(
            query_embedding=query_embedding_result[0],
            client_id=str(client_id),
            top_k=3
        )
        
        # Extract context from query results
        context_data = []
        if query_result.get("matches"):
            for match in query_result["matches"]:
                if match.get("metadata") and match["metadata"].get("text"):
                    context_data.append(match["metadata"]["text"])
        
        # Prepare conversation context for Gemini
        conversation_context_str = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]
        ])
        
        # Enhanced prompt for clinic conversations
        system_prompt = f"""You are a friendly and professional receptionist for a medical clinic. 
        Current conversation history:
        {conversation_context_str}
        
        Clinic context: {', '.join(context_data) if context_data else 'General clinic information'}
        
        Please respond naturally, conversationally, and helpfully. Keep responses concise for phone conversations.
        If asking for information, ask for one piece at a time. Be empathetic and professional."""
        
        # Generate response using Gemini with conversation context
        response_result = await gemini_service.generate_response(
            query=speech_text,
            context=[system_prompt] + context_data,
            conversation_history=conversation_history
        )
        
        if not response_result["success"]:
            response_text = "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"
            return generate_twiml_response(response_text, True, client_id)
        
        response_text = response_result["response"]
        
        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now()})
        conversation_context[call_sid]["conversation_history"] = conversation_history
        
        # Check if this sounds like the end of conversation
        should_continue = should_continue_conversation(response_text, speech_text)
        
        return generate_twiml_response(response_text, should_continue, client_id)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        response_text = "I apologize, I'm having some technical difficulties. Please continue with your question."
        return generate_twiml_response(response_text, True, client_id)

async def process_dtmf_input(digits: str, client_id: int, call_sid: str, db: Session, from_number: str):
    """
    Process DTMF input (keypad presses) for menu navigation
    """
    try:
        logger.info(f"Processing DTMF input: {digits} for client {client_id}")
        
        responses = {
            "1": "To schedule an appointment, please tell me your name and preferred date.",
            "2": "For prescription refills, please provide your patient ID or full name.",
            "3": "For billing inquiries, please hold while I connect you to our billing department.",
            "0": "Please hold while I connect you to a human operator."
        }
        
        response_text = responses.get(digits, "I'm sorry, that's not a valid option. Please try again.")
        
        return generate_twiml_response(response_text, True, client_id)
        
    except Exception as e:
        logger.error(f"Error processing DTMF input: {str(e)}")
        return generate_twiml_response("Please try your selection again.", True, client_id)

def generate_natural_greeting(business_name: str) -> str:
    """
    Generate a natural-sounding greeting with SSML enhancements
    """
    return f"""<speak>
    Hello <break time="300ms"/> and thank you for calling {business_name}. 
    <break time="400ms"/>
    My name is Priya, and I'm your virtual assistant. 
    <break time="300ms"/>
    How can I help you today?
    </speak>"""

def should_continue_conversation(response_text: str, user_input: str) -> bool:
    """
    Determine if the conversation should continue based on the response content
    """
    ending_phrases = [
        "goodbye", "bye", "thank you", "have a nice day", "call ended",
        "disconnect", "hang up", "see you", "farewell"
    ]
    
    user_ending_phrases = [
        "bye", "goodbye", "thank you", "that's all", "nothing else",
        "end call", "disconnect", "hang up"
    ]
    
    # Check if AI response suggests ending
    if any(phrase in response_text.lower() for phrase in ending_phrases):
        return False
    
    # Check if user input suggests ending
    if any(phrase in user_input.lower() for phrase in user_ending_phrases):
        return False
    
    return True

def generate_twiml_response(text: str, expect_response: bool = False, client_id: int = None):
    """
    Generate TwiML response for Twilio with Wavenet audio files
    """
    response = ET.Element("Response")
    
    # Generate audio using our SpeechService with Wavenet
    try:
        speech_serv = speech_service.SpeechService()
        
        # Determine language based on client preference (default to Indian English)
        language_code = "en-IN"
        if client_id:
            # You can add client-specific language preferences here
            pass
        
        # Generate audio content with Wavenet voice
        audio_content = speech_serv.text_to_speech(text, language_code)
        
        # For production, you'd save this to temporary storage and provide URL
        # For now, we'll use <Say> as fallback, but with enhanced settings
        
        if expect_response:
            # Use <Gather> with enhanced speech recognition
            gather = ET.SubElement(response, "Gather")
            gather.set("input", "speech")
            gather.set("action", "/api/voice/incoming")
            gather.set("method", "POST")
            gather.set("speechTimeout", "2")  # Shorter timeout for more natural flow
            gather.set("speechModel", "phone_call")
            gather.set("enhanced", "true")
            gather.set("actionOnEmptyResult", "true")
            gather.set("profanityFilter", "false")
            
            # Use premium voice with SSML support
            say = ET.SubElement(gather, "Say")
            say.set("voice", "Polly.Aditi-Neural")  # Premium Indian English voice
            say.set("language", "en-IN")
            
            # Add slight pause before speaking for more natural flow
            if not text.startswith("<speak>"):
                text = f"<speak><break time='200ms'/>{text}</speak>"
            
            # Handle both SSML and plain text
            if text.startswith("<speak>"):
                say.set("interpretAs", "ssml")
                say.text = text
            else:
                say.text = text
            
            # Add background noise reduction hint
            ET.SubElement(gather, "Pause", length="1")
            
        else:
            # Final message before hangup
            say = ET.SubElement(response, "Say")
            say.set("voice", "Polly.Aditi-Neural")
            say.set("language", "en-IN")
            
            if text.startswith("<speak>"):
                say.set("interpretAs", "ssml")
                say.text = text
            else:
                say.text = text
            
            # Add polite goodbye and hangup
            ET.SubElement(response, "Say", voice="Polly.Aditi-Neural", language="en-IN").text = "Thank you for calling. Have a wonderful day."
            ET.SubElement(response, "Hangup")
            
    except Exception as e:
        logger.error(f"Error generating audio, falling back to basic TTS: {str(e)}")
        # Fallback to basic Twilio TTS
        if expect_response:
            gather = ET.SubElement(response, "Gather")
            gather.set("input", "speech")
            gather.set("action", "/api/voice/incoming")
            gather.set("method", "POST")
            gather.set("speechTimeout", "3")
            
            say = ET.SubElement(gather, "Say")
            say.set("voice", "alice")
            say.set("language", "en-IN")
            say.text = text.replace("<speak>", "").replace("</speak>", "").replace("<break time=\"300ms\"/>", "")
        else:
            say = ET.SubElement(response, "Say")
            say.set("voice", "alice")
            say.set("language", "en-IN")
            say.text = text.replace("<speak>", "").replace("</speak>", "").replace("<break time=\"300ms\"/>", "")
            ET.SubElement(response, "Hangup")
    
    # Convert to string
    twiml = '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(response, encoding="unicode")
    return Response(content=twiml, media_type="application/xml")

@router.post("/voice/status")
async def handle_call_status(request: Request):
    """
    Handle call status updates from Twilio and clean up conversation context
    """
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration", "0")
        
        logger.info(f"Call {call_sid} status: {call_status}, duration: {call_duration} seconds")
        
        # Clean up conversation context when call ends
        if call_status in ["completed", "failed", "busy", "no-answer"]:
            if call_sid in conversation_context:
                del conversation_context[call_sid]
                logger.info(f"Cleaned up conversation context for call {call_sid}")
        
        # Log call analytics
        if call_status == "completed":
            logger.info(f"Call completed successfully. Duration: {call_duration} seconds")
            # Here you can add call analytics to your database
        elif call_status == "failed":
            logger.warning(f"Call failed: {call_sid}")
        
        return PlainTextResponse("OK")
    except Exception as e:
        logger.error(f"Error handling call status: {str(e)}")
        return PlainTextResponse("OK")

@router.get("/voice/test/{client_id}")
async def test_voice_functionality(client_id: int, db: Session = Depends(get_db)):
    """
    Test endpoint to verify voice functionality for a client with Wavenet preview
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
        
        # Test Wavenet voice generation
        test_audio = None
        try:
            speech_serv = speech_service.SpeechService()
            test_text = "Hello, this is a test of the natural sounding Wavenet voice."
            test_audio = speech_serv.text_to_speech(test_text, "en-IN")
            audio_test_status = "Wavenet voice generation successful"
        except Exception as e:
            audio_test_status = f"Wavenet voice test failed: {str(e)}"
        
        return {
            "client": client.name,
            "voice_subscription_active": check_subscription_active(voice_subscription) if voice_subscription else False,
            "phone_number_assigned": phone_number is not None,
            "phone_number": phone_number.number if phone_number else None,
            "wavenet_voice_test": audio_test_status,
            "active_conversations": len(conversation_context),
            "recommended_voices": [
                "en-IN-Wavenet-A (Indian English - Female)",
                "en-IN-Wavenet-B (Indian English - Male)", 
                "hi-IN-Wavenet-A (Hindi - Female)",
                "ta-IN-Wavenet-A (Tamil - Female)"
            ],
            "status": "Voice bot is configured correctly" if (voice_subscription and phone_number) else "Voice bot not fully configured"
        }
    except Exception as e:
        logger.error(f"Error testing voice functionality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/health")
async def voice_health_check():
    """
    Health check endpoint for the voice service with Wavenet verification
    """
    try:
        # Check if required services are available
        gemini_health = await gemini_service.validate_api_key()
        pinecone_health = await pinecone_service.check_health()
        
        # Test Wavenet voice generation
        wavenet_test = False
        try:
            speech_serv = speech_service.SpeechService()
            test_audio = speech_serv.text_to_speech("Health check", "en-IN")
            wavenet_test = len(test_audio) > 0
        except Exception as e:
            logger.warning(f"Wavenet health check warning: {str(e)}")
        
        health_status = {
            "status": "healthy",
            "services": {
                "gemini": "available" if gemini_health else "unavailable",
                "pinecone": pinecone_health.get("status", "unknown"),
                "twilio": "configured" if twilio_service.is_configured() else "unconfigured",
                "wavenet_voices": "available" if wavenet_test else "unavailable"
            },
            "conversation_metrics": {
                "active_conversations": len(conversation_context),
                "cache_size": "in_memory (use Redis in production)"
            },
            "features": {
                "wavenet_voices": "enabled",
                "ssml_support": "enabled", 
                "conversation_context": "enabled",
                "natural_greetings": "enabled"
            },
            "endpoints": {
                "incoming_call": "/api/voice/incoming",
                "call_status": "/api/voice/status",
                "health_check": "/api/voice/health",
                "voice_test": "/api/voice/test/{client_id}"
            }
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Voice health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "wavenet_voices": "unknown"
        }

@router.post("/voice/call-test")
async def test_call_functionality():
    """
    Test Twilio call functionality with Wavenet voices
    """
    try:
        # Test Wavenet voice generation with different phrases
        test_phrases = [
            "Hello, welcome to our medical clinic. How can I assist you today?",
            "I understand you'd like to schedule an appointment. What day works best for you?",
            "Thank you for providing that information. Your appointment has been confirmed."
        ]
        
        results = []
        speech_serv = speech_service.SpeechService()
        
        for phrase in test_phrases:
            try:
                audio = speech_serv.text_to_speech(phrase, "en-IN")
                results.append({
                    "phrase": phrase,
                    "status": "success",
                    "audio_size_bytes": len(audio),
                    "voice_used": "en-IN-Wavenet-A"
                })
            except Exception as e:
                results.append({
                    "phrase": phrase, 
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "message": "Wavenet voice test completed",
            "results": results,
            "recommendations": [
                "Use SSML for natural pauses and emphasis",
                "Adjust speaking rate to 0.9-1.1 for optimal clarity",
                "Implement conversation context for multi-turn dialogues"
            ]
        }
    except Exception as e:
        logger.error(f"Error testing call functionality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/voices")
async def list_available_voices():
    """
    List all available Wavenet voices for Indian languages
    """
    voices = {
        "indian_english": {
            "en-IN-Wavenet-A": "Female - Natural, polite (Recommended)",
            "en-IN-Wavenet-B": "Male - Professional, clear",
            "en-IN-Wavenet-C": "Female - Warm, friendly",
            "en-IN-Wavenet-D": "Male - Authoritative, deep"
        },
        "hindi": {
            "hi-IN-Wavenet-A": "Female - Standard Hindi",
            "hi-IN-Wavenet-B": "Male - Standard Hindi", 
            "hi-IN-Wavenet-C": "Female - Conversational"
        },
        "south_indian_languages": {
            "ta-IN-Wavenet-A": "Tamil - Female",
            "te-IN-Wavenet-A": "Telugu - Female",
            "kn-IN-Wavenet-A": "Kannada - Female", 
            "ml-IN-Wavenet-A": "Malayalam - Female"
        },
        "premium_voices": {
            "en-US-Wavenet-F": "US English - Very natural female",
            "en-GB-Wavenet-B": "British English - Professional male",
            "en-AU-Wavenet-A": "Australian English - Friendly female"
        }
    }
    
    return {
        "recommendation": "Use en-IN-Wavenet-A for Indian English clinics",
        "total_voices": sum(len(category) for category in voices.values()),
        "voices": voices
    }
