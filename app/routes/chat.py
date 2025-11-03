from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging
import uuid
from datetime import datetime, timedelta
from app.database import get_db
from app.services import gemini_service, pinecone_service #subscription_service
from app.models import Client, Subscription
from app.schemas import ChatRequest, ChatResponse
from app.utils.date_utils import check_subscription_active

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Handle chat messages from the web chat widget.
    Returns AI responses based on the client's PDF knowledge base.
    """
    try:
        # Validate client exists
        client = db.query(Client).filter(Client.id == chat_request.client_id).first()
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found"
            )
        
        # Check if web chat subscription is active
        web_subscription = db.query(Subscription).filter(
            Subscription.client_id == chat_request.client_id,
            Subscription.bot_type == "web"
        ).first()
        
        if not web_subscription or not check_subscription_active(web_subscription):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Web chat subscription is not active"
            )
        
        # Generate embedding for the query
        query_embedding = await gemini_service.generate_embeddings([chat_request.message])
        
        if not query_embedding or len(query_embedding) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embedding"
            )
        
        # Query Pinecone for relevant context
        query_result = await pinecone_service.query_embeddings(
            query_embedding=query_embedding[0],
            client_id=str(chat_request.client_id),
            top_k=5
        )
        
        if not query_result["success"]:
            logger.error(f"Pinecone query failed: {query_result.get('error')}")
            # Continue without context rather than failing completely
        
        # Extract context from query results
        context = []
        if query_result.get("matches"):
            for match in query_result["matches"]:
                if "metadata" in match and "text" in match["metadata"]:
                    context.append(match["metadata"]["text"])
        
        # Generate response using Gemini
        response_result = await gemini_service.generate_response(
            query=chat_request.message,
            context=context,
            conversation_history=chat_request.conversation_history
        )
        
        if not response_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate response: {response_result.get('error')}"
            )
        
        # Return the successful response
        return ChatResponse(
            success=True,
            response=response_result["response"],
            conversation_id=chat_request.conversation_id or generate_conversation_id(),
            client_id=chat_request.client_id
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your message"
        )

def generate_conversation_id() -> str:
    """Generate a unique conversation ID"""
    import uuid
    return str(uuid.uuid4())

# ADD THESE 3 ROUTES TO YOUR EXISTING chat.py

@router.post("/clients/{client_id}/web_chat/activate")
async def activate_client_web_chat(client_id: int, db: Session = Depends(get_db)):
    """Activate web chat and generate embed codes"""
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Generate embed code and URL if first time
        if not client.embed_code:
            unique_id = f"{client.business_name.lower().replace(' ', '-')}-{str(uuid.uuid4().hex[:8])}"
            embed_code = f'''
            <div id="chatbot-container-{unique_id}"></div>
            <script>
                (function() {{
                    var script = document.createElement('script');
                    script.src = 'https://botcore-0n2z.onrender.com/static/js/chat-widget.js?client_id={unique_id}';
                    script.async = true;
                    document.head.appendChild(script);
                }})();
            </script>
            '''
            client.embed_code = embed_code.strip()
            client.chatbot_url = f"https://botcore-0n2z.onrender.com/chat/{unique_id}"
            client.unique_id = unique_id
        
        client.web_chat_active = True
        client.web_chat_start_date = client.web_chat_start_date or datetime.utcnow()
        db.commit()
        
        return {"message": "Web chat activated successfully", "client_id": client_id}
        
    except Exception as e:
        logger.error(f"Error activating web chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate web chat")

@router.post("/clients/{client_id}/web_chat/deactivate")
async def deactivate_client_web_chat(client_id: int, db: Session = Depends(get_db)):
    """Deactivate web chat"""
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        client.web_chat_active = False
        db.commit()
        
        return {"message": "Web chat deactivated successfully", "client_id": client_id}
        
    except Exception as e:
        logger.error(f"Error deactivating web chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deactivate web chat")

@router.post("/clients/{client_id}/web_chat/subscription")
async def add_web_chat_subscription(client_id: int, months: int, db: Session = Depends(get_db)):
    """Add subscription time to web chat"""
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        current_time = datetime.utcnow()
        
        # If no expiry date or expired, set from current time
        if not client.web_chat_expiry_date or client.web_chat_expiry_date < current_time:
            client.web_chat_expiry_date = current_time + timedelta(days=30 * months)
        else:
            # Extend existing subscription
            client.web_chat_expiry_date = client.web_chat_expiry_date + timedelta(days=30 * months)
        
        # Ensure active when adding subscription
        client.web_chat_active = True
        db.commit()
        
        return {
            "message": f"Added {months} month(s) to web chat subscription",
            "new_expiry_date": client.web_chat_expiry_date.isoformat(),
            "client_id": client_id
        }
        
    except Exception as e:
        logger.error(f"Error adding subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add subscription")

@router.get("/api/chat/health")
async def health_check():
    """Health check endpoint for the chat service"""
    try:
        # Check if required services are available
        gemini_health = await gemini_service.validate_api_key()
        pinecone_health = await pinecone_service.check_health()
        
        return {
            "status": "healthy",
            "services": {
                "gemini": "available" if gemini_health else "unavailable",
                "pinecone": pinecone_health.get("status", "unknown")
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )

@router.post("/api/clients/{client_id}/generate-embed-code")
async def generate_embed_code(client_id: str, db: Session = Depends(get_db)):
    """Generate embed code for web chat widget"""
    try:
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Generate unique ID if not exists
        if not client.unique_id:
            unique_id = f"{client.business_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
            client.unique_id = unique_id
        else:
            unique_id = client.unique_id
        
        # Your actual Render domain
        BASE_URL = "https://botcore-z6j0.onrender.com"
        
        # Generate embed code
        embed_code = f'''<div id="chatbot-container-{unique_id}"></div>
<script>
    (function() {{
        var script = document.createElement('script');
        script.src = '{BASE_URL}/static/js/chat-widget.js?client_id={client_id}';
        script.async = true;
        document.head.appendChild(script);
    }})();
</script>'''
        
        # Generate chatbot URL
        chatbot_url = f"{BASE_URL}/chat/{unique_id}"
        
        # Save to database
        client.embed_code = embed_code
        client.chatbot_url = chatbot_url
        db.commit()
        db.refresh(client)
        
        return {
            "success": True,
            "embed_code": embed_code,
            "chatbot_url": chatbot_url
        }
        
    except Exception as e:
        logger.error(f"Error generating embed code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
