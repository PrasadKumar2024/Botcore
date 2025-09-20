from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging

from app.database import get_db
from app.services import gemini_service, pinecone_service, subscription_service
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
