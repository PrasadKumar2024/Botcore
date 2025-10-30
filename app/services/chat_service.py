# app/services/chat_service.py
import os
import logging
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from sqlalchemy.orm import Session
import google.generativeai as genai

from app.models import Client, MessageLog, WebchatSession, KnowledgeChunk
from app.services.gemini_service import gemini_service
from app.services.pinecone_service import pinecone_service
from app.database import get_db

logger = logging.getLogger(__name__)

class ChatService:
    """
    Comprehensive chat service for handling embedded chatbot functionality
    Integrates Gemini AI, Pinecone RAG, and conversation management
    """
    
    def __init__(self):
        self.max_context_length = 4000  # Characters for context window
        self.max_conversation_history = 10  # Last N messages to remember
        self.default_welcome_message = "Hello! I'm an AI assistant. How can I help you today?"
    
    async def process_chat_message(
        self,
        client_id: str,
        message: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate AI response with RAG
        
        Args:
            client_id: Unique client identifier
            message: User's message text
            session_id: Optional session ID for conversation tracking
            conversation_history: Previous messages in conversation
            db: Database session
            
        Returns:
            Dictionary with response data
        """
        try:
            logger.info(f"Processing chat message for client {client_id}: {message[:100]}...")
            
            # Validate client exists and web chat is active
            client = db.query(Client).filter(Client.id == client_id).first()
            if not client:
                return self._error_response("Client not found")
            
            if not client.web_chat_active:
                return self._error_response("Web chat is not active for this client")
            
            # Check subscription expiry
            if (client.web_chat_expiry_date and 
                client.web_chat_expiry_date < datetime.utcnow()):
                return self._error_response("Web chat subscription has expired")
            
            # Generate or validate session
            session_id = await self._manage_session(client_id, session_id, db)
            
            # Generate embedding for query
            query_embedding = await gemini_service.generate_embedding_async(message)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return await self._fallback_response(message, client.business_name)
            
            # Search for relevant context in Pinecone
            context_chunks = await self._search_relevant_context(
                query_embedding, client_id, top_k=5
            )
            
            # Build RAG prompt with context
            rag_prompt = self._build_rag_prompt(
                message=message,
                context_chunks=context_chunks,
                business_name=client.business_name,
                conversation_history=conversation_history
            )
            
            # Generate AI response
            ai_response = await self._generate_ai_response(rag_prompt, client.business_name)
            
            # Log the interaction
            await self._log_interaction(
                db=db,
                client_id=client_id,
                session_id=session_id,
                user_message=message,
                ai_response=ai_response,
                context_used=context_chunks
            )
            
            # Prepare response
            return {
                "success": True,
                "response": ai_response,
                "session_id": session_id,
                "conversation_id": session_id,  # For compatibility
                "client_id": client_id,
                "sources": [chunk.get("source", "Knowledge Base") for chunk in context_chunks[:3]],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return await self._fallback_response(message, "our business")
    
    async def _search_relevant_context(
        self, 
        query_embedding: List[float], 
        client_id: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search Pinecone for relevant context chunks
        
        Args:
            query_embedding: Embedding vector of the query
            client_id: Client ID for filtering
            top_k: Number of top results to return
            
        Returns:
            List of relevant context chunks with metadata
        """
        try:
            # Query Pinecone with client filter
            search_results = await pinecone_service.query_embeddings(
                query_embedding=query_embedding,
                client_id=client_id,
                top_k=top_k,
                include_metadata=True
            )
            
            if not search_results.get("success"):
                logger.warning("Pinecone query failed, using empty context")
                return []
            
            # Extract and format context chunks
            context_chunks = []
            for match in search_results.get("matches", []):
                metadata = match.get("metadata", {})
                context_chunks.append({
                    "text": metadata.get("text", ""),
                    "score": match.get("score", 0),
                    "source": metadata.get("filename", "Knowledge Base"),
                    "chunk_index": metadata.get("chunk_index", 0)
                })
            
            # Sort by relevance score and limit context length
            context_chunks.sort(key=lambda x: x["score"], reverse=True)
            return self._limit_context_length(context_chunks)
            
        except Exception as e:
            logger.error(f"Error searching context: {str(e)}")
            return []
    
    def _build_rag_prompt(
        self,
        message: str,
        context_chunks: List[Dict],
        business_name: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Build RAG prompt with context and conversation history
        
        Args:
            message: Current user message
            context_chunks: Relevant context from knowledge base
            business_name: Name of the business
            conversation_history: Previous conversation messages
            
        Returns:
            Formatted prompt for AI
        """
        # Build context section
        context_section = ""
        if context_chunks:
            context_section = "RELEVANT INFORMATION FROM KNOWLEDGE BASE:\n"
            for i, chunk in enumerate(context_chunks, 1):
                context_section += f"{i}. {chunk['text']}\n"
        else:
            context_section = "No specific information found in knowledge base.\n"
        
        # Build conversation history section
        history_section = ""
        if conversation_history:
            history_section = "PREVIOUS CONVERSATION:\n"
            for msg in conversation_history[-self.max_conversation_history:]:
                role = "USER" if msg.get("role") == "user" else "ASSISTANT"
                content = msg.get("content", "")[:200]  # Truncate long messages
                history_section += f"{role}: {content}\n"
        
        # Construct the full prompt
        prompt = f"""You are a helpful AI assistant for {business_name}. Your role is to answer customer questions accurately based on the provided context.

IMPORTANT RULES:
1. Answer STRICTLY using information from the context below
2. If the context doesn't contain relevant information, say: "I don't have enough specific information about that in my knowledge base. Please contact {business_name} directly for the most accurate assistance."
3. Be friendly, professional, and helpful
4. Keep responses concise but informative (2-4 sentences typically)
5. Never make up information or use external knowledge
6. If you're unsure, acknowledge the limitation

{context_section}

{history_section}

CURRENT QUESTION: {message}

YOUR RESPONSE (as {business_name}'s AI assistant, using ONLY the context above):"""
        
        return prompt
    
    async def _generate_ai_response(self, prompt: str, business_name: str) -> str:
        """
        Generate AI response using Gemini
        
        Args:
            prompt: Formatted prompt for AI
            business_name: Business name for error handling
            
        Returns:
            AI-generated response text
        """
        try:
            # Use Gemini for response generation
            response = gemini_service.generate_response(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=500,
                system_message=f"You are a helpful AI assistant for {business_name}."
            )
            
            if response and response.strip():
                return response.strip()
            else:
                return f"I apologize, but I'm having trouble generating a response right now. Please contact {business_name} directly for assistance."
                
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return f"I'm experiencing technical difficulties. Please contact {business_name} directly for assistance."
    
    async def _manage_session(
        self, 
        client_id: str, 
        session_id: Optional[str], 
        db: Session
    ) -> str:
        """
        Manage webchat session - create new or update existing
        
        Args:
            client_id: Client ID
            session_id: Existing session ID or None
            db: Database session
            
        Returns:
            Valid session ID
        """
        try:
            if session_id:
                # Update existing session
                session = db.query(WebchatSession).filter(
                    WebchatSession.session_id == session_id,
                    WebchatSession.client_id == client_id
                ).first()
                
                if session and session.is_active:
                    session.last_activity = datetime.utcnow()
                    session.message_count += 1
                    db.commit()
                    return session_id
            
            # Create new session
            new_session_id = f"session_{uuid.uuid4().hex[:16]}"
            new_session = WebchatSession(
                client_id=client_id,
                session_id=new_session_id,
                started_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                message_count=1,
                is_active=True
            )
            
            db.add(new_session)
            db.commit()
            
            return new_session_id
            
        except Exception as e:
            logger.error(f"Error managing session: {str(e)}")
            # Return a fallback session ID
            return f"session_fallback_{uuid.uuid4().hex[:8]}"
    
    async def _log_interaction(
        self,
        db: Session,
        client_id: str,
        session_id: str,
        user_message: str,
        ai_response: str,
        context_used: List[Dict]
    ) -> None:
        """
        Log chat interaction to database
        
        Args:
            db: Database session
            client_id: Client ID
            session_id: Session ID
            user_message: User's original message
            ai_response: AI's response
            context_used: Context chunks used for response
        """
        try:
            # Create message log entry
            message_log = MessageLog(
                client_id=client_id,
                channel="web",
                message_text=user_message,
                response_text=ai_response,
                session_id=session_id,
                timestamp=datetime.utcnow()
            )
            
            db.add(message_log)
            db.commit()
            
            logger.debug(f"Logged interaction for client {client_id}, session {session_id}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            # Don't raise error - logging failure shouldn't break chat
    
    def _limit_context_length(self, context_chunks: List[Dict], max_length: int = None) -> List[Dict]:
        """
        Limit total context length to fit within token limits
        
        Args:
            context_chunks: List of context chunks
            max_length: Maximum total character length
            
        Returns:
            Limited list of context chunks
        """
        if max_length is None:
            max_length = self.max_context_length
        
        limited_chunks = []
        total_length = 0
        
        for chunk in context_chunks:
            chunk_text = chunk.get("text", "")
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length <= max_length:
                limited_chunks.append(chunk)
                total_length += chunk_length
            else:
                # Add partial chunk if there's space
                remaining_space = max_length - total_length
                if remaining_space > 100:  # Only add if meaningful space remains
                    partial_chunk = chunk.copy()
                    partial_chunk["text"] = chunk_text[:remaining_space] + "..."
                    limited_chunks.append(partial_chunk)
                break
        
        return limited_chunks
    
    async def _fallback_response(self, message: str, business_name: str) -> Dict[str, Any]:
        """
        Generate fallback response when main processing fails
        
        Args:
            message: User's original message
            business_name: Business name
            
        Returns:
            Fallback response dictionary
        """
        fallback_responses = [
            f"I'm currently experiencing technical difficulties. Please contact {business_name} directly for assistance with your question: '{message}'",
            f"I apologize, but I'm unable to process your request right now. For the most accurate information, please reach out to {business_name} directly.",
            f"Thank you for your message. I'm temporarily unavailable. Please contact {business_name} for assistance with: '{message}'"
        ]
        
        import random
        response = random.choice(fallback_responses)
        
        return {
            "success": False,
            "response": response,
            "session_id": f"fallback_{uuid.uuid4().hex[:8]}",
            "conversation_id": f"fallback_{uuid.uuid4().hex[:8]}",
            "client_id": "unknown",
            "sources": [],
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Service temporarily unavailable"
        }
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response
        
        Args:
            error_message: Error description
            
        Returns:
            Error response dictionary
        """
        return {
            "success": False,
            "response": f"I apologize, but I'm unable to assist at the moment. {error_message}",
            "session_id": f"error_{uuid.uuid4().hex[:8]}",
            "conversation_id": f"error_{uuid.uuid4().hex[:8]}", 
            "client_id": "unknown",
            "sources": [],
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message
        }
    
    async def generate_embed_code(self, client: Client) -> str:
        """
        Generate embed code for client's website
        
        Args:
            client: Client object
            
        Returns:
            HTML embed code string
        """
        try:
            # Generate unique ID if not exists
            if not client.unique_id:
                client.unique_id = f"{client.business_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
            
            # Generate chatbot URL
            chatbot_url = f"https://{os.getenv('DOMAIN', 'yourdomain.com')}/web_chat/{client.unique_id}"
            client.chatbot_url = chatbot_url
            
            # Create embed code
            embed_code = f'''
<!-- OwnBot Chat Widget -->
<div id="ownbot-chat-widget-{client.unique_id}"></div>
<script>
    window.ownBotConfig = {{
        apiBaseUrl: 'https://{os.getenv('DOMAIN', 'yourdomain.com')}',
        clientId: '{client.id}',
        position: '{client.webchat_position or "bottom-right"}',
        primaryColor: '{client.webchat_primary_color or "#007bff"}',
        greetingMessage: '{client.webchat_welcome_message or self.default_welcome_message}',
        autoOpen: false
    }};
</script>
<script src="https://{os.getenv('DOMAIN', 'yourdomain.com')}/static/js/chat-widget.js"></script>
<link rel="stylesheet" href="https://{os.getenv('DOMAIN', 'yourdomain.com')}/static/css/chat-widget.css">
<!-- End OwnBot Chat Widget -->
            '''.strip()
            
            client.embed_code = embed_code
            return embed_code
            
        except Exception as e:
            logger.error(f"Error generating embed code: {str(e)}")
            return "<!-- Error generating embed code -->"
    
    async def get_chat_analytics(self, client_id: str, db: Session, days: int = 30) -> Dict[str, Any]:
        """
        Get chat analytics for a client
        
        Args:
            client_id: Client ID
            db: Database session
            days: Number of days to look back
            
        Returns:
            Analytics data dictionary
        """
        try:
            from datetime import timedelta
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get message counts
            total_messages = db.query(MessageLog).filter(
                MessageLog.client_id == client_id,
                MessageLog.timestamp >= start_date
            ).count()
            
            # Get active sessions
            active_sessions = db.query(WebchatSession).filter(
                WebchatSession.client_id == client_id,
                WebchatSession.is_active == True
            ).count()
            
            # Get popular questions (simplified)
            recent_messages = db.query(MessageLog).filter(
                MessageLog.client_id == client_id,
                MessageLog.timestamp >= start_date
            ).order_by(MessageLog.timestamp.desc()).limit(100).all()
            
            return {
                "total_messages": total_messages,
                "active_sessions": active_sessions,
                "period_days": days,
                "recent_activity": len(recent_messages)
            }
            
        except Exception as e:
            logger.error(f"Error getting chat analytics: {str(e)}")
            return {"error": "Failed to retrieve analytics"}


# Global singleton instance
chat_service = ChatService()

