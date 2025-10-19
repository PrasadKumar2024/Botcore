import os
import logging
from typing import Optional, List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Service for interacting with Google Gemini AI for generating intelligent responses
    Handles AI conversation, context-aware responses, and error handling
    """
    
    def __init__(self):
        """Initialize Gemini AI with API key from environment"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = None
        self.is_available = False
        
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
            logger.warning("AI responses will not be available until API key is configured")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_available = True
                logger.info("✅ Gemini AI initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini AI: {e}")
                self.is_available = False
    
    def check_availability(self) -> bool:
        """
        Check if Gemini service is available and configured
        
        Returns:
            True if service is available, False otherwise
        """
        return self.is_available and self.model is not None
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate AI response using Gemini with retry logic
        
        Args:
            prompt: The prompt to send to Gemini
            max_retries: Number of retry attempts on failure
            
        Returns:
            Generated response text
        """
        if not self.check_availability():
            logger.error("Gemini service not available")
            return "I apologize, but the AI service is currently unavailable. Please try again later or contact support."
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating Gemini response (attempt {attempt + 1}/{max_retries})")
                
                # Configure generation parameters for better responses
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,  # Balance between creativity and consistency
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                if response and response.text:
                    logger.info(f"✅ Successfully generated response ({len(response.text)} chars)")
                    return response.text.strip()
                else:
                    logger.warning("Empty response from Gemini")
                    if attempt == max_retries - 1:
                        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
            except Exception as e:
                logger.error(f"Error generating Gemini response (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
                
                # Wait before retry (exponential backoff)
                import time
                time.sleep(2 ** attempt)
        
        return "Service temporarily unavailable. Please try again."
    
    def create_rag_prompt(self, context: str, query: str, business_name: str) -> str:
        """
        Create a RAG (Retrieval Augmented Generation) prompt for answering questions
        
        Args:
            context: Retrieved context from knowledge base
            query: User's question
            business_name: Name of the business
            
        Returns:
            Formatted prompt optimized for Gemini
        """
        prompt = f"""You are a knowledgeable AI assistant representing {business_name}. Your role is to help customers by answering their questions accurately based on the business's official documents and information.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using information from the context provided below
2. Be friendly, professional, and helpful in your tone
3. If the context doesn't contain enough information to answer the question, politely acknowledge this and suggest contacting {business_name} directly
4. Keep your response concise and focused (2-4 sentences is ideal)
5. Use the business name naturally when appropriate
6. Do NOT make up information or use external knowledge
7. If you're uncertain, it's better to say you don't know than to guess

CONTEXT FROM {business_name.upper()}'S DOCUMENTS:
{context}

CUSTOMER'S QUESTION:
{query}

YOUR RESPONSE (as {business_name}'s AI assistant):"""
        
        return prompt
    
    def generate_contextual_response(
        self, 
        context: str, 
        query: str, 
        business_name: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a contextual response using RAG approach
        
        Args:
            context: Retrieved context from knowledge base
            query: User's question
            business_name: Name of the business
            conversation_history: Optional list of previous messages for context
            
        Returns:
            AI-generated response
        """
        # Handle case where no context is available
        if not context or not context.strip():
            logger.warning(f"No context available for query: {query}")
            return f"I'm the AI assistant for {business_name}. I don't have enough information in my knowledge base to answer that question accurately. Please make sure relevant documents have been uploaded and processed, or contact {business_name} directly for assistance."
        
        # Build the prompt with context
        prompt = self.create_rag_prompt(context, query, business_name)
        
        # Add conversation history if provided
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\nRECENT CONVERSATION HISTORY:\n"
            for msg in conversation_history[-4:]:  # Last 4 messages for context
                role = "Customer" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            
            prompt = history_text + "\n" + prompt
        
        # Generate and return response
        return self.generate_response(prompt)
    
    def generate_simple_response(self, query: str, business_name: str) -> str:
        """
        Generate a simple response without RAG (for general queries)
        
        Args:
            query: User's question
            business_name: Name of the business
            
        Returns:
            AI-generated response
        """
        prompt = f"""You are a helpful AI assistant for {business_name}. 
        
The customer has asked: {query}

Provide a brief, helpful response. If this requires specific information about {business_name}'s services or policies, politely suggest they contact {business_name} directly.

Your response:"""
        
        return self.generate_response(prompt)
    
    def validate_api_key(self) -> bool:
        """
        Validate that the Gemini API key is working
        
        Returns:
            True if API key is valid and working, False otherwise
        """
        if not self.api_key:
            logger.error("No API key configured")
            return False
        
        try:
            # Try to list models as a validation check
            test_prompt = "Test"
            response = self.model.generate_content(test_prompt)
            
            if response:
                logger.info("✅ Gemini API key validated successfully")
                return True
            else:
                logger.error("❌ Gemini API key validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ API key validation error: {e}")
            return False
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Generate a summary of the provided text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        if not self.check_availability():
            return text[:max_length] + "..." if len(text) > max_length else text
        
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters:

{text}

Summary:"""
        
        return self.generate_response(prompt)
    
    def extract_key_points(self, text: str) -> List[str]:
        """
        Extract key points from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key points
        """
        if not self.check_availability():
            return ["AI service unavailable"]
        
        prompt = f"""Extract 3-5 key points from the following text. Format each point as a brief sentence.

Text:
{text}

Key Points (one per line, numbered):"""
        
        response = self.generate_response(prompt)
        
        # Parse numbered list
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                point = line.lstrip('0123456789.-•) ').strip()
                if point:
                    points.append(point)
        
        return points if points else [response]
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current Gemini model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "gemini-pro",
            "is_available": self.is_available,
            "has_api_key": bool(self.api_key),
            "provider": "Google Generative AI"
        }

# Singleton instance
gemini_service = GeminiService()

