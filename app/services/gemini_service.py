import os
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
# Add this import
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Comprehensive service for Google Gemini AI
    Handles chat completion, embeddings, and advanced AI features
    """
    
    def __init__(self):
        """Initialize Gemini AI with comprehensive configuration"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.embedding_model = "models/embedding-001"  # Keep Gemini embedding model reference
        self.max_retries = 3
        self.request_timeout = 30
        self.is_available = False
        self.model = None
    
        # Add Hugging Face for embeddings only
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY")  # Add this line
        self.hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Add this line
        self.embedding_dimension = 384  # Keep this as is
    
        # Initialize the service
        self.initialize()
    
    def initialize(self):
        """Initialize Gemini AI with comprehensive error handling"""
        try:
            if not self.api_key:
                logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
                logger.warning("AI responses and embeddings will not be available")
                return
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Initialize the main model
            self.model = genai.GenerativeModel(self.model_name)
            
            # Test the configuration with a simple API call
            self._test_connection()
            
            self.is_available = True
            logger.info(f"✅ Gemini AI initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini AI: {e}")
            self.is_available = False
    
    def _test_connection(self):
        """Test connection to Gemini API"""
        try:
            # Simple test to verify API key and connectivity
            test_response = self.model.generate_content("Test connection", safety_settings={
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',  # ✅ Correct key
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
})
            
            if test_response and test_response.text:
                logger.debug("✅ Gemini API connection test passed")
            else:
                raise ValueError("Empty response from Gemini API")
                
        except Exception as e:
            logger.error(f"❌ Gemini API connection test failed: {e}")
            raise
    
    def check_availability(self) -> bool:
        """
        Check if Gemini service is available and configured
        
        Returns:
            True if service is available, False otherwise
        """
        return self.is_available and self.model is not None
    
def generate_embedding(self, text: str) -> List[float]:
    """
    Generate embedding vector for text using Hugging Face API
    """
    try:
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for embedding")
            return [0.0] * self.embedding_dimension
        
        if not self.hf_token:
            logger.warning("⚠️ Hugging Face token not configured, using zero vector")
            return [0.0] * self.embedding_dimension
        
        # Use Hugging Face Inference API for embeddings
        client = InferenceClient(token=self.hf_token)
        embedding = client.feature_extraction(text, model=self.hf_embedding_model)
        
        # Convert to list
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        # Flatten if needed (HF returns 2D array)
        if isinstance(embedding_list[0], list):
            embedding_list = embedding_list[0]
        
        logger.debug(f"✅ Generated Hugging Face embedding with dimension: {len(embedding_list)}")
        return embedding_list
            
    except Exception as e:
        logger.error(f"❌ Error generating Hugging Face embedding: {str(e)}")
        return [0.0] * self.embedding_dimension
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Async wrapper for embedding generation
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.generate_embedding, text)
            return embedding
        except Exception as e:
            logger.error(f"❌ Error in async embedding generation: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate AI response using Gemini with advanced configuration
        
        Args:
            prompt: The prompt to send to Gemini
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            system_message: Optional system message for context
            
        Returns:
            Generated response text
        """
        if not self.check_availability():
            logger.error("❌ Gemini service not available")
            return "I apologize, but the AI service is currently unavailable. Please try again later or contact support."
        
        try:
            # Build the full prompt with system message if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            logger.info(f"🤖 Generating Gemini response (temperature: {temperature})")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=max_tokens,
                stop_sequences=None
            )
            
            # Safety settings to minimize blocking
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response and response.text:
                response_text = response.text.strip()
                logger.info(f"✅ Successfully generated response ({len(response_text)} chars)")
                return response_text
            else:
                logger.warning("❌ Empty or blocked response from Gemini")
                return "I apologize, but I couldn't generate a response. This might be due to content safety filters or technical issues."
                
        except Exception as e:
            logger.error(f"❌ Error generating Gemini response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
    
    def create_rag_prompt(
        self, 
        context: str, 
        query: str, 
        business_name: str,
        instructions: Optional[str] = None
    ) -> str:
        """
        Create an optimized RAG prompt for Gemini
        
        Args:
            context: Retrieved context from knowledge base
            query: User's question
            business_name: Name of the business
            instructions: Custom instructions for the AI
            
        Returns:
            Formatted prompt optimized for Gemini
        """
        base_instructions = f"""You are a knowledgeable AI assistant for {business_name}. Your role is to help customers by answering their questions accurately based ONLY on the provided context.

CRITICAL RULES:
1. Answer STRICTLY using information from the context provided below
2. If the context doesn't contain relevant information, say: "I don't have enough information to answer that accurately. Please contact {business_name} directly for assistance."
3. Be friendly, professional, and concise (2-4 sentences)
4. Never make up information or use external knowledge
5. If unsure, acknowledge the limitation
6. Use the business name naturally when appropriate"""

        custom_instructions = instructions or ""
        
        prompt = f"""{base_instructions}
{custom_instructions}

CONTEXT FROM {business_name.upper()}'S KNOWLEDGE BASE:
{context}

CUSTOMER QUESTION:
{query}

YOUR RESPONSE (as {business_name}'s AI assistant, using ONLY the context above):"""
        
        return prompt
    
    def generate_contextual_response(
        self, 
        context: str, 
        query: str, 
        business_name: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.3  # Lower temperature for more consistent RAG responses
    ) -> str:
        """
        Generate a contextual response using RAG approach with conversation history
        
        Args:
            context: Retrieved context from knowledge base
            query: User's question
            business_name: Name of the business
            conversation_history: Optional list of previous messages
            temperature: Controls response randomness
            
        Returns:
            AI-generated response
        """
        # Validate inputs
        if not context or not context.strip():
            logger.warning(f"⚠️ No context available for query: {query}")
            return f"I'm the AI assistant for {business_name}. I don't have enough information in my knowledge base to answer that question accurately. Please ensure relevant documents have been uploaded, or contact {business_name} directly for assistance."
        
        if not query or len(query.strip()) < 2:
            return "Could you please clarify your question? I want to make sure I provide you with the most accurate information."
        
        # Build enhanced prompt with conversation history
        prompt = self.create_rag_prompt(context, query, business_name)
        
        # Add conversation history for context continuity
        if conversation_history and len(conversation_history) > 0:
            history_context = self._format_conversation_history(conversation_history)
            prompt = f"{history_context}\n\n{prompt}"
        
        # Generate response with lower temperature for more factual accuracy
        return self.generate_response(prompt, temperature=temperature, max_tokens=512)
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """
        Format conversation history for context
        
        Args:
            history: List of message dictionaries
            
        Returns:
            Formatted conversation history string
        """
        formatted = "PREVIOUS CONVERSATION (for context only):\n"
        
        # Take last 6 messages (3 exchanges) for context
        for msg in history[-6:]:
            role = "CUSTOMER" if msg.get("role") in ["user", "customer"] else "ASSISTANT"
            content = msg.get("content", "")[:200]  # Truncate long messages
            formatted += f"{role}: {content}\n"
        
        return formatted
    
    def generate_simple_response(
        self, 
        query: str, 
        business_name: str,
        use_rag_fallback: bool = True
    ) -> str:
        """
        Generate a simple response for general queries
        
        Args:
            query: User's question
            business_name: Name of the business
            use_rag_fallback: Whether to suggest RAG for specific queries
            
        Returns:
            AI-generated response
        """
        if use_rag_fallback and self._requires_specific_knowledge(query):
            return f"I'm {business_name}'s AI assistant. For specific questions about {business_name}'s services, policies, or offerings, I can provide more accurate answers if you've uploaded relevant documents to my knowledge base. Otherwise, please contact {business_name} directly for the most accurate information."
        
        prompt = f"""You are a helpful AI assistant for {business_name}. 

Customer question: {query}

Provide a brief, helpful response. If this requires specific information about {business_name}'s services, products, or policies, politely explain that you need more specific context.

Response:"""
        
        return self.generate_response(prompt, temperature=0.7)
    
    def _requires_specific_knowledge(self, query: str) -> bool:
        """
        Determine if a query likely requires specific business knowledge
        
        Args:
            query: User's question
            
        Returns:
            True if query likely needs specific business context
        """
        specific_keywords = [
            'price', 'cost', 'fee', 'hour', 'open', 'close', 'service',
            'product', 'policy', 'procedure', 'requirement', 'document',
            'form', 'application', 'process', 'timeline', 'deadline'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in specific_keywords)
    
    def validate_api_key(self) -> Dict[str, Any]:
        """
        Comprehensive API key validation
        
        Returns:
            Dictionary with validation results
        """
        try:
            if not self.api_key:
                return {
                    "valid": False,
                    "error": "No API key configured",
                    "details": "GEMINI_API_KEY environment variable is missing"
                }
            
            # Test both chat and embedding capabilities
            chat_test = self.model.generate_content("Test", safety_settings={
                'HARASSMENT': 'block_none',
                'HATE_SPEECH': 'block_none',
                'SEXUALLY_EXPLICIT': 'block_none',
                'DANGEROUS_CONTENT': 'block_none'
            })
            
            embedding_test = genai.embed_content(
                model=self.embedding_model,
                content="Test embedding",
                task_type="retrieval_document"
            )
            
            if chat_test and embedding_test:
                return {
                    "valid": True,
                    "model": self.model_name,
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": self.embedding_dimension
                }
            else:
                return {
                    "valid": False,
                    "error": "API test failed",
                    "details": "One or more API tests returned empty response"
                }
                
        except Exception as e:
            return {
                "valid": False,
                "error": "API validation error",
                "details": str(e)
            }
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Generate a summary of the provided text with retry logic
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        if not self.check_availability():
            # Fallback: truncate text
            return text[:max_length] + "..." if len(text) > max_length else text
        
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters. Focus on the key points and main ideas.

Text:
{text}

Concise Summary:"""
        
        return self.generate_response(prompt, temperature=0.3, max_tokens=300)
    
    def extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from text with improved parsing
        
        Args:
            text: Text to analyze
            max_points: Maximum number of key points to extract
            
        Returns:
            List of key points
        """
        if not self.check_availability():
            return ["AI service unavailable for key point extraction"]
        
        prompt = f"""Extract {max_points} key points from the following text. Format each point as a concise, complete sentence without numbering or bullets.

Text:
{text}

Key Points (one per line, no numbering):"""
        
        response = self.generate_response(prompt, temperature=0.3)
        
        # Improved parsing of response
        points = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove common prefixes and ensure it's a meaningful sentence
            if line and len(line) > 10:  # Minimum length for a meaningful point
                # Clean up the line
                clean_line = line.lstrip('0123456789.-•) ').strip()
                if clean_line and clean_line[0].isupper():  # Should start with capital letter
                    points.append(clean_line)
        
        return points[:max_points] if points else ["No key points could be extracted"]
    
    def classify_intent(self, query: str, possible_intents: List[str]) -> Dict[str, Any]:
        """
        Classify the intent of a user query
        
        Args:
            query: User's question
            possible_intents: List of possible intent labels
            
        Returns:
            Dictionary with classification results
        """
        if not self.check_availability():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error": "Service unavailable"
            }
        
        intent_list = ", ".join(possible_intents)
        
        prompt = f"""Classify the user's query into one of these intents: {intent_list}

Query: "{query}"

Respond with ONLY the intent label that best matches, nothing else."""
        
        try:
            response = self.generate_response(prompt, temperature=0.1, max_tokens=50)
            classified_intent = response.strip().lower()
            
            # Calculate basic confidence (simplified)
            confidence = 0.8 if classified_intent in [i.lower() for i in possible_intents] else 0.3
            
            return {
                "intent": classified_intent,
                "confidence": confidence,
                "possible_intents": possible_intents
            }
            
        except Exception as e:
            logger.error(f"❌ Error classifying intent: {str(e)}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Gemini service
        
        Returns:
            Dictionary with service information
        """
        return {
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "is_available": self.is_available,
            "has_api_key": bool(self.api_key),
            "provider": "Google Generative AI",
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for Gemini service
        
        Returns:
            Health status dictionary
        """
        try:
            # Test basic connectivity
            api_validation = self.validate_api_key()
            
            # Test embedding generation
            embedding_test = await self.generate_embedding_async("health check")
            
            health_status = {
                "service": "gemini",
                "status": "healthy" if api_validation.get("valid") else "unhealthy",
                "api_key_valid": api_validation.get("valid", False),
                "embedding_working": len(embedding_test) == self.embedding_dimension,
                "model_available": self.is_available,
                "timestamp": time.time(),
                "details": api_validation
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Gemini health check failed: {str(e)}")
            return {
                "service": "gemini",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (sequential)
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                logger.debug(f"✅ Generated embedding {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"❌ Failed to generate embedding for text {i+1}: {str(e)}")
                # Add zero vector as fallback
                embeddings.append([0.0] * self.embedding_dimension)
        
        return embeddings
    
    async def batch_generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts asynchronously
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        tasks = [self.generate_embedding_async(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        final_embeddings = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error(f"❌ Async embedding failed for text {i+1}: {str(emb)}")
                final_embeddings.append([0.0] * self.embedding_dimension)
            else:
                final_embeddings.append(emb)
        
        return final_embeddings


# Global singleton instance
gemini_service = GeminiService()
