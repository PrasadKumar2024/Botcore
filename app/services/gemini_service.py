import os
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, Generator, AsyncIterator, Union
import google.generativeai as genai
from dotenv import load_dotenv
import re
import threading
from enum import Enum
from dataclasses import dataclass
from typing import AsyncGenerator
FunctionDeclaration = Any  # Will be properly typed based on your Gemini SDK version
# Add this import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

load_dotenv()

logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    GREETING = "greeting"
    CLOSING = "closing"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    BILLING = "billing"
    SUPPORT = "support"
    HUMAN_HANDOFF = "human_handoff"
    QUESTION = "question"  # Fallback for RAG
    UNKNOWN = "unknown"

@dataclass
class NLUResult:
    intent: IntentType
    confidence: float
    sentiment: float      # -1.0 to 1.0
    urgency: str          # low, medium, high
    entities: Dict[str, Any]
    topic_allowed: bool   # False if politics/religion

# --- REFLEX PATTERNS (The 0ms "Nervous System") ---
_REFLEX_PATTERNS = {
    IntentType.CLOSING: [
        r"\b(bye|goodbye|stop|end call|hang up|quit)\b"
    ],
    IntentType.HUMAN_HANDOFF: [
        r"\b(human|agent|representative|operator|person|support team)\b"
    ],
    IntentType.GREETING: [
        r"^(hi|hello|hey|good morning|good evening)$"  # Strict to prevent "Hello I have a problem"
    ]
}
class GeminiService:
    """
    Comprehensive service for Google Gemini AI
    Handles chat completion, embeddings, and advanced AI features
    """
    
    def __init__(self):
        """Initialize Gemini AI with comprehensive configuration"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.embedding_model = "models/embedding-001"  # Keep Gemini embedding model r
        self.max_retries = 3
        self.request_timeout = 30
        self.is_available = False
        self.model = None
    
        # Add Hugging Face for embeddings only
          # Add this line
        self.embedding_dimension = 1024# Keep this as is
    
        # Initialize the service
        self.initialize()
    
    def initialize(self):
        """Initialize Gemini AI with comprehensive error handling"""
        try:
            if not self.api_key:
                logger.warning("âš ï¸ GEMINI_API_KEY not found in environment variables")
                logger.warning("AI responses and embeddings will not be available")
                return
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Initialize the main model
            self.model = genai.GenerativeModel(self.model_name)
            
            # Test the configuration with a simple API call
            self._test_connection()
            
            self.is_available = True
            logger.info(f"âœ… Gemini AI initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"âŒ Failed to initialize Gemini AI: {e}")
            self.is_available = False

    def generate_stream(self, prompt: str, system_message: Optional[str] = None,
                        temperature: float = 0.6, max_tokens: int = 1024) -> Generator[str, None, None]:
        """
        Token/text-level streaming generator.
    
        - Tries provider streaming entrypoints (non-breaking checks).
        - Normalizes many event shapes (str, dict, objects with attributes).
        - If no streaming entrypoint exists, raises RuntimeError (strict streaming mode).
        """
        client = getattr(self, "client", None) or getattr(self, "_client", None) or None
    
        def _normalize_event_to_text(event: Any) -> Optional[str]:
            # plain string
            if isinstance(event, str):
                return event
            # dict shapes
            if isinstance(event, dict):
                # delta content patterns
                d = event.get("delta") or event.get("delta_chunk") or {}
                if isinstance(d, dict):
                    t = d.get("content") or d.get("text") or d.get("message")
                    if t:
                        return t
                # choices[].delta.text or choices[].text
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    try:
                        first = choices[0]
                        if isinstance(first, dict):
                            delta = first.get("delta") or {}
                            if isinstance(delta, dict):
                                t = delta.get("content") or delta.get("text")
                                if t:
                                    return t
                            t = first.get("text") or first.get("message")
                            if t:
                                return t
                    except Exception:
                        pass
                # fallback keys
                for k in ("text", "content", "output_text", "message", "output"):
                    if k in event and event[k]:
                        return event[k]
                return None
            # object shapes (attributes)
            for attr in ("delta", "text", "content", "output_text", "message"):
                val = getattr(event, attr, None)
                if val:
                    if attr == "delta":
                        if isinstance(val, dict):
                            return val.get("content") or val.get("text") or None
                        else:
                            return getattr(val, "text", None) or getattr(val, "content", None)
                    return val
            # choices attribute on object
            choices = getattr(event, "choices", None)
            if choices:
                try:
                    first = choices[0]
                    delta = getattr(first, "delta", None) or (first.get("delta") if isinstance(first, dict) else None)
                    if delta:
                        return getattr(delta, "text", None) or (delta.get("content") if isinstance(delta, dict) else None)
                    return getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
                except Exception:
                    pass
            return None
    
        # try provider streaming methods (non-breaking)
        if client is not None:
            # common candidate callsites (be permissive)
            tries = [
                (getattr(client, "responses", None), "stream"),
                (getattr(client, "responses", None), "stream_create"),
                (client, "stream"),
                (client, "streaming_generate"),
                (client, "generate_stream"),
            ]
            for obj, method_name in tries:
                if obj is None:
                    continue
                stream_fn = getattr(obj, method_name, None)
                if not stream_fn or not callable(stream_fn):
                    continue
    
                # try several common kwarg signatures
                variants = [
                    {"model": getattr(self, "model", None), "prompt": prompt, "system_message": system_message, "temperature": temperature, "max_tokens": max_tokens},
                    {"model": getattr(self, "model", None), "input": prompt, "system_message": system_message, "temperature": temperature, "max_output_tokens": max_tokens},
                    {"prompt": prompt, "temperature": temperature},
                    {"input": [{"role":"user","content": prompt}], "temperature": temperature, "max_tokens": max_tokens},
                ]
                for kwargs in variants:
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    try:
                        gen = stream_fn(**kwargs)
                    except TypeError:
                        # signature mismatch -> try next variant
                        continue
                    except Exception:
                        # provider stream failed -> try next attempt
                        break
    
                    # If we got an iterator/generator/stream object, iterate and normalize
                    try:
                        for evt in gen:
                            if evt is None:
                                continue
                            text_piece = _normalize_event_to_text(evt)
                            if text_piece:
                                yield text_piece
                        # finished normally: return
                        return
                    except Exception:
                        # streaming from provider errored: break to try other methods
                        break
    
        # If we get here: strict streaming required but no provider stream usable
        raise RuntimeError("No provider streaming entrypoint found. Implement provider streaming or attach a client supporting streaming.")

    def rewrite_query(self, user_query: str, conversation_history: List[Dict] = None) -> str:
        """
        Rewrites the user's query to be specific and searchable.
        """
        if not self.check_availability():
            return user_query

        prompt = f"""You are an AI Search Optimizer. Rewrite this query to be perfect for a vector search.
        1. Remove filler words.
        2. Expand synonyms (e.g., "timings" -> "operating hours schedule").
        3. OUTPUT ONLY THE REWRITTEN QUERY. NO QUOTES.

        User Raw Query: "{user_query}"
        """
        
        try:
            # Fast, low-temp generation
            response = self.generate_response(prompt, temperature=0.1, max_tokens=50)
            cleaned = response.strip().replace('"', '').replace("'", "")
            logger.info(f"ðŸ”„ Rewritten: {user_query} -> {cleaned}")
            return cleaned
        except Exception:
            return user_query
            
    def _test_connection(self):
        """Test connection to Gemini API"""
        try:
            # Simple test to verify API key and connectivity
            test_response = self.model.generate_content("Test connection", safety_settings={
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',  # âœ… Correct key
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
})
            
            if test_response and test_response.text:
                logger.debug("âœ… Gemini API connection test passed")
            else:
                raise ValueError("Empty response from Gemini API")
                
        except Exception as e:
            logger.error(f"âŒ Gemini API connection test failed: {e}")
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
        Generate embedding using Gemini native API
        (FINAL: Combines best validation + removes GoogleAPIError)
        """
    # Input validation
        if not text or not text.strip():
            logger.warning("Empty text for embedding")
            return [0.0] * self.embedding_dimension

        if not self.api_key:
            logger.error("Gemini API key not configured")
            raise Exception("Gemini API key not configured")

        if not self.embedding_model:
            logger.error("Gemini embedding model not available")
            raise Exception("Gemini embedding model not configured")

        try:
        # Text cleaning
            clean_text = text.strip().replace('\n', ' ').replace('\r', ' ')
            if len(clean_text) > 10000:
                logger.warning(f"Truncating long text: {len(clean_text)} chars")
                clean_text = clean_text[:10000]

        # Generate embedding
            result = genai.embed_content(
                model=self.embedding_model,
                content=clean_text,
                task_type="retrieval_document"
            )

        # Validate response
            if not result or 'embedding' not in result:
                raise Exception("Invalid embedding response from Gemini")
        
            embedding = result['embedding']
        
        # Enhanced validation
            if not embedding or len(embedding) == 0:
                raise Exception("Empty embedding vector")
        
            if all(abs(x) < 1e-8 for x in embedding):
                raise Exception("Zero vector embedding")
        
        # Dimension check
            if len(embedding) != self.embedding_dimension:
                logger.warning(f"Dimension mismatch: {len(embedding)} vs {self.embedding_dimension}")

        # Success logging
            magnitude = (sum(x*x for x in embedding)) ** 0.5
            logger.info(f"Gemini embedding: {len(embedding)}D, magnitude: {magnitude:.3f}")
        
            return embedding

        except Exception as e:
            logger.error(f"Gemini embedding failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Async wrapper for embedding generation
        (FIX: Now includes retry and propagates errors)
        """
        try:
            loop = asyncio.get_event_loop()
            # This will now retry 3 times if self.generate_embedding fails
            embedding = await loop.run_in_executor(None, self.generate_embedding, text)
            return embedding
        except Exception as e:
            logger.error(f"âŒ Async embedding generation failed after retries: {str(e)}")
            # If it fails after all retries, return zero vector as last resort
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
        max_tokens: int = 600,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate AI response using Gemini with advanced configuration
        """
        if not self.check_availability():
            logger.error("âŒ Gemini service not available")
            return "I apologize, but the AI service is currently unavailable. Please try again later or contact support."
        
        try:
            # Build the full prompt with system message if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            logger.info(f"ðŸ¤– Generating Gemini response (temperature: {temperature})")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=max_tokens,
            )
            
            # âœ… FIX 1: ULTRA-ROBUST SAFETY SETTINGS
            # Using list of dictionaries is the most compatible way to prevent "finish_reason: 2"
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # âœ… FIX 2: PREVENT CRASH ON BLOCKED CONTENT
            # We strictly check if the response has text parts before accessing .text
            try:
                if response.parts:
                    response_text = response.text.strip()
                    logger.info(f"âœ… Successfully generated response ({len(response_text)} chars)")
                    return response_text
                else:
                    # If parts is empty, it was blocked or empty
                    logger.warning(f"âš ï¸ Gemini response empty. Finish Reason: {response.candidates[0].finish_reason}")
                    return "I apologize, but I couldn't generate a response due to safety guidelines."
            except ValueError:
                # Catch the specific Google API error for blocked content
                logger.warning("âš ï¸ Gemini blocked the response (ValueError caught).")
                return "I apologize, but I couldn't generate a response due to safety guidelines."
            
        except Exception as e:
            logger.error(f"âŒ Error generating Gemini response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties."
                
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
        temperature: float = 0.6,          # balanced for facts + warmth
    ) -> str:
        """
        Generate a human-like response that always blends PDF facts with a friendly tone.
        """
        if not query or len(query.strip()) < 2:
            return "Could you please clarify your question? I'd love to help!"

        logger.info("Using unified RAG prompt for seamless blending.")

        prompt = self._create_unified_rag_prompt(
            context, query, business_name, conversation_history
        )
        return self.generate_response(prompt, temperature=temperature, max_tokens=512)

    
    def _create_unified_rag_prompt(
        self,
        context: str,
        query: str,
        business_name: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Best RAG prompt - forces LLM to use PDF context"""

        # Build conversation history if exists
        history_section = ""
        if conversation_history and len(conversation_history) > 0:
            history_section = "RECENT CONVERSATION:\n"
            for msg in conversation_history[-3:]:
                role = "Customer" if msg.get("role") in ("user", "customer") else "Assistant"
                content = msg.get('content', '')[:150]
                history_section += f"{role}: {content}\n"
            history_section += "\n"

        # Build knowledge base section with strong markers
        knowledge_section = ""
        if context and context.strip():
            knowledge_section = f"""ðŸ“š OFFICIAL INFORMATION FROM {business_name.upper()}:
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
{context.strip()}
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

"""
        else:
            # No.t context available
            return f"""You are {business_name}'s assistant but you don't have access to specific information right now.

Customer asks: "{query}"

Respond EXACTLY like this: "I don't have that specific information in my current knowledge base. Please contact {business_name} directly for accurate details. Is there anything else I can help with?"

Your response:"""

        # Main prompt with clear instructions
        return f"""You are {business_name}'s official AI assistant.

{history_section}{knowledge_section}
ðŸŽ¯ YOUR TASK:
Answer the customer's question using ONLY the official information above.

ðŸ“‹ STRICT RULES:
1. Quote facts from the official information directly
2. Answer in 2-3 sentences maximum
3. Be friendly and professional
4. Add one relevant emoji
5. If the official information doesn't contain the answer, say: "I don't have that specific detail. Please contact {business_name} directly!"

âœ… EXAMPLE:
Customer: "What are your hours?"
You: "We're open Monday-Saturday, 9:00 AM to 8:00 PM! ðŸ˜Š Emergency care is available 24/7."

â“ CUSTOMER QUESTION: {query}

ðŸ’¬ YOUR ANSWER (use the official information above):""" 
    
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
    
    # UPDATED CODE FOR gemini.py

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
            use_rag_fallback: Whether to suggest RAG (now controlled by chat.py)
            
        Returns:
            AI-generated response
        """
        
        # This logic is now controlled by chat.py
        # If chat.py calls this with use_rag_fallback=True, it means it *thinks*
        # it might be a specific question but RAG is off.
        if use_rag_fallback:
             # This check is now an LLM call, not keywords
            if self._requires_specific_knowledge(query, business_name):
                return f"I'm {business_name}'s AI assistant. For specific questions about {business_name}'s services, policies, or offerings, I need to have the relevant documents in my knowledge base. Otherwise, please contact {business_name} directly for the most accurate information."
        
        # If use_rag_fallback=False, it means RAG was already tried and failed.
        # So, we just answer the question as a general AI.
        
        prompt = f"""You are a helpful and friendly AI assistant for {business_name}.
You are having a general conversation.
DO NOT mention your knowledge base or PDFs. Just answer the question.

Customer question: {query}

Brief, helpful response:"""
        
        return self.generate_response(prompt, temperature=0.7)

    def _requires_specific_knowledge(self, query: str, business_name: str) -> bool:
        """
        Determine if a query likely requires specific business knowledge
        (UPGRADED to use an LLM for better accuracy)
        
        Args:
            query: User's question
            business_name: Name of the business
            
        Returns:
            True if query likely needs specific business context
        """
        logger.info(f"Checking if query needs specific knowledge: '{query}'")
        
        prompt = f"""You are an expert query classifier.
Your task is to determine if a user's question requires specific, internal knowledge about a business (like its hours, prices, services, or policies) or if it's a general greeting or chit-chat.

Respond with ONLY 'True' (if it needs specific info) or 'False' (if it's general).

Business Name: {business_name}

-- Examples of 'True' (needs specific info) --
'What are your hours?'
'How much does a consultation cost?'
'timings'
'address'
'Do you offer service X?'

-- Examples of 'False' (general chat) --
'Hi'
'How are you?'
'Thanks'
'Tell me a joke'

-- User Query --
Query: "{query}"

Classification (True/False):"""

        try:
            response = self.generate_response(prompt, temperature=0.0, max_tokens=10)
            result = response.strip().lower()
            logger.info(f"Knowledge check for '{query}': {result}")
            return result == 'true'
        except Exception as e:
            logger.error(f"Failed knowledge check: {e}")
            # Fail safe: assume it's general chat
            return False

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
                clean_line = line.lstrip('0123456789.-â€¢) ').strip()
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
            logger.error(f"âŒ Error classifying intent: {str(e)}")
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
            logger.error(f"âŒ Gemini health check failed: {str(e)}")
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
                logger.debug(f"âœ… Generated embedding {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"âŒ Failed to generate embedding for text {i+1}: {str(e)}")
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
                logger.error(f"âŒ Async embedding failed for text {i+1}: {str(emb)}")
                final_embeddings.append([0.0] * self.embedding_dimension)
            else:
                final_embeddings.append(emb)
        
        return final_embeddings

# =========================
# FAST INTENT CLASSIFIER
# =========================

FAST_INTENT_PATTERNS = {
    "greeting": [
        r"\b(hi|hello|hey|good morning|good evening)\b",
    ],
    "goodbye": [
        r"\b(bye|goodbye|see you|thanks|thank you)\b",
    ],
    "booking": [
        r"\b(book|appointment|schedule|reschedule|cancel appointment)\b",
    ],
    "billing": [
        r"\b(bill|billing|price|cost|refund|payment|invoice)\b",
    ],
    "complaint": [
        r"\b(not working|complaint|issue|problem|bad service|angry)\b",
    ],
    "human_handoff": [
        r"\b(agent|human|representative|support person)\b",
    ],
}

def fast_intent_classify(text: str) -> Optional[Dict[str, Any]]:
    text_l = text.lower()
    for intent, patterns in FAST_INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, text_l):
                return {
                    "intent": intent,
                    "confidence": 0.85,
                    "source": "fast_classifier",
                }
    return None


# =========================
# SYSTEM PROMPT (PERSONA)
# =========================

def build_system_prompt(business_name: str) -> str:
    return f"""
You are the official AI voice assistant for {business_name}.

RULES:
- Be calm, friendly, and professional.
- Use ONLY provided knowledge when answering factual questions.
- If unsure, say you don’t know and offer to connect to a human.
- Never hallucinate prices, policies, or medical/legal facts.
- Keep answers concise and natural for voice.
""".strip()


# =========================
# STRUCTURED NLU (GEMINI)
# =========================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group()) if match else {}
    except Exception:
        return {}


async def gemini_nlu(
    self,
    query: str,
    business_name: str,
) -> Dict[str, Any]:
    """
    Full NLU analysis via Gemini (JSON-only).
    """
    system = build_system_prompt(business_name)

    prompt = f"""
Analyze the user message and return STRICT JSON.

JSON SCHEMA:
{{
  "intent": "<string>",
  "confidence": <0.0-1.0>,
  "entities": {{ }},
  "sentiment": <float -1.0 to 1.0>,
  "urgency": "low|medium|high"
}}

User message: "{query}"
"""

    response = self.generate_response(
        prompt,
        temperature=0.1,
        max_tokens=300,
        system_message=system,
    )

    data = extract_json_from_text(response)

    return {
        "intent": data.get("intent", "unknown"),
        "confidence": float(data.get("confidence", 0.0)),
        "entities": data.get("entities", {}),
        "sentiment": float(data.get("sentiment", 0.0)),
        "urgency": data.get("urgency", "low"),
        "source": "gemini",
    }


# =========================
# TOKEN-LEVEL STREAMING WRAPPER
# =========================
async def generate_rag_response_async(
    self,
    *,
    user_query: str,
    rag_context: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    intent: str = "question",
    temperature: float = 0.15,
) -> str:
    """
    Generate a single RAG response (non-streaming) for knowledge base queries.
    Optimized for factual accuracy with low temperature.
    """
    try:
        # Build intent-aware system prompt
        system_prompts = {
            "frustrated": (
                "You are a professional, empathetic assistant. "
                "The user is frustrated. Start with a genuine apology, "
                "acknowledge their concern, then provide a clear answer. "
                "Use the CONTEXT below to answer accurately."
            ),
            "urgent": (
                "You are a professional assistant. The user needs urgent information. "
                "Be concise and direct. Skip pleasantries. "
                "Use the CONTEXT below to answer accurately."
            ),
            "confused": (
                "You are a patient, helpful assistant. The user is confused. "
                "Explain step-by-step in simple terms. "
                "Use the CONTEXT below to answer accurately."
            ),
            "greeting": (
                "You are a warm, professional assistant. "
                "Respond to the greeting briefly and offer to help."
            ),
            "question": (
                "You are a professional, knowledgeable assistant. "
                "Use ONLY the CONTEXT below to answer factual questions accurately."
            ),
        }
        
        system_message = system_prompts.get(intent, system_prompts["question"])
        system_message += f"\n\nCONTEXT:\n{rag_context}"
        
        messages = [
            {"role": "system", "parts": [{"text": system_message}]}
        ]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "parts": [{"text": msg.get("text", "")}]
                })
        
        messages.append({
            "role": "user",
            "parts": [{"text": user_query}]
        })
        
        response = await self.model.generate_content_async(
            messages,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 350,
            },
        )
        
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    texts = [
                        part.text 
                        for part in candidate.content.parts 
                        if hasattr(part, 'text')
                    ]
                    return " ".join(texts)
        
        return ""
    
    except Exception as e:
        logger.exception(f"Gemini RAG generation failed: {e}")
        return ""
async def generate_with_functions_async(
    self,
    *,
    user_text: str,
    system_message: str = "",
    available_functions: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    cancel_token = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Generate response with function calling capability.
    Yields dictionaries with 'type' field:
    - {'type': 'text', 'content': str}
    - {'type': 'function_call', 'name': str, 'arguments': dict}
    - {'type': 'error', 'message': str}
    """
    try:
        # Build conversation history
        messages = []
        
        if system_message:
            messages.append({
                "role": "system",
                "parts": [{"text": system_message}]
            })
        
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "parts": [{"text": msg.get("text", "")}]
                })
        
        messages.append({
            "role": "user",
            "parts": [{"text": user_text}]
        })
        
        # Configure generation with function declarations
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Add function declarations if provided
        tools = None
        if available_functions:
            tools = [{"function_declarations": available_functions}]
        
        # Generate with streaming
        response = await self.model.generate_content_async(
            messages,
            generation_config=generation_config,
            tools=tools,
            stream=True,
        )
        
        async for chunk in response:
            if cancel_token and cancel_token.is_cancelled():
                break
            
            if not hasattr(chunk, 'candidates') or not chunk.candidates:
                continue
            
            candidate = chunk.candidates[0]
            
            if not hasattr(candidate, 'content') or not candidate.content:
                continue
            
            if not hasattr(candidate.content, 'parts'):
                continue
            
            for part in candidate.content.parts:
                # Check for function call
                if hasattr(part, 'function_call') and part.function_call:
                    func_call = part.function_call
                    yield {
                        "type": "function_call",
                        "name": func_call.name,
                        "arguments": dict(func_call.args) if func_call.args else {},
                    }
                
                # Regular text content
                elif hasattr(part, 'text') and part.text:
                    yield {
                        "type": "text",
                        "content": part.text
                    }
        
    except Exception as e:
        logger.exception(f"Gemini function calling generation failed: {e}")
        yield {
            "type": "error",
            "message": str(e)
        }
async def stream_response(
    self,
    *,
    prompt: str,
    system_message: str,
    cancel_event: Optional[asyncio.Event] = None,
):
    """
    Async token-level streaming generator.
    Wraps existing generate_stream().
    """
    try:
        for token in self.generate_stream(
            prompt=prompt,
            system_message=system_message,
        ):
            if cancel_event and cancel_event.is_set():
                break
            yield token
    except Exception as e:
        logger.error(f"Gemini streaming failed: {e}")
        return


# =========================
# UNIFIED NLU ENTRYPOINT
# =========================

async def analyze_user_input(
    self,
    query: str,
    business_name: str,
) -> Dict[str, Any]:
    """
    Fast path → Gemini fallback → confidence fusion.
    """

    # 1️⃣ Fast regex classifier
    fast = fast_intent_classify(query)
    if fast:
        return {
            **fast,
            "entities": {},
            "sentiment": 0.0,
            "urgency": "low",
        }

    # 2️⃣ Gemini structured NLU
    gem = await gemini_nlu(self, query, business_name)

    # 3️⃣ Confidence guardrail
    if gem["confidence"] < 0.6:
        gem["intent"] = "clarification"

    return gem
    
    async def generate_stream_async(
        self,
        *,
        user_text: str,
        system_message: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        rag_context: Optional[str] = None,
        cancel_token = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """
        Streams tokens from Gemini with full context awareness.
        Supports conversation history and RAG context injection.
        """
        try:
            messages = []
        
            # System message with RAG context if available
            enhanced_system = system_message
            if rag_context:
                enhanced_system += f"\n\nRELEVANT CONTEXT:\n{rag_context}"
        
            if enhanced_system:
                messages.append({
                    "role": "system",
                    "parts": [{"text": enhanced_system}]
                })
        
        # Add conversation history for context
            if conversation_history:
                for msg in conversation_history[-6:]:
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    if text:
                        messages.append({
                            "role": role,
                            "parts": [{"text": text}]
                        })
        
        # Current user message
            messages.append({
                "role": "user",
                "parts": [{"text": user_text}]
            })
        
        # Generate with enhanced configuration
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            }
        
            response = await self.model.generate_content_async(
                messages,
                generation_config=generation_config,
                stream=True,
            )
        
            async for chunk in response:
                if cancel_token and cancel_token.is_cancelled():
                    break
            
            # Handle different response formats
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    yield part.text
    
        except Exception as e:
            logger.exception(f"Gemini streaming failed: {e}")
            yield ""

    
# Global singleton instance
gemini_service = GeminiService()
