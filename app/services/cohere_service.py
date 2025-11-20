import os
import logging
import asyncio
from typing import List
import cohere

logger = logging.getLogger(__name__)

class CohereService:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        self.client = None
        self.is_available = False
        
        if not self.api_key:
            logger.error("❌ COHERE_API_KEY not found in environment variables")
            return
            
        try:
            self.client = cohere.Client(self.api_key)
            self.is_available = True
            logger.info("✅ Cohere service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cohere client: {str(e)}")
            self.is_available = False

    async def generate_embedding_async(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding using Cohere's embed-english-v3.0 model (1024 dimensions)
        
        Args:
            text: Text to generate embedding for
            input_type: 'search_document' for storage, 'search_query' for queries
            
        Returns:
            List of 1024 floats representing the embedding vector
        """
        # Input validation
        if not self.is_available:
            logger.error("❌ Cohere service not available")
            return self._get_zero_embedding()
            
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for embedding")
            return self._get_zero_embedding()
        
        # Truncate very long texts (Cohere has limits)
        processed_text = text.strip()
        if len(processed_text) > 4096:  # Cohere's typical limit
            logger.warning(f"⚠️ Truncating long text from {len(processed_text)} to 4096 characters")
            processed_text = processed_text[:4096]
        
        try:
            # Generate embedding with Cohere
            response = self.client.embed(
                texts=[processed_text],
                model="embed-english-v3.0",
                input_type=input_type,
                truncate="END"  # Handle long texts gracefully
            )
            
            embedding = response.embeddings[0]
            
            # Validate embedding
            if not embedding or len(embedding) != 1024:
                logger.error(f"❌ Invalid embedding received: length {len(embedding) if embedding else 0}")
                return self._get_zero_embedding()
            
            logger.debug(f"✅ Cohere embedding generated: {len(embedding)}D (input_type: {input_type})")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Cohere API error: {str(e)}")
            return self._get_zero_embedding()
        except Exception as e:
            logger.error(f"❌ Unexpected error generating Cohere embedding: {str(e)}")
            return self._get_zero_embedding()
    
    async def generate_document_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for document storage
        """
        return await self.generate_embedding_async(text, input_type="search_document")
    
    async def generate_query_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for search queries (optimized for retrieval)
        """
        return await self.generate_embedding_async(text, input_type="search_query")

    
        # --- ADD THIS NEW METHOD TO CohereService CLASS ---
    async def rerank_documents(self, query: str, documents: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of documents to find the MOST relevant ones.
        """
        if not documents or not self.is_available:
            return []
            
        try:
            # Using the existing self.client
            response = self.client.rerank(
                model="rerank-english-v3.0", 
                query=query,
                documents=documents,
                top_n=top_n
            )
            
            reranked = []
            for result in response.results:
                reranked.append({
                    "text": documents[result.index],
                    "score": result.relevance_score,
                    "index": result.index
                })
            
            return reranked
        except Exception as e:
            logger.error(f"❌ Cohere Rerank failed: {e}")
            # Fallback: return original list with dummy scores
            return [{"text": d, "score": 0.5, "index": i} for i, d in enumerate(documents[:top_n])]
            
    def _get_zero_embedding(self) -> List[float]:
        """Return a zero vector of 1024 dimensions as fallback"""
        return [0.0] * 1024
    
    def health_check(self) -> dict:
        """
        Check Cohere service health
        """
        return {
            "service": "cohere",
            "available": self.is_available,
            "has_api_key": bool(self.api_key),
            "status": "healthy" if self.is_available else "unavailable"
        }


# Global instance
cohere_service = CohereService()
