import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
import uuid
from pinecone import Pinecone, ServerlessSpec
import json
import time
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class PineconeService:
    """
    Service for managing vector embeddings and semantic search with Pinecone
    Handles storing document chunks as vectors and retrieving relevant context
    """
    
    def __init__(self):
        """Initialize Pinecone connection"""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        self.index_name = os.getenv("PINECONE_INDEX", "botcore-knowledge")
        self.dimension = 768  # Standard embedding dimension
        self.metric = "cosine"  # Similarity metric
        
        self.pc = None
        self.index = None
        self._initialized = False
        
        # Initialize connection
        self.initialize()
    
    def initialize(self):
        """Initialize Pinecone client and index"""
        try:
            if not self.api_key:
                logger.error("❌ PINECONE_API_KEY not found in environment variables")
                raise ValueError("Pinecone API key is required")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("✅ Pinecone client initialized")
            
            # Create or connect to index
            self._ensure_index_exists()
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"📊 Pinecone index stats: {stats.total_vector_count} vectors")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {str(e)}")
            self._initialized = False
            raise
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            # Get list of existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"📝 Creating new Pinecone index: {self.index_name}")
                
                # Create serverless index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                logger.info("⏳ Waiting for index to be ready...")
                time.sleep(10)
                
                logger.info(f"✅ Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"✅ Pinecone index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"❌ Error ensuring index exists: {str(e)}")
            raise
    
    def is_configured(self) -> bool:
        """Check if Pinecone is properly configured"""
        return self._initialized and self.pc is not None and self.index is not None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Gemini's embedding model
        This provides MUCH better semantic search than simple embeddings
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Import Gemini service for embeddings
            from app.services.gemini_service import gemini_service
            
            # Use Gemini to generate actual embeddings
            embedding = gemini_service.generate_embedding(text)
            
            if embedding and len(embedding) > 0:
                logger.debug(f"✅ Generated Gemini embedding with dimension: {len(embedding)}")
                return embedding
            else:
                logger.warning("❌ Gemini embedding failed, using fallback")
                return self._generate_fallback_embedding(text)
                
        except ImportError as e:
            logger.error(f"❌ Gemini service not available: {str(e)}")
            return self._generate_fallback_embedding(text)
        except Exception as e:
            logger.error(f"❌ Error generating Gemini embedding: {str(e)}")
            # Fallback to simple embedding
            return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding if Gemini fails"""
        try:
            # More sophisticated fallback embedding
            embedding = [0.0] * self.dimension
            text_lower = text.lower().strip()
            
            if not text_lower:
                return embedding
            
            # Use word frequency and character distribution
            words = text_lower.split()
            for i, word in enumerate(words[:100]):  # Limit words
                for j, char in enumerate(word[:10]):  # Limit chars per word
                    idx = (i * 10 + j) % self.dimension
                    embedding[idx] += (ord(char) * (i + 1)) / 10000.0
            
            # Normalize
            magnitude = sum(x**2 for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            
            logger.debug("✅ Generated fallback embedding")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error in fallback embedding: {str(e)}")
            # Return zero vector as last resort
            return [0.0] * self.dimension
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Async wrapper for embedding generation to avoid blocking event loop
        """
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.generate_embedding, text)
            return embedding
        except Exception as e:
            logger.error(f"❌ Error in async embedding generation: {str(e)}")
            return self._generate_fallback_embedding(text)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def store_knowledge_chunk(
        self,
        chunk_id: str,
        client_id: str,
        document_id: str,
        chunk_text: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a single knowledge chunk in Pinecone
        
        Args:
            chunk_id: Unique ID for the chunk (from database)
            client_id: Client UUID
            document_id: Document UUID
            chunk_text: The text content of the chunk
            chunk_index: Index of chunk in document
            metadata: Additional metadata
            
        Returns:
            Vector ID in Pinecone
        """
        try:
            if not self.is_configured():
                logger.error("❌ Pinecone not configured")
                return None
            
            # Generate vector ID
            vector_id = f"{client_id}_{document_id}_{chunk_id}"
            
            # Generate embedding for the chunk text (async)
            embedding = await self.generate_embedding_async(chunk_text)
            
            # Prepare metadata
            vector_metadata = {
                "client_id": str(client_id),
                "document_id": str(document_id),
                "chunk_id": str(chunk_id),
                "chunk_index": chunk_index,
                "chunk_text": chunk_text[:1000],  # Store truncated text for reference
                "text_length": len(chunk_text),
                "timestamp": time.time()
            }
            
            # Add custom metadata if provided
            if metadata:
                vector_metadata.update(metadata)
            
            # Upsert vector to Pinecone
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                }]
            )
            
            logger.info(f"✅ Stored chunk in Pinecone: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"❌ Error storing chunk in Pinecone: {str(e)}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def store_knowledge_chunks(
        self,
        client_id: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Store multiple knowledge chunks in Pinecone (batch operation)
        
        Args:
            client_id: Client UUID
            chunks: List of chunk dictionaries with chunk_text, metadata, etc.
            
        Returns:
            Number of chunks successfully stored
        """
        try:
            if not self.is_configured():
                logger.error("❌ Pinecone not configured")
                return 0
            
            vectors = []
            stored_count = 0
            
            logger.info(f"📦 Preparing {len(chunks)} chunks for Pinecone storage...")
            
            for chunk_data in chunks:
                try:
                    chunk_text = chunk_data.get("chunk_text", "")
                    metadata = chunk_data.get("metadata", {})
                    
                    if not chunk_text or len(chunk_text.strip()) < 10:
                        logger.warning("⚠️ Skipping empty or too short chunk")
                        continue
                    
                    # Generate unique vector ID
                    document_id = metadata.get("document_id", str(uuid.uuid4()))
                    chunk_id = metadata.get("chunk_id", str(uuid.uuid4()))
                    chunk_index = metadata.get("chunk_index", 0)
                    
                    vector_id = f"{client_id}_{document_id}_{chunk_id}"
                    
                    # Generate embedding (async)
                    embedding = await self.generate_embedding_async(chunk_text)
                    
                    # Prepare metadata
                    vector_metadata = {
                        "client_id": str(client_id),
                        "document_id": str(document_id),
                        "chunk_id": str(chunk_id),
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text[:1000],  # Truncated for storage
                        "text_length": len(chunk_text),
                        "source": metadata.get("source", "unknown"),
                        "timestamp": time.time()
                    }
                    
                    # Add to batch
                    vectors.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": vector_metadata
                    })
                    
                except Exception as chunk_error:
                    logger.error(f"❌ Error processing chunk: {str(chunk_error)}")
                    continue
            
            if not vectors:
                logger.warning("⚠️ No valid vectors to store")
                return 0
            
            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                try:
                    self.index.upsert(vectors=batch)
                    stored_count += len(batch)
                    logger.info(f"✅ Stored batch {i//batch_size + 1}: {len(batch)} vectors")
                    
                    # Small delay between batches to avoid rate limiting
                    if i + batch_size < len(vectors):
                        await asyncio.sleep(0.1)
                        
                except Exception as batch_error:
                    logger.error(f"❌ Error storing batch: {str(batch_error)}")
                    # Continue with next batch instead of failing completely
            
            logger.info(f"✅ Successfully stored {stored_count}/{len(chunks)} chunks in Pinecone")
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Error in batch storage: {str(e)}")
            return 0
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def search_similar_chunks(

        self,
        client_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic search
        
        Args:
            client_id: Client UUID to filter results
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            include_metadata: Whether to include full metadata
            
        Returns:
            List of matching chunks with metadata and scores
        """
        try:
            if not self.is_configured():
                logger.error("❌ Pinecone not configured")
                return []
            
            if not query or len(query.strip()) < 2:
                logger.warning("⚠️ Query too short for meaningful search")
                return []
            
            # Generate embedding for query (async)
            query_embedding = await self.generate_embedding_async(query)
            
            logger.info(f"🔍 Searching Pinecone for client {client_id} with query: {query[:50]}...")
            
            # Query Pinecone with client filter
            results = self.index.query(
                vector=query_embedding,
                filter={
                    "client_id": {"$eq": str(client_id)}
                },
                top_k=top_k * 2,  # Get more results to filter by score
                include_metadata=True
            )
            
            # Process and filter results
            similar_chunks = []
            
            for match in results.matches:
                # Only include results above similarity threshold
                if match.score >= min_score:
                    chunk_data = {
                        "chunk_id": match.metadata.get("chunk_id"),
                        "document_id": match.metadata.get("document_id"),
                        "chunk_text": match.metadata.get("chunk_text", ""),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "score": float(match.score),
                        "source": match.metadata.get("source", "unknown"),
                        "text_length": match.metadata.get("text_length", 0),
                    }
                    
                    if include_metadata:
                        chunk_data["metadata"] = match.metadata
                    
                    similar_chunks.append(chunk_data)
            
            # Sort by score descending and limit to top_k
            similar_chunks.sort(key=lambda x: x["score"], reverse=True)
            similar_chunks = similar_chunks[:top_k]
            
            logger.info(f"✅ Found {len(similar_chunks)} relevant chunks (threshold: {min_score})")

            return {
            "success": True,           # ✅ Add success flag
            "matches": similar_chunks  # ✅ Wrap in expected format
             }
            
        except Exception as e:
            logger.error(f"❌ Error searching Pinecone: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def delete_client_vectors(self, client_id: str) -> bool:
        """
        Delete all vectors for a specific client
        
        Args:
            client_id: Client UUID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_configured():
                logger.error("❌ Pinecone not configured")
                return False
            
            logger.info(f"🗑️ Deleting all vectors for client: {client_id}")
            
            # Delete vectors with client_id filter
            self.index.delete(
                filter={
                    "client_id": {"$eq": str(client_id)}
                }
            )
            
            logger.info(f"✅ Deleted all Pinecone vectors for client: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting client vectors: {str(e)}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def delete_document_vectors(self, document_id: str) -> bool:
        """
        Delete all vectors for a specific document
        
        Args:
            document_id: Document UUID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_configured():
                logger.error("❌ Pinecone not configured")
                return False
            
            logger.info(f"🗑️ Deleting vectors for document: {document_id}")
            
            # Delete vectors with document_id filter
            self.index.delete(
                filter={
                    "document_id": {"$eq": str(document_id)}
                }
            )
            
            logger.info(f"✅ Deleted Pinecone vectors for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document vectors: {str(e)}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            if not self.is_configured():
                return {
                    "configured": False,
                    "error": "Pinecone not configured"
                }
            
            stats = self.index.describe_index_stats()
            
            return {
                "configured": True,
                "index_name": self.index_name,
                "total_vectors": stats.total_vector_count,
                "dimension": self.dimension,
                "metric": self.metric,
                "namespaces": stats.namespaces if hasattr(stats, 'namespaces') else {},
                "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting index stats: {str(e)}")
            return {
                "configured": False,
                "error": str(e)
            }
    
    async def get_client_vector_count(self, client_id: str) -> int:
        """
        Get count of vectors for a specific client
        
        Args:
            client_id: Client UUID
            
        Returns:
            Number of vectors for the client
        """
        try:
            if not self.is_configured():
                return 0
            
            # Use approximate count by querying with a zero vector
            # Note: This is an approximation for large datasets
            results = self.index.query(
                vector=[0.0] * self.dimension,
                filter={
                    "client_id": {"$eq": str(client_id)}
                },
                top_k=10000,  # Maximum allowed by Pinecone
                include_metadata=False
            )
            
            return len(results.matches)
            
        except Exception as e:
            logger.error(f"❌ Error getting client vector count: {str(e)}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for Pinecone service
        
        Returns:
            Health status dictionary
        """
        try:
            stats = await self.get_index_stats()
            
            if not stats.get("configured", False):
                return {
                    "healthy": False,
                    "status": "not_configured",
                    "error": stats.get("error", "Unknown error")
                }
            
            # Test a simple operation
            test_embedding = self._generate_fallback_embedding("health check")
            
            return {
                "healthy": True,
                "status": "operational",
                "index_name": self.index_name,
                "total_vectors": stats.get("total_vectors", 0),
                "dimension": self.dimension,
                "initialized": self._initialized
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test Pinecone connection
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            if not self.is_configured():
                return False
            
            # Try to get index stats as connection test
            stats = self.index.describe_index_stats()
            logger.info(f"✅ Pinecone connection test passed. Total vectors: {stats.total_vector_count}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone connection test failed: {str(e)}")
            return False
    
    async def cleanup_old_vectors(self, older_than_days: int = 30) -> int:
        """
        Clean up vectors older than specified days
        
        Args:
            older_than_days: Delete vectors older than this many days
            
        Returns:
            Number of vectors deleted
        """
        try:
            if not self.is_configured():
                return 0
            
            cutoff_timestamp = time.time() - (older_than_days * 24 * 60 * 60)
            
            # Note: This is a simplified implementation
            # In production, you might need to scan and delete in batches
            logger.info(f"🔄 Cleaning up vectors older than {older_than_days} days...")
            
            # This would require a more complex implementation with scanning
            # For now, just log the intention
            logger.warning("⚠️ Vector cleanup requires batch scanning implementation")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old vectors: {str(e)}")
            return 0


# Global instance
pinecone_service = PineconeService()
