import os
import logging
from typing import List, Dict, Any, Optional
import uuid
from pinecone import Pinecone, ServerlessSpec
import json
import time

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
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {str(e)}")
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
        return self.pc is not None and self.index is not None and self.api_key is not None
    
    def generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding for text (placeholder for actual embedding service)
        In production, use a proper embedding model like Gemini Embeddings or OpenAI
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        # This is a placeholder - replace with actual embedding generation
        # For now, we'll use a simple hash-based approach
        try:
            # Normalize text
            text_lower = text.lower().strip()
            
            # Create a simple deterministic embedding based on text
            # In production, replace this with actual embedding API call
            embedding = [0.0] * self.dimension
            
            # Simple character-based feature generation
            for i, char in enumerate(text_lower[:self.dimension]):
                embedding[i % self.dimension] += ord(char) / 1000.0
            
            # Normalize the vector
            magnitude = sum(x**2 for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
    
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
            
            # Generate embedding for the chunk text
            embedding = self.generate_simple_embedding(chunk_text)
            
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
                    
                    if not chunk_text:
                        logger.warning("⚠️ Skipping empty chunk")
                        continue
                    
                    # Generate unique vector ID
                    document_id = metadata.get("document_id", str(uuid.uuid4()))
                    chunk_id = metadata.get("chunk_id", str(uuid.uuid4()))
                    chunk_index = metadata.get("chunk_index", 0)
                    
                    vector_id = f"{client_id}_{document_id}_{chunk_id}"
                    
                    # Generate embedding
                    embedding = self.generate_simple_embedding(chunk_text)
                    
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
            
            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                try:
                    self.index.upsert(vectors=batch)
                    stored_count += len(batch)
                    logger.info(f"✅ Stored batch {i//batch_size + 1}: {len(batch)} vectors")
                    
                except Exception as batch_error:
                    logger.error(f"❌ Error storing batch: {str(batch_error)}")
            
            logger.info(f"✅ Successfully stored {stored_count}/{len(chunks)} chunks in Pinecone")
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Error in batch storage: {str(e)}")
            return 0
    
    async def search_similar_chunks(
        self,
        client_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic search
        
        Args:
            client_id: Client UUID to filter results
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching chunks with metadata and scores
        """
        try:
            if not self.is_configured():
                logger.error("❌ Pinecone not configured")
                return []
            
            # Generate embedding for query
            query_embedding = self.generate_simple_embedding(query)
            
            logger.info(f"🔍 Searching Pinecone for client {client_id} with query: {query[:50]}...")
            
            # Query Pinecone with client filter
            results = self.index.query(
                vector=query_embedding,
                filter={
                    "client_id": {"$eq": str(client_id)}
                },
                top_k=top_k,
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
                        "metadata": match.metadata
                    }
                    similar_chunks.append(chunk_data)
            
            logger.info(f"✅ Found {len(similar_chunks)} relevant chunks (threshold: {min_score})")
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching Pinecone: {str(e)}")
            return []
    
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
                "namespaces": stats.namespaces if hasattr(stats, 'namespaces') else {}
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
            
            # Query with filter and top_k=1 just to get count
            # Note: Pinecone doesn't have direct count API, so we approximate
            results = self.index.query(
                vector=[0.0] * self.dimension,
                filter={
                    "client_id": {"$eq": str(client_id)}
                },
                top_k=10000,
                include_metadata=False
            )
            
            return len(results.matches)
            
        except Exception as e:
            logger.error(f"❌ Error getting client vector count: {str(e)}")
            return 0
    
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
