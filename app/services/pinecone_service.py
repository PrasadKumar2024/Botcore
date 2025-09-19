import os
import logging
import uuid
from typing import List, Dict, Optional, Any
import pinecone
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeService:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ownbot-index")
        self.dimension = 768  # Gemini embedding dimension
        self.metric = "cosine"
        self.pod_type = "p1"
        
        # Initialize Pinecone
        self.init_pinecone()
    
    def init_pinecone(self):
        """Initialize Pinecone connection and ensure index exists"""
        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists, create if not
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    pod_type=self.pod_type
                )
            
            # Connect to the index
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise Exception(f"Pinecone initialization failed: {str(e)}")
    
    async def store_embeddings(self, client_id: str, document_id: str, vectors: List[Dict]) -> Dict:
        """
        Store embeddings in Pinecone for a specific client and document
        
        Args:
            client_id: Unique identifier for the client
            document_id: Unique identifier for the document
            vectors: List of vectors with embeddings and metadata
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Prepare vectors for upsert
            pinecone_vectors = []
            for vector in vectors:
                pinecone_vectors.append({
                    "id": vector["id"],
                    "values": vector["values"],
                    "metadata": {
                        **vector["metadata"],
                        "stored_at": datetime.utcnow().isoformat()
                    }
                })
            
            # Upsert vectors to Pinecone
            upsert_response = self.index.upsert(vectors=pinecone_vectors)
            
            logger.info(f"Stored {len(pinecone_vectors)} vectors for client {client_id}, document {document_id}")
            
            return {
                "success": True,
                "upserted_count": upsert_response.get("upserted_count", len(pinecone_vectors)),
                "client_id": client_id,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "client_id": client_id,
                "document_id": document_id
            }
    
    async def query_embeddings(self, query_embedding: List[float], client_id: str, 
                              top_k: int = 5, filter: Optional[Dict] = None) -> Dict:
        """
        Query Pinecone for similar embeddings
        
        Args:
            query_embedding: The embedding vector to query with
            client_id: Filter results by client ID
            top_k: Number of results to return
            filter: Additional filter criteria
            
        Returns:
            Dictionary with query results
        """
        try:
            # Build filter - always filter by client_id
            query_filter = {"client_id": {"$eq": client_id}}
            
            # Add additional filters if provided
            if filter:
                query_filter = {**query_filter, **filter}
            
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                filter=query_filter
            )
            
            # Extract matches
            matches = []
            for match in query_response.get("matches", []):
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return {
                "success": True,
                "matches": matches,
                "query_count": len(matches)
            }
            
        except Exception as e:
            logger.error(f"Error querying embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_embeddings(self, client_id: str, document_id: Optional[str] = None) -> Dict:
        """
        Delete embeddings from Pinecone for a client or specific document
        
        Args:
            client_id: Client identifier
            document_id: Optional document identifier (delete all if not provided)
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Build filter
            if document_id:
                filter = {"client_id": {"$eq": client_id}, "document_id": {"$eq": document_id}}
            else:
                filter = {"client_id": {"$eq": client_id}}
            
            # Delete vectors
            delete_response = self.index.delete(filter=filter)
            
            logger.info(f"Deleted embeddings for client {client_id}" + 
                       (f", document {document_id}" if document_id else ""))
            
            return {
                "success": True,
                "deleted_count": delete_response.get("deleted_count", 0),
                "client_id": client_id,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "client_id": client_id,
                "document_id": document_id
            }
    
    async def get_document_stats(self, client_id: str, document_id: str) -> Dict:
        """
        Get statistics about a document's embeddings in Pinecone
        
        Args:
            client_id: Client identifier
            document_id: Document identifier
            
        Returns:
            Dictionary with document statistics
        """
        try:
            # Query for all vectors from this document
            filter = {"client_id": {"$eq": client_id}, "document_id": {"$eq": document_id}}
            
            # Use a zero vector query to get all matches (not efficient for large sets)
            query_response = self.index.query(
                vector=[0] * self.dimension,  # Zero vector
                top_k=10000,  # Max allowed by Pinecone
                include_values=False,
                include_metadata=True,
                filter=filter
            )
            
            # Extract basic stats
            matches = query_response.get("matches", [])
            chunk_count = len(matches)
            
            # Calculate average score (not meaningful for zero vector, but we have the data)
            total_score = sum(match.score for match in matches)
            avg_score = total_score / chunk_count if chunk_count > 0 else 0
            
            # Extract unique chunk indices
            chunk_indices = set()
            for match in matches:
                if "chunk_index" in match.metadata:
                    chunk_indices.add(match.metadata["chunk_index"])
            
            return {
                "success": True,
                "chunk_count": chunk_count,
                "unique_chunks": len(chunk_indices),
                "avg_score": avg_score,
                "client_id": client_id,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "client_id": client_id,
                "document_id": document_id
            }
    
    async def get_client_stats(self, client_id: str) -> Dict:
        """
        Get statistics about all embeddings for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with client statistics
        """
        try:
            # Query for all vectors from this client
            filter = {"client_id": {"$eq": client_id}}
            
            # Use a zero vector query to get all matches
            query_response = self.index.query(
                vector=[0] * self.dimension,  # Zero vector
                top_k=10000,  # Max allowed by Pinecone
                include_values=False,
                include_metadata=True,
                filter=filter
            )
            
            # Extract basic stats
            matches = query_response.get("matches", [])
            total_chunks = len(matches)
            
            # Count unique documents
            document_ids = set()
            for match in matches:
                if "document_id" in match.metadata:
                    document_ids.add(match.metadata["document_id"])
            
            return {
                "success": True,
                "total_chunks": total_chunks,
                "document_count": len(document_ids),
                "client_id": client_id
            }
            
        except Exception as e:
            logger.error(f"Error getting client stats: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "client_id": client_id
            }
    
    async def check_health(self) -> Dict:
        """
        Check Pinecone service health
        
        Returns:
            Dictionary with health status
        """
        try:
            # Get index stats to verify connection
            stats = self.index.describe_index_stats()
            
            return {
                "success": True,
                "status": "healthy",
                "index_stats": stats,
                "index_name": self.index_name,
                "dimension": self.dimension
            }
            
        except Exception as e:
            logger.error(f"Pinecone health check failed: {str(e)}")
            return {
                "success": False,
                "status": "unhealthy",
                "error": str(e)
            }

# Global instance
pinecone_service = PineconeService()
