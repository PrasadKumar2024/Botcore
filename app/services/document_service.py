import os
import uuid
import logging
import json
import asyncio
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import PyPDF2
import pdfplumber
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models import Document, Client, KnowledgeChunk
from app.services.gemini_service import gemini_service
from app.services.pinecone_service import pinecone_service

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Comprehensive document processing service with Pinecone integration
    Handles PDF processing, chunking, vector storage, and semantic search
    """
    
    def __init__(self):
        """Initialize document service with dependencies"""
        self.gemini_service = gemini_service
        self.pinecone_service = pinecone_service
        self.max_chunk_size = 1000
        self.chunk_overlap = 200
        self.max_retries = 3
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file using PyPDF2 with pdfplumber fallback
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            logger.info(f"📄 Extracting text from PDF: {file_path}")
            
            # Try PyPDF2 first
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    logger.info(f"Processing {total_pages} pages with PyPDF2")
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text += f"\n--- Page {page_num} ---\n{page_text.strip()}"
                        except Exception as e:
                            logger.warning(f"PyPDF2 error on page {page_num}: {e}")
                            continue
                
                if text.strip():
                    logger.info(f"✅ PyPDF2 extracted {len(text)} characters")
                    return text.strip()
                else:
                    logger.warning("PyPDF2 returned empty text, trying pdfplumber")
                    
            except Exception as pdf2_error:
                logger.warning(f"PyPDF2 failed, trying pdfplumber: {pdf2_error}")
            
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(file_path) as pdf:
                    total_pages = len(pdf.pages)
                    logger.info(f"Processing {total_pages} pages with pdfplumber")
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text += f"\n--- Page {page_num} ---\n{page_text.strip()}"
                        except Exception as e:
                            logger.warning(f"pdfplumber error on page {page_num}: {e}")
                            continue
                
                if text.strip():
                    logger.info(f"✅ pdfplumber extracted {len(text)} characters")
                    return text.strip()
                else:
                    raise Exception("Both PDF libraries failed to extract text")
                    
            except Exception as plumber_error:
                logger.error(f"pdfplumber also failed: {plumber_error}")
                raise Exception(f"Failed to extract text from PDF: {plumber_error}")
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise Exception(f"PDF file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: The text to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunk_size = chunk_size or self.max_chunk_size
        overlap = overlap or self.chunk_overlap
        
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        logger.info(f"Chunking text of length {text_length} with chunk_size={chunk_size}, overlap={overlap}")
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at natural boundaries for better context
            if end < text_length:
                # Look for sentence/paragraph boundaries
                best_break = self._find_optimal_breakpoint(text, start, end)
                if best_break > start:
                    end = best_break
            
            # Extract and clean the chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) > 50:  # Only add meaningful chunks
                chunk_data = {
                    "chunk_text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
                chunks.append(chunk_data)
                logger.debug(f"Created chunk {len(chunks)}: {len(chunk_text)} characters")
            
            # Move start position with overlap
            start = end - overlap if end < text_length else text_length
        
        logger.info(f"✅ Created {len(chunks)} chunks from text")
        return chunks
    
    def _find_optimal_breakpoint(self, text: str, start: int, end: int) -> int:
        """
        Find optimal break point for chunking at natural boundaries
        
        Args:
            text: Full text
            start: Start position
            end: End position
            
        Returns:
            Optimal break position
        """
        # Priority order of breakpoints
        breakpoints = [
            ('\n\n', 2),      # Paragraph break
            ('. ', 2),        # Sentence end with space
            ('.\n', 2),       # Sentence end with newline
            ('? ', 2),        # Question end
            ('! ', 2),        # Exclamation end
            ('\n', 1),        # Line break
            ('.', 1),         # Sentence end without space
            ('?', 1),         # Question end
            ('!', 1),         # Exclamation end
            (',', 1),         # Comma
            (';', 1),         # Semicolon
            (' ', 1),         # Space
        ]
        
        # Search from end backwards
        search_start = max(start, end - 100)  # Look in last 100 chars
        
        for delimiter, min_length in breakpoints:
            break_pos = text.rfind(delimiter, search_start, end)
            if break_pos > start and (break_pos + len(delimiter)) - start >= min_length:
                return break_pos + len(delimiter)
        
        return end  # No good break found, use original end
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def process_document_async(self, document_id: str, db: Session) -> Dict[str, Any]:
        """
        Async document processing with Pinecone integration
        
        Args:
            document_id: UUID of the document to process
            db: Database session
            
        Returns:
            Processing results with statistics
        """
        try:
            # Get document from database
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                logger.error(f"Document {document_id} not found")
                return {"success": False, "error": "Document not found"}
            
            logger.info(f"🔄 Processing document: {document.filename} (ID: {document_id})")
            
            # Check if file exists
            if not os.path.exists(document.file_path):
                logger.error(f"File not found: {document.file_path}")
                document.processed = False
                document.processing_error = "File not found"
                db.commit()
                return {"success": False, "error": "File not found"}
            
            # Extract text from PDF
            try:
                text = self.extract_text_from_pdf(document.file_path)
            except Exception as e:
                logger.error(f"Failed to extract text: {e}")
                document.processed = False
                document.processing_error = str(e)
                db.commit()
                return {"success": False, "error": f"Text extraction failed: {e}"}
            
            if not text or len(text.strip()) < 50:
                logger.warning(f"No meaningful text extracted: {len(text or '')} chars")
                document.processed = True
                document.processed_at = datetime.utcnow()
                document.processing_error = "No text content found"
                db.commit()
                return {"success": True, "warning": "No meaningful text found", "chunks_created": 0}
            
            # Split into chunks
            chunks_data = self.chunk_text(text)
            
            if not chunks_data:
                logger.warning("No chunks created from document")
                document.processed = True
                document.processed_at = datetime.utcnow()
                document.processing_error = "Failed to create text chunks"
                db.commit()
                return {"success": True, "warning": "No chunks created", "chunks_created": 0}
            
            # Delete existing chunks for this document
            deleted_count = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == document_id
            ).delete()
            
            if deleted_count > 0:
                logger.info(f"🗑️ Deleted {deleted_count} existing chunks")
            
            # Delete existing vectors from Pinecone
            try:
                await self.pinecone_service.delete_document_vectors(str(document_id))
                logger.info(f"🗑️ Deleted existing Pinecone vectors for document {document_id}")
            except Exception as e:
                logger.warning(f"Could not delete Pinecone vectors: {e}")
            
            # Prepare chunks for database and Pinecone
            db_chunks = []
            pinecone_chunks = []
            
            for i, chunk_data in enumerate(chunks_data):
                chunk_id = str(uuid.uuid4())
                
                # Database chunk
                db_chunk = KnowledgeChunk(
                    id=chunk_id,
                    client_id=document.client_id,
                    document_id=document.id,
                    chunk_text=chunk_data["chunk_text"],
                    chunk_index=i,
                    chunk_metadata=json.dumps({
                        "filename": document.filename,
                        "total_chunks": len(chunks_data),
                        "char_count": chunk_data["char_count"],
                        "word_count": chunk_data["word_count"],
                        "chunk_number": i + 1,
                        "start_pos": chunk_data["start_pos"],
                        "end_pos": chunk_data["end_pos"]
                    })
                )
                db_chunks.append(db_chunk)
                
                # Pinecone chunk data
                pinecone_chunk = {
                    "chunk_text": chunk_data["chunk_text"],
                    "metadata": {
                        "chunk_id": chunk_id,
                        "client_id": str(document.client_id),
                        "document_id": str(document.id),
                        "chunk_index": i,
                        "filename": document.filename,
                        "source": "document_upload",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                pinecone_chunks.append(pinecone_chunk)
            
            # Store chunks in database
            for chunk in db_chunks:
                db.add(chunk)
            
            # Store chunks in Pinecone
            pinecone_stored = 0
            if self.pinecone_service.is_configured():
                try:
                    pinecone_stored = await self.pinecone_service.store_knowledge_chunks(
                        client_id=str(document.client_id),
                        chunks=pinecone_chunks
                    )
                    logger.info(f"✅ Stored {pinecone_stored}/{len(pinecone_chunks)} chunks in Pinecone")
                except Exception as e:
                    logger.error(f"❌ Failed to store chunks in Pinecone: {e}")
                    # Continue even if Pinecone fails - database chunks are primary
            
            # Update document status
            document.processed = True
            document.processed_at = datetime.utcnow()
            document.processing_error = None
            db.commit()
            
            logger.info(f"✅ Successfully processed {document.filename}: {len(db_chunks)} DB chunks, {pinecone_stored} Pinecone vectors")
            
            return {
                "success": True,
                "document_id": str(document_id),
                "chunks_created": len(db_chunks),
                "pinecone_stored": pinecone_stored,
                "total_text_length": len(text),
                "filename": document.filename
            }
            
        except Exception as e:
            
            logger.error(f"❌ Error processing document {document_id}: {e}")
            
            # Update document with error
            try:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.processed = False
                    document.processing_error = str(e)
                    db.commit()
            except Exception as update_error:
                logger.error(f"Failed to update document error status: {update_error}")
            
            return {"success": False, "error": str(e)}
    
    def process_document_sync(self, document_id: str, db: Session) -> bool:
        """
        Synchronous wrapper for document processing
        
        Args:
            document_id: UUID of the document to process
            db: Database session
            
        Returns:
            True if processing succeeded
        """
        try:
            # Run async processing in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.process_document_async(document_id, db))
            loop.close()
            
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Error in sync document processing: {e}")
            return False
    
    async def get_relevant_context(self, client_id: str, query: str, db: Session, max_chunks: int = 5) -> str:
        """
        Retrieve relevant context using Pinecone semantic search with database fallback
        
        Args:
            client_id: UUID of the client
            query: Search query
            db: Database session
            max_chunks: Maximum number of chunks to return
            
        Returns:
            Concatenated text from relevant chunks
        """
        try:
            # Try Pinecone semantic search first
            if self.pinecone_service.is_configured():
                try:
                    similar_chunks = await self.pinecone_service.search_similar_chunks(
                        client_id=str(client_id),
                        query=query,
                        top_k=max_chunks,
                        min_score=0.6
                    )
                    
                    if similar_chunks:
                        context_text = "\n\n".join([chunk["chunk_text"] for chunk in similar_chunks])
                        logger.info(f"✅ Found {len(similar_chunks)} relevant chunks via Pinecone")
                        return context_text
                    else:
                        logger.info("No relevant chunks found via Pinecone, falling back to database")
                        
                except Exception as e:
                    logger.warning(f"Pinecone search failed, falling back to database: {e}")
            
            # Fallback to database keyword search
            return self._get_relevant_context_from_db(client_id, query, db, max_chunks)
            
        except Exception as e:
            logger.error(f"Error retrieving context for client {client_id}: {e}")
            return self._get_relevant_context_from_db(client_id, query, db, max_chunks)
    
    def _get_relevant_context_from_db(self, client_id: str, query: str, db: Session, max_chunks: int = 5) -> str:
        """
        Fallback context retrieval using database keyword matching
        
        Args:
            client_id: UUID of the client
            query: Search query
            db: Database session
            max_chunks: Maximum number of chunks to return
            
        Returns:
            Concatenated text from relevant chunks
        """
        try:
            chunks = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.client_id == client_id
            ).all()
            
            if not chunks:
                logger.warning(f"No knowledge chunks found for client {client_id}")
                return ""
            
            # Simple keyword matching
            query_lower = query.lower()
            query_words = [word for word in query_lower.split() if len(word) > 3]
            
            if not query_words:
                return "\n\n".join([chunk.chunk_text for chunk in chunks[:max_chunks]])
            
            scored_chunks = []
            for chunk in chunks:
                score = 0
                chunk_lower = chunk.chunk_text.lower()
                
                for word in query_words:
                    occurrences = chunk_lower.count(word)
                    score += occurrences * 2
                    
                    if query_lower in chunk_lower:
                        score += 10
                
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            if not scored_chunks:
                return "\n\n".join([chunk.chunk_text for chunk in chunks[:max_chunks]])
            
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            top_chunks = [chunk.chunk_text for score, chunk in scored_chunks[:max_chunks]]
            
            return "\n\n".join(top_chunks)
            
        except Exception as e:
            logger.error(f"Error in database context retrieval: {e}")
            return ""
    
    async def reprocess_all_documents(self, client_id: str, db: Session) -> Dict[str, Any]:
        """
        Reprocess all documents for a client with Pinecone integration
        
        Args:
            client_id: UUID of the client
            db: Database session
            
        Returns:
            Reprocessing statistics
        """
        try:
            logger.info(f"🔄 Reprocessing all documents for client {client_id}")
            
            # Delete all existing knowledge chunks for this client
            deleted_count = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.client_id == client_id
            ).delete()
            logger.info(f"🗑️ Deleted {deleted_count} existing DB chunks")
            
            # Delete all vectors for this client from Pinecone
            if self.pinecone_service.is_configured():
                try:
                    await self.pinecone_service.delete_client_vectors(str(client_id))
                    logger.info(f"🗑️ Deleted all Pinecone vectors for client {client_id}")
                except Exception as e:
                    logger.warning(f"Could not delete Pinecone vectors: {e}")
            
            # Get all documents for this client
            documents = db.query(Document).filter(
                Document.client_id == client_id
            ).all()
            
            if not documents:
                logger.warning(f"No documents found for client {client_id}")
                return {
                    "success": True,
                    "documents_processed": 0,
                    "total_documents": 0,
                    "message": "No documents found"
                }
            
            processed_count = 0
            total_chunks = 0
            failed_documents = []
            
            # Process each document
            for document in documents:
                logger.info(f"Processing document: {document.filename}")
                result = await self.process_document_async(str(document.id), db)
                
                if result.get("success"):
                    processed_count += 1
                    total_chunks += result.get("chunks_created", 0)
                else:
                    failed_documents.append({
                        "filename": document.filename,
                        "error": result.get("error", "Unknown error")
                    })
            
            logger.info(f"✅ Reprocessed {processed_count}/{len(documents)} documents, created {total_chunks} chunks")
            
            return {
                "success": True,
                "documents_processed": processed_count,
                "total_documents": len(documents),
                "total_chunks_created": total_chunks,
                "failed_documents": failed_documents
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error reprocessing documents for client {client_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "total_documents": 0
            }
    
    async def sync_chunks_to_pinecone(self, client_id: str, db: Session) -> Dict[str, Any]:
        """
        Sync existing database chunks to Pinecone
        
        Args:
            client_id: UUID of the client
            db: Database session
            
        Returns:
            Sync statistics
        """
        try:
            if not self.pinecone_service.is_configured():
                return {"success": False, "error": "Pinecone not configured"}
            
            # Get all chunks for this client
            chunks = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.client_id == client_id
            ).all()
            
            if not chunks:
                return {"success": True, "message": "No chunks to sync", "chunks_synced": 0}
            
            logger.info(f"🔄 Syncing {len(chunks)} database chunks to Pinecone for client {client_id}")
            
            # Prepare chunks for Pinecone
            pinecone_chunks = []
            for chunk in chunks:
                pinecone_chunk = {
                    "chunk_text": chunk.chunk_text,
                    "metadata": {
                        "chunk_id": str(chunk.id),
                        "client_id": str(client_id),
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "source": "database_sync",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                pinecone_chunks.append(pinecone_chunk)
            
            # Store in Pinecone
            stored_count = await self.pinecone_service.store_knowledge_chunks(
                client_id=str(client_id),
                chunks=pinecone_chunks
            )
            
            logger.info(f"✅ Synced {stored_count}/{len(chunks)} chunks to Pinecone")
            
            return {
                "success": True,
                "chunks_synced": stored_count,
                "total_chunks": len(chunks),
                "client_id": str(client_id)
            }
            
        except Exception as e:
            logger.error(f"❌ Error syncing chunks to Pinecone: {e}")
            return {"success": False, "error": str(e)}
    
    def get_document_stats(self, client_id: str, db: Session) -> Dict[str, Any]:
        """
        Get comprehensive document statistics including Pinecone data
        
        Args:
            client_id: UUID of the client
            db: Database session
            
        Returns:
            Dictionary with document statistics
        """
        try:
            # Database statistics
            total_documents = db.query(Document).filter(
                Document.client_id == client_id
            ).count()
            
            processed_documents = db.query(Document).filter(
                Document.client_id == client_id,
                Document.processed == True
            ).count()
            
            total_chunks = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.client_id == client_id
            ).count()
            
            total_size = db.query(Document).filter(
                Document.client_id == client_id
            ).with_entities(db.func.sum(Document.file_size)).scalar() or 0
            
            # Pinecone statistics
            pinecone_stats = {}
            if self.pinecone_service.is_configured():
                try:
                    pinecone_stats = self.pinecone_service.get_index_stats()
                    pinecone_vector_count = asyncio.run(
                        self.pinecone_service.get_client_vector_count(str(client_id))
                    )
                    pinecone_stats["client_vectors"] = pinecone_vector_count
                except Exception as e:
                    logger.warning(f"Could not get Pinecone stats: {e}")
                    pinecone_stats = {"error": str(e)}
            
            return {
                "database": {
                    "total_documents": total_documents,
                    "processed_documents": processed_documents,
                    "pending_documents": total_documents - processed_documents,
                    "total_chunks": total_chunks,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                },
                "pinecone": pinecone_stats,
                "services": {
                    "pinecone_configured": self.pinecone_service.is_configured(),
                    "gemini_configured": self.gemini_service.check_availability()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats for client {client_id}: {e}")
            return {
                "database": {
                    "total_documents": 0,
                    "processed_documents": 0,
                    "pending_documents": 0,
                    "total_chunks": 0,
                    "total_size_bytes": 0,
                    "total_size_mb": 0
                },
                "pinecone": {"error": str(e)},
                "services": {
                    "pinecone_configured": False,
                    "gemini_configured": False
                }
            }
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def validate_and_save_pdf(self, file, client_id: str, upload_dir: str, original_filename: str) -> str:
        """
        Validate and save uploaded PDF file with enhanced validation
        
        Args:
            file: Uploaded file object
            client_id: UUID of the client
            upload_dir: Directory to save uploads
            original_filename: Original name of the file
            
        Returns:
            Path to saved file
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate file extension
            if not original_filename.lower().endswith('.pdf'):
                raise ValueError("Only PDF files are allowed")
            
            # Validate file size (max 50MB)
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to start
            
            if file_size > 50 * 1024 * 1024:  # 50MB
                raise ValueError("File size must be less than 50MB")
            
            if file_size == 0:
                raise ValueError("File is empty")
            
            # Generate unique filename
            file_extension = os.path.splitext(original_filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Ensure upload directory exists
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
            
            logger.info(f"💾 Saved file: {file_path} ({len(content)} bytes)")
            
            # Validate PDF content
            try:
                with open(file_path, 'rb') as f:
                    # Try PyPDF2 first
                    try:
                        pdf_reader = PyPDF2.PdfReader(f)
                        num_pages = len(pdf_reader.pages)
                        logger.info(f"✅ PDF validated with PyPDF2: {num_pages} pages")
                    except Exception:
                        # Try pdfplumber as fallback
                        f.seek(0)
                        with pdfplumber.open(f) as pdf:
                            num_pages = len(pdf.pages)
                        logger.info(f"✅ PDF validated with pdfplumber: {num_pages} pages")
                        
            except Exception as e:
                # Delete invalid file
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving PDF file: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for document service
        
        Returns:
            Health status dictionary
        """
        try:
            # Check dependencies
            pinecone_health = await self.pinecone_service.health_check()
            gemini_health = await self.gemini_service.health_check()
            
            health_status = {
                "service": "document_processor",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "dependencies": {
                    "pinecone": pinecone_health,
                    "gemini": gemini_health
                },
                "configuration": {
                    "max_chunk_size": self.max_chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "max_retries": self.max_retries
                }
            }
            
            # Determine overall status
            if not pinecone_health.get("healthy") or not gemini_health.get("healthy"):
                health_status["status"] = "degraded"
                health_status["issues"] = "Some dependencies are unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Document service health check failed: {e}")
            return {
                "service": "document_processor",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global singleton instance
document_service = DocumentService()
