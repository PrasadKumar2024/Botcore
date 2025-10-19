
import os
import uuid
import logging
import json
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import PyPDF2

from app.models import Document, Client, KnowledgeChunk
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for handling document operations including upload, processing, and knowledge base management
    """
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from PDF file using PyPDF2
        Returns the complete text content from all pages
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"Extracting text from {total_pages} pages in {file_path}")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
                
                logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
                return text.strip()
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise Exception(f"PDF file not found: {file_path}")
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Invalid PDF file {file_path}: {e}")
            raise Exception(f"Invalid or corrupted PDF file: {e}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation
        
        Args:
            text: The text to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        logger.info(f"Chunking text of length {text_length} with chunk_size={chunk_size}, overlap={overlap}")
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary for better context
            if end < text_length:
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                
                # Search for sentence endings in order of preference
                best_break = -1
                for delimiter in ['. ', '.\n', '? ', '! ', '\n\n']:
                    break_pos = text.rfind(delimiter, search_start, end)
                    if break_pos > start:
                        best_break = break_pos + len(delimiter)
                        break
                
                if best_break > start:
                    end = best_break
            
            # Extract and clean the chunk
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} characters")
            
            # Move start position with overlap
            start = end - overlap if end < text_length else text_length
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    @staticmethod
    def process_document(document_id: str, db: Session) -> bool:
        """
        Process a document: extract text, chunk it, and store in knowledge base
        
        Args:
            document_id: UUID of the document to process
            db: Database session
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get document from database
            document = db.query(Document).filter(Document.id == document_id).first()
            
            if not document:
                logger.error(f"Document {document_id} not found in database")
                return False
            
            logger.info(f"Processing document: {document.filename} (ID: {document_id})")
            
            # Check if file exists
            if not os.path.exists(document.file_path):
                logger.error(f"File not found at path: {document.file_path}")
                document.processed = False
                document.processing_error = "File not found"
                db.commit()
                return False
            
            # Extract text from PDF
            try:
                text = DocumentService.extract_text_from_pdf(document.file_path)
            except Exception as e:
                logger.error(f"Failed to extract text from {document.filename}: {e}")
                document.processed = False
                document.processing_error = str(e)
                db.commit()
                return False
            
            if not text or len(text.strip()) < 10:
                logger.warning(f"No meaningful text extracted from {document.filename}")
                document.processed = True
                document.processed_at = datetime.utcnow()
                document.processing_error = "No text content found"
                db.commit()
                return False
            
            # Split into chunks
            chunks = DocumentService.chunk_text(text)
            
            if not chunks:
                logger.warning(f"No chunks created from {document.filename}")
                document.processed = True
                document.processed_at = datetime.utcnow()
                document.processing_error = "Failed to create text chunks"
                db.commit()
                return False
            
            # Delete existing chunks for this document (in case of reprocessing)
            deleted_count = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == document_id
            ).delete()
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} existing chunks for document {document_id}")
            
            # Store chunks in database
            for i, chunk_text in enumerate(chunks):
                chunk = KnowledgeChunk(
                    id=str(uuid.uuid4()),
                    client_id=document.client_id,
                    document_id=document.id,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    chunk_metadata=json.dumps({
                        "filename": document.filename,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk_text),
                        "chunk_number": i + 1
                    })
                )
                db.add(chunk)
            
            # Mark document as processed
            document.processed = True
            document.processed_at = datetime.utcnow()
            document.processing_error = None
            
            db.commit()
            
            logger.info(f"✅ Successfully processed {document.filename}: {len(chunks)} chunks created")
            return True
            
        except Exception as e:
            db.rollback()
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
            
            return False
    
    @staticmethod
    def get_relevant_context(client_id: str, query: str, db: Session, max_chunks: int = 5) -> str:
        """
        Retrieve relevant text chunks for a query using keyword matching
        
        Args:
            client_id: UUID of the client
            query: Search query
            db: Database session
            max_chunks: Maximum number of chunks to return
            
        Returns:
            Concatenated text from relevant chunks
        """
        try:
            # Get all chunks for this client
            chunks = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.client_id == client_id
            ).all()
            
            if not chunks:
                logger.warning(f"No knowledge chunks found for client {client_id}")
                return ""
            
            logger.info(f"Searching {len(chunks)} chunks for client {client_id}")
            
            # Extract meaningful keywords from query (words longer than 3 characters)
            query_lower = query.lower()
            query_words = [word for word in query_lower.split() if len(word) > 3]
            
            if not query_words:
                # If no meaningful words, return first few chunks
                logger.info("No meaningful keywords found, returning first chunks")
                return "\n\n".join([chunk.chunk_text for chunk in chunks[:max_chunks]])
            
            # Score each chunk based on keyword matches
            scored_chunks = []
            
            for chunk in chunks:
                score = 0
                chunk_lower = chunk.chunk_text.lower()
                
                for word in query_words:
                    # Count occurrences of each keyword
                    occurrences = chunk_lower.count(word)
                    score += occurrences * 2  # Weight for individual word matches
                    
                    # Bonus for exact query phrase match
                    if query_lower in chunk_lower:
                        score += 10
                
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            if not scored_chunks:
                # No matches found, return first chunks as fallback
                logger.info("No keyword matches found, returning first chunks as fallback")
                return "\n\n".join([chunk.chunk_text for chunk in chunks[:max_chunks]])
            
            # Sort by score (highest first) and get top chunks
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            top_chunks = [chunk.chunk_text for score, chunk in scored_chunks[:max_chunks]]
            
            logger.info(f"Found {len(scored_chunks)} matching chunks, returning top {len(top_chunks)}")
            
            return "\n\n".join(top_chunks)
            
        except Exception as e:
            logger.error(f"Error retrieving context for client {client_id}: {e}")
            return ""
    
    @staticmethod
    def reprocess_all_documents(client_id: str, db: Session) -> int:
        """
        Reprocess all documents for a client
        
        Args:
            client_id: UUID of the client
            db: Database session
            
        Returns:
            Number of documents successfully processed
        """
        try:
            logger.info(f"Reprocessing all documents for client {client_id}")
            
            # Delete all existing knowledge chunks for this client
            deleted_count = db.query(KnowledgeChunk).filter(
                KnowledgeChunk.client_id == client_id
            ).delete()
            
            logger.info(f"Deleted {deleted_count} existing chunks for client {client_id}")
            
            # Get all documents for this client
            documents = db.query(Document).filter(
                Document.client_id == client_id
            ).all()
            
            if not documents:
                logger.warning(f"No documents found for client {client_id}")
                return 0
            
            processed_count = 0
            
            # Process each document
            for document in documents:
                logger.info(f"Reprocessing document: {document.filename}")
                if DocumentService.process_document(str(document.id), db):
                    processed_count += 1
            
            logger.info(f"✅ Reprocessed {processed_count}/{len(documents)} documents for client {client_id}")
            return processed_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error reprocessing documents for client {client_id}: {e}")
            return 0
    
    @staticmethod
    def get_document_stats(client_id: str, db: Session) -> Dict:
        """
        Get statistics about documents and knowledge chunks for a client
        
        Args:
            client_id: UUID of the client
            db: Database session
            
        Returns:
            Dictionary with document statistics
        """
        try:
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
            
            return {
                "total_documents": total_documents,
                "processed_documents": processed_documents,
                "pending_documents": total_documents - processed_documents,
                "total_chunks": total_chunks,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats for client {client_id}: {e}")
            return {
                "total_documents": 0,
                "processed_documents": 0,
                "pending_documents": 0,
                "total_chunks": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0
            }
    
    @staticmethod
    def validate_and_save_pdf(file, client_id: str, upload_dir: str, original_filename: str) -> Optional[str]:
        """
        Validate and save uploaded PDF file
        
        Args:
            file: Uploaded file object
            client_id: UUID of the client
            upload_dir: Directory to save uploads
            original_filename: Original name of the file
            
        Returns:
            Path to saved file or None if validation failed
        """
        try:
            # Validate file extension
            if not original_filename.lower().endswith('.pdf'):
                raise ValueError("Invalid file type. Only PDF files are allowed.")
            
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
            
            logger.info(f"Saved file: {file_path} ({len(content)} bytes)")
            
            # Verify file is a valid PDF
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                    logger.info(f"Validated PDF: {num_pages} pages")
            except Exception as e:
                # Delete invalid file
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise ValueError(f"Invalid PDF file: {str(e)}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving PDF file: {e}")
            raise

# Singleton instance
document_service = DocumentService()
