import os
import uuid
import logging
from sqlalchemy.orm import Session
from typing import List, Optional
import pdfplumber
from PyPDF2 import PdfReader
import PyPDF2
from datetime import datetime

from app.models import Document, Client
from app.schemas import DocumentCreate
from app.services.pinecone_service import PineconeService
from app.services.gemini_service import GeminiService
from app.utils.file_utils import secure_filename, get_file_size, validate_pdf

logger = logging.getLogger(__name__)

class DocumentService:
    @staticmethod
    def create_document(db: Session, document_data: DocumentCreate) -> Document:
        """
        Create a new document record in the database
        """
        document = Document(
            filename=document_data.filename,
            file_path=document_data.file_path,
            client_id=document_data.client_id,
            file_size=get_file_size(document_data.file_path),
            uploaded_at=datetime.now()
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        logger.info(f"Created document record for client {document_data.client_id}: {document_data.filename}")
        return document

    @staticmethod
    def get_documents(db: Session, client_id: int) -> List[Document]:
        """
        Get all documents for a specific client
        """
        return db.query(Document).filter(
            Document.client_id == client_id
        ).order_by(Document.uploaded_at.desc()).all()

    @staticmethod
    def get_document(db: Session, document_id: int) -> Optional[Document]:
        """
        Get a specific document by ID
        """
        return db.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def delete_document(db: Session, document_id: int) -> bool:
        """
        Delete a document from database and filesystem
        """
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            return False
        
        # Delete file from filesystem
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Remove from vector database
        PineconeService.remove_document_vectors(document_id)
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        logger.info(f"Deleted document {document_id}: {document.filename}")
        return True

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from PDF file using pdfplumber (more accurate than PyPDF2)
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                logger.error(f"Error extracting text with PyPDF2: {e2}")
                raise Exception(f"Failed to extract text from PDF: {e2}")
        
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap for context preservation
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                for break_point in ['. ', '\n', '? ', '! ']:
                    break_pos = text.rfind(break_point, start, end)
                    if break_pos != -1:
                        end = break_pos + len(break_point)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, considering overlap
            start = end - overlap if end - overlap > start else end
        
        return chunks

    @staticmethod
    def process_document(db: Session, file_path: str, client_id: int) -> bool:
        """
        Process a single document: extract text, chunk it, and store in vector database
        """
        try:
            # Extract text from PDF
            text = DocumentService.extract_text_from_pdf(file_path)
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return False
            
            # Split text into chunks
            chunks = DocumentService.chunk_text(text)
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Store chunks in vector database
            document = db.query(Document).filter(
                Document.file_path == file_path,
                Document.client_id == client_id
            ).first()
            
            if document:
                PineconeService.store_document_chunks(
                    client_id=client_id,
                    document_id=document.id,
                    chunks=chunks
                )
                
                # Mark document as processed
                document.processed = True
                document.processed_at = datetime.now()
                db.commit()
                
                logger.info(f"Processed document {document.id} with {len(chunks)} chunks")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False

    @staticmethod
    def process_documents_for_client(db: Session, client_id: int) -> None:
        """
        Process all unprocessed documents for a client
        """
        documents = db.query(Document).filter(
            Document.client_id == client_id,
            Document.processed == False
        ).all()
        
        for document in documents:
            DocumentService.process_document(db, document.file_path, client_id)

    @staticmethod
    def reprocess_documents(db: Session, client_id: int) -> None:
        """
        Reprocess all documents for a client (e.g., after adding new documents)
        """
        # First remove all existing vectors for this client
        PineconeService.remove_client_vectors(client_id)
        
        # Mark all documents as unprocessed
        db.query(Document).filter(
            Document.client_id == client_id
        ).update({Document.processed: False})
        db.commit()
        
        # Process all documents
        DocumentService.process_documents_for_client(db, client_id)
        
        logger.info(f"Reprocessed all documents for client {client_id}")

    @staticmethod
    def remove_document_from_knowledge(db: Session, document_id: int) -> None:
        """
        Remove a document's vectors from the knowledge base
        """
        PineconeService.remove_document_vectors(document_id)
        
        # Mark document as unprocessed
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processed = False
            document.processed_at = None
            db.commit()
        
        logger.info(f"Removed document {document_id} from knowledge base")

    @staticmethod
    def search_documents(db: Session, client_id: int, query: str, top_k: int = 5) -> List[str]:
        """
        Search for relevant document chunks based on query
        """
        try:
            # Get relevant chunks from vector database
            results = PineconeService.search_similar_chunks(
                client_id=client_id,
                query=query,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents for client {client_id}: {e}")
            return []

    @staticmethod
    def get_document_stats(db: Session, client_id: int) -> dict:
        """
        Get statistics about documents for a client
        """
        total_documents = db.query(Document).filter(
            Document.client_id == client_id
        ).count()
        
        processed_documents = db.query(Document).filter(
            Document.client_id == client_id,
            Document.processed == True
        ).count()
        
        total_size = db.query(Document).filter(
            Document.client_id == client_id
        ).with_entities(db.func.sum(Document.file_size)).scalar() or 0
        
        return {
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "pending_documents": total_documents - processed_documents,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

    @staticmethod
    def validate_and_save_pdf(file, client_id: int, upload_dir: str) -> Optional[str]:
        """
        Validate and save uploaded PDF file
        """
        try:
            # Validate file type
            if not validate_pdf(file.filename, file.content_type):
                raise ValueError("Invalid file type. Only PDF files are allowed.")
            
            # Generate secure filename
            original_filename = secure_filename(file.filename)
            file_extension = os.path.splitext(original_filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Ensure upload directory exists
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
            
            # Verify file is a valid PDF
            try:
                with open(file_path, 'rb') as f:
                    PyPDF2.PdfReader(f)
            except Exception as e:
                os.remove(file_path)
                raise ValueError(f"Invalid PDF file: {str(e)}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving PDF file: {e}")
            raise
