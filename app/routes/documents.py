from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File, Request, Query
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import uuid
from datetime import datetime

from app.database import get_db
from app.models import Client, Document  # Using your actual Document model
from app.services.client_service import ClientService  # ADDED: Import ClientService

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# FIXED: Display upload documents page with client context
@router.get("/upload_documents")
async def upload_documents_page(
    request: Request, 
    client_id: str = Query(..., description="Client ID"),  # ADDED: client_id parameter
    db: Session = Depends(get_db)
):
    """Show the upload documents page for a specific client"""
    try:
        # Get client from database
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # FIXED: Pass client to template
        return templates.TemplateResponse("upload_documents.html", {
            "request": request,
            "client": client  # ✅ NOW CLIENT IS AVAILABLE IN TEMPLATE
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading upload page: {str(e)}")

# Handle document upload
@router.post("/upload_documents")
async def upload_documents(
    request: Request,
    client_id: str = Query(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Handle PDF file upload for a client"""
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Get client
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        stored_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"uploads/{stored_filename}"
        
        # Save the file and get file size
        content = await file.read()
        file_size = len(content)
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Create document record using your actual Document model
        document = Document(
            id=str(uuid.uuid4()),
            client_id=client_id,
            filename=file.filename,  # Original filename
            stored_filename=stored_filename,  # Unique stored filename
            file_path=file_path,  # Path to stored file
            file_size=file_size,  # File size in bytes
            processed=False,  # Not processed yet
            processing_error=None,  # No error initially
            uploaded_at=datetime.utcnow(),
            processed_at=None  # Will be set when processed
        )
        
        db.add(document)
        db.commit()
        
        # Redirect to buy number page after successful upload
        return RedirectResponse(f"/buy_number?client_id={client_id}", status_code=303)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



# Reprocess knowledge base for a client
@router.post("/api/documents/{client_id}/reprocess")
async def reprocess_documents(client_id: str, db: Session = Depends(get_db)):
    """Reprocess knowledge base for a client"""
    try:
        # Get client
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get all documents for this client
        documents = db.query(Document).filter(Document.client_id == client_id).all()
        
        if not documents:
            return {
                "status": "success", 
                "message": "No documents to process",
                "processed_count": 0
            }
        
        # Mark all documents as processed and set processed_at timestamp
        for doc in documents:
            doc.processed = True
            doc.processed_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "status": "success", 
            "message": f"Knowledge base reprocessed for {len(documents)} documents",
            "processed_count": len(documents)
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")

# Get client documents for displaying in UI
@router.get("/api/documents/{client_id}")
async def get_client_documents(client_id: str, db: Session = Depends(get_db)):
    """Get all documents for a client"""
    try:
        documents = db.query(Document).filter(Document.client_id == client_id).all()
        
        return {
            "status": "success",
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "stored_filename": doc.stored_filename,
                    "file_size": doc.file_size,
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processed": doc.processed,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                    "processing_error": doc.processing_error
                }
                for doc in documents
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

# Delete a document
@router.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete the physical file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete the database record
        db.delete(document)
        db.commit()
        
        return {
            "status": "success",
            "message": "Document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

# Get document processing status
@router.get("/api/documents/{client_id}/status")
async def get_documents_status(client_id: str, db: Session = Depends(get_db)):
    """Get document processing status for a client"""
    try:
        documents = db.query(Document).filter(Document.client_id == client_id).all()
        
        processed_count = sum(1 for doc in documents if doc.processed)
        total_count = len(documents)
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "total_count": total_count,
            "progress_percentage": (processed_count / total_count * 100) if total_count > 0 else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")
