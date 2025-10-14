from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File, Request
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import uuid
from datetime import datetime

from app.database import get_db
from app.models import Client, ClientDocument

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Display upload documents page
@router.get("/upload_documents")
async def upload_documents_page(request: Request, db: Session = Depends(get_db)):
    """Show the upload documents page"""
    # You might want to pass client context here if needed
    return templates.TemplateResponse("upload_documents.html", {"request": request})

# Handle document upload
@router.post("/upload_documents")
async def upload_documents(
    request: Request,
    client_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Handle PDF file upload for a client"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Get client
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"uploads/{unique_filename}"
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record in database
        document = ClientDocument(
            id=str(uuid.uuid4()),
            client_id=client_id,
            filename=file.filename,
            file_path=file_path,
            uploaded_at=datetime.utcnow(),
            processed=False
        )
        
        db.add(document)
        db.commit()
        
        # Redirect back to client page
        return RedirectResponse(f"/client/{client_id}", status_code=303)
        
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
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get all documents for this client
        documents = db.query(ClientDocument).filter(ClientDocument.client_id == client_id).all()
        
        if not documents:
            return {
                "status": "success", 
                "message": "No documents to process",
                "processed_count": 0
            }
        
        # TODO: Add your actual PDF processing logic here
        # For now, just mark them as processed
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

# Get client documents
@router.get("/api/documents/{client_id}")
async def get_client_documents(client_id: str, db: Session = Depends(get_db)):
    """Get all documents for a client"""
    try:
        documents = db.query(ClientDocument).filter(ClientDocument.client_id == client_id).all()
        
        return {
            "status": "success",
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processed": doc.processed,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None
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
        document = db.query(ClientDocument).filter(ClientDocument.id == document_id).first()
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
