
from fastapi import APIRouter, Request, Depends, Form, File, UploadFile, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import uuid

from app.database import get_db
from app.services.client_service import ClientService
from app.services.document_service import DocumentService
from app.schemas import DocumentCreate
from app.models import Client

router = APIRouter(prefix="/clients/{client_id}", tags=["documents"])
templates = Jinja2Templates(directory="templates")

# Configure upload directory
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@router.get("/documents", response_class=HTMLResponse)
async def upload_documents_form(
    request: Request, 
    client_id: int, 
    db: Session = Depends(get_db)
):
    """Page 3: PDF upload page for a specific client"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return templates.TemplateResponse("upload_documents.html", {
        "request": request,
        "client": client
    })

@router.post("/documents", response_class=HTMLResponse)
async def upload_documents(
    request: Request,
    client_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Process PDF uploads and redirect to number purchase"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Process each uploaded file
    for file in files:
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a PDF"
            )
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
        
        # Save file to upload directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create document record in database
        document_data = DocumentCreate(
            filename=file.filename,
            file_path=file_path,
            client_id=client_id
        )
        
        DocumentService.create_document(db, document_data)
    
    # Process documents for AI (this would be async in production)
    DocumentService.process_documents_for_client(db, client_id)
    
    return RedirectResponse(
        url=f"/clients/{client_id}/numbers", 
        status_code=303
    )

@router.post("/documents/skip", response_class=HTMLResponse)
async def skip_documents(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Skip document upload and redirect to number purchase"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return RedirectResponse(
        url=f"/clients/{client_id}/numbers", 
        status_code=303
    )

@router.get("/data", response_class=HTMLResponse)
async def client_data_tab(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Data tab for client detail page - show uploaded PDFs"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    documents = DocumentService.get_documents(db, client_id)
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "tab": "data",
        "documents": documents
    })

@router.post("/documents/upload", response_class=HTMLResponse)
async def upload_additional_document(
    request: Request,
    client_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a new document for an existing client (from data tab)"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="File must be a PDF"
        )
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
    
    # Save file to upload directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create document record in database
    document_data = DocumentCreate(
        filename=file.filename,
        file_path=file_path,
        client_id=client_id
    )
    
    DocumentService.create_document(db, document_data)
    
    # Process the new document for AI
    DocumentService.process_document(db, file_path, client_id)
    
    return RedirectResponse(
        url=f"/clients/{client_id}?tab=data", 
        status_code=303
    )

@router.post("/documents/{document_id}/delete", response_class=HTMLResponse)
async def delete_document(
    request: Request,
    client_id: int,
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document from the data tab"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get document to delete
    document = DocumentService.get_document(db, document_id)
    if not document or document.client_id != client_id:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file from filesystem
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete document record from database
    DocumentService.delete_document(db, document_id)
    
    # Remove document from AI knowledge base
    DocumentService.remove_document_from_knowledge(db, document_id)
    
    return RedirectResponse(
        url=f"/clients/{client_id}?tab=data", 
        status_code=303
    )

@router.post("/documents/reprocess", response_class=HTMLResponse)
async def reprocess_documents(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Reprocess all documents for a client (from data tab)"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Reprocess all documents for AI
    DocumentService.reprocess_documents(db, client_id)
    
    return RedirectResponse(
        url=f"/clients/{client_id}?tab=data", 
        status_code=303
      )
