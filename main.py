# main.py - Enhanced with Knowledge Base
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import logging
import os
from datetime import datetime, timedelta
import uuid
import random
import shutil
from typing import Dict, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OwnBot", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")

# Storage
clients = {}
documents = {}
phone_numbers = {}
subscriptions = {}
sessions = {}
knowledge_bases = {}  # NEW: Store knowledge base info

# NEW: Knowledge Base Management
class KnowledgeBase:
    def __init__(self, client_id: str, client_name: str):
        self.client_id = client_id
        self.client_name = client_name
        self.created_at = datetime.now().isoformat()
        self.documents = []
        self.status = "active"
        
    def add_document(self, filename: str, file_path: str):
        """Add document to knowledge base"""
        document_info = {
            "filename": filename,
            "file_path": file_path,
            "uploaded_at": datetime.now().isoformat(),
            "processed": False  # Will be True when AI processes it
        }
        self.documents.append(document_info)
        return document_info
    
    def get_info(self):
        """Return knowledge base info"""
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "created_at": self.created_at,
            "document_count": len(self.documents),
            "status": self.status,
            "documents": self.documents
        }

# Dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "clients": list(clients.values())
    })

@app.get("/clients", response_class=HTMLResponse)
async def clients_page(request: Request):
    # Enhance client data with KB info
    client_data = []
    for client in clients.values():
        client_info = client.copy()
        client_info["has_knowledge_base"] = client["id"] in knowledge_bases
        if client["id"] in knowledge_bases:
            kb = knowledge_bases[client["id"]]
            client_info["document_count"] = len(kb.documents)
        else:
            client_info["document_count"] = 0
        client_data.append(client_info)
    
    return templates.TemplateResponse("clients.html", {
        "request": request, 
        "clients": client_data
    })

# Step 1: Add Client
@app.get("/clients/add", response_class=HTMLResponse)
async def add_client_form(request: Request):
    return templates.TemplateResponse("add_client.html", {"request": request})

@app.post("/clients/add")
async def submit_client_form(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...)
):
    try:
        client_id = str(uuid.uuid4())
        
        clients[client_id] = {
            "id": client_id,
            "business_name": business_name,
            "business_type": business_type,
            "created_at": datetime.now().isoformat()
        }
        
        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = client_id
        
        logger.info(f"New client created: {business_name} (ID: {client_id})")
        
        # Redirect to Step 2
        response = RedirectResponse(url="/upload_documents", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
        
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Error creating client"}
        )

# Step 2: Upload Documents
@app.get("/upload_documents", response_class=HTMLResponse)
async def upload_documents_form(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    return templates.TemplateResponse("upload_documents.html", {
        "request": request,
        "client": client
    })

@app.post("/upload_documents")
async def process_documents(
    request: Request,
    files: list[UploadFile] = File(...)
):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    try:
        client = clients.get(client_id, {})
        
        # Create upload directory
        upload_dir = f"uploads/{client_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create knowledge base if it doesn't exist
        if client_id not in knowledge_bases:
            knowledge_bases[client_id] = KnowledgeBase(client_id, client["business_name"])
            logger.info(f"Created knowledge base for client: {client['business_name']}")
        
        kb = knowledge_bases[client_id]
        
        # Save files and add to knowledge base
        uploaded_count = 0
        for file in files:
            if file.content_type == "application/pdf":
                file_path = f"{upload_dir}/{file.filename}"
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Add to knowledge base
                kb.add_document(file.filename, file_path)
                
                # Also store in documents for backward compatibility
                if client_id not in documents:
                    documents[client_id] = []
                
                documents[client_id].append({
                    "filename": file.filename,
                    "uploaded_at": datetime.now().isoformat(),
                    "file_path": file_path
                })
                uploaded_count += 1
        
        logger.info(f"Uploaded {uploaded_count} documents to knowledge base for client {client['business_name']}")
        return RedirectResponse(url="/buy_number", status_code=303)
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Error uploading documents"}
        )

@app.post("/skip_documents")
async def skip_documents_step(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Create empty knowledge base even if no documents
    if client_id not in knowledge_bases:
        knowledge_bases[client_id] = KnowledgeBase(client_id, client["business_name"])
        logger.info(f"Created empty knowledge base for client: {client['business_name']}")
    
    logger.info(f"Documents skipped for client {client['business_name']}")
    return RedirectResponse(url="/buy_number", status_code=303)

# NEW: Knowledge Base Status Page
@app.get("/knowledge_base/{client_id}")
async def view_knowledge_base(client_id: str, request: Request):
    if client_id not in clients:
        return JSONResponse({"error": "Client not found"}, status_code=404)
    
    client = clients[client_id]
    kb_info = None
    
    if client_id in knowledge_bases:
        kb_info = knowledge_bases[client_id].get_info()
    
    return templates.TemplateResponse("knowledge_base.html", {
        "request": request,
        "client": client,
        "knowledge_base": kb_info
    })

# NEW: API to get knowledge base status
@app.get("/api/knowledge_base/{client_id}")
async def get_knowledge_base_status(client_id: str):
    if client_id not in clients:
        return JSONResponse({"error": "Client not found"}, status_code=404)
    
    if client_id not in knowledge_bases:
        return JSONResponse({
            "exists": False,
            "message": "No knowledge base created yet"
        })
    
    kb = knowledge_bases[client_id]
    return JSONResponse({
        "exists": True,
        "knowledge_base": kb.get_info()
    })

# Step 3: Buy Number (unchanged)
@app.get("/buy_number", response_class=HTMLResponse)
async def buy_number_form(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    return templates.TemplateResponse("buy_number.html", {
        "request": request,
        "client": client
    })

@app.post("/api/numbers/buy")
async def buy_number(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return JSONResponse({"status": "error", "message": "No session"})
    
    try:
        # Generate phone number
        phone_number = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
        
        phone_numbers[client_id] = {
            "number": phone_number,
            "purchased_at": datetime.now().isoformat()
        }
        
        logger.info(f"Number purchased for client {client_id}: {phone_number}")
        return JSONResponse({
            "status": "success",
            "phone_number": phone_number
        })
        
    except Exception as e:
        logger.error(f"Error buying number: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Error buying number"}
        )

@app.post("/skip_number")
async def skip_number_step(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    logger.info(f"Number purchase skipped for client {client_id}")
    return RedirectResponse(url="/clients_bots", status_code=303)

# Step 4: Bots Configuration - Enhanced with KB info
@app.get("/clients_bots", response_class=HTMLResponse)
async def bots_configuration(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    phone_data = phone_numbers.get(client_id, {})
    phone_number = phone_data.get("number", "Not purchased")
    
    # Knowledge base info
    kb_info = None
    if client_id in knowledge_bases:
        kb_info = knowledge_bases[client_id].get_info()
    
    # Initialize subscriptions
    if client_id not in subscriptions:
        subscriptions[client_id] = {
            "whatsapp": {"active": False, "start_date": None, "end_date": None},
            "voice": {"active": False, "start_date": None, "end_date": None},
            "web": {"active": False, "start_date": None, "end_date": None}
        }
    
    return templates.TemplateResponse("clients_bots.html", {
        "request": request,
        "client": client,
        "phone_number": phone_number,
        "has_phone": client_id in phone_numbers,
        "subscriptions": subscriptions[client_id],
        "knowledge_base": kb_info,
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    })

# Step 5: Complete - Enhanced with KB summary
@app.get("/complete", response_class=HTMLResponse)
async def complete_setup(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Knowledge base summary
    kb_summary = None
    if client_id in knowledge_bases:
        kb = knowledge_bases[client_id]
        kb_summary = {
            "document_count": len(kb.documents),
            "documents": [doc["filename"] for doc in kb.documents[:5]]  # First 5 docs
        }
    
    # Clear session
    if session_id in sessions:
        del sessions[session_id]
    
    response = templates.TemplateResponse("complete.html", {
        "request": request,
        "client": client,
        "knowledge_base": kb_summary,
        "has_phone": client_id in phone_numbers
    })
    response.delete_cookie("session_id")
    return response

# Health check with KB stats
@app.get("/health")
async def health_check():
    kb_stats = {
        "total_knowledge_bases": len(knowledge_bases),
        "clients_with_kb": [client_id for client_id in knowledge_bases if knowledge_bases[client_id].documents]
    }
    
    return {
        "status": "success",
        "message": "OwnBot running",
        "timestamp": datetime.now().isoformat(),
        "clients_count": len(clients),
        "knowledge_bases": kb_stats
    }

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working", "status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
