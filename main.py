# main.py - PROPER TEMPLATE-BASED VERSION
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
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OwnBot", version="1.0.0")

# Create necessary directories for Render.com
os.makedirs("app/static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

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

# Templates - THIS IS CRITICAL
templates = Jinja2Templates(directory="templates")

# Storage
clients = {}
documents = {}
phone_numbers = {}
subscriptions = {}
sessions = {}
knowledge_bases = {}
twilio_numbers = {}
bots = {}

# Knowledge Base Management
class KnowledgeBase:
    def __init__(self, client_id: str, client_name: str):
        self.client_id = client_id
        self.client_name = client_name
        self.created_at = datetime.now().isoformat()
        self.documents = []
        self.status = "active"
        
    def add_document(self, filename: str, file_path: str):
        document_info = {
            "filename": filename,
            "file_path": file_path,
            "uploaded_at": datetime.now().isoformat(),
            "processed": False
        }
        self.documents.append(document_info)
        return document_info
    
    def get_info(self):
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "created_at": self.created_at,
            "document_count": len(self.documents),
            "status": self.status,
            "documents": self.documents
        }

class Bot:
    def __init__(self, client_id: str, client_name: str):
        self.client_id = client_id
        self.client_name = client_name
        self.created_at = datetime.now().isoformat()
        self.status = "active"
        self.channels = {
            "whatsapp": {"active": False, "number": None},
            "website": {"active": False, "widget_id": f"widget_{uuid.uuid4().hex[:8]}"},
            "telegram": {"active": False, "username": None}
        }
        self.config = {
            "welcome_message": f"Hello! Welcome to {client_name}. How can I help you today?",
            "response_mode": "auto",
            "business_hours": "24/7"
        }
    
    def activate_channel(self, channel: str, **kwargs):
        self.channels[channel]["active"] = True
        for key, value in kwargs.items():
            self.channels[channel][key] = value
    
    def get_info(self):
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "created_at": self.created_at,
            "status": self.status,
            "channels": self.channels,
            "config": self.config
        }

# ========== PROPER TEMPLATE ROUTES ==========

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    total_clients = len(clients)
    active_bots = len([bot for bot in bots.values() if bot.status == "active"])
    clients_with_kb = len([kb for kb in knowledge_bases.values() if kb.documents])
    
    recent_clients = list(clients.values())[-5:]
    
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "total_clients": total_clients,
        "active_bots": active_bots,
        "clients_with_kb": clients_with_kb,
        "recent_clients": recent_clients
    })

@app.get("/clients", response_class=HTMLResponse)
async def clients_page(request: Request):
    client_data = []
    for client in clients.values():
        client_info = client.copy()
        client_info["has_knowledge_base"] = client["id"] in knowledge_bases
        client_info["has_bot"] = client["id"] in bots
        client_info["has_phone"] = client["id"] in phone_numbers
        
        if client["id"] in knowledge_bases:
            kb = knowledge_bases[client["id"]]
            client_info["document_count"] = len(kb.documents)
        else:
            client_info["document_count"] = 0
            
        if client["id"] in bots:
            bot = bots[client["id"]]
            client_info["bot_status"] = bot.status
            client_info["active_channels"] = sum(1 for channel in bot.channels.values() if channel["active"])
        else:
            client_info["bot_status"] = "Not Created"
            client_info["active_channels"] = 0
            
        client_data.append(client_info)
    
    return templates.TemplateResponse("clients.html", {
        "request": request, 
        "clients": client_data
    })

@app.get("/clients/add", response_class=HTMLResponse)
async def add_client_form(request: Request):
    return templates.TemplateResponse("add_client.html", {"request": request})

@app.post("/clients/add")
async def submit_client_form(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...),
    industry: str = Form(""),
    description: str = Form("")
):
    try:
        client_id = str(uuid.uuid4())
        
        clients[client_id] = {
            "id": client_id,
            "business_name": business_name,
            "business_type": business_type,
            "industry": industry,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        session_id = str(uuid.uuid4())
        sessions[session_id] = client_id
        
        # Create knowledge base
        knowledge_bases[client_id] = KnowledgeBase(client_id, business_name)
        
        # Create bot
        bots[client_id] = Bot(client_id, business_name)
        
        logger.info(f"New client created: {business_name} (ID: {client_id})")
        
        response = RedirectResponse(url="/upload_documents", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
        
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        return JSONResponse(status_code=500, content={"detail": "Error creating client"})

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
        
        upload_dir = f"uploads/{client_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        kb = knowledge_bases[client_id]
        
        uploaded_count = 0
        for file in files:
            if file.filename.endswith('.pdf'):
                file_path = f"{upload_dir}/{file.filename}"
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                kb.add_document(file.filename, file_path)
                
                if client_id not in documents:
                    documents[client_id] = []
                
                documents[client_id].append({
                    "filename": file.filename,
                    "uploaded_at": datetime.now().isoformat(),
                    "file_path": file_path
                })
                uploaded_count += 1
        
        logger.info(f"Uploaded {uploaded_count} documents for client {client['business_name']}")
        return RedirectResponse(url="/buy_number", status_code=303)
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        return JSONResponse(status_code=500, content={"detail": "Error uploading documents"})

@app.post("/skip_documents")
async def skip_documents_step(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    logger.info(f"Documents skipped for client {client_id}")
    return RedirectResponse(url="/buy_number", status_code=303)

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
        client_data = clients.get(client_id, {})
        business_name = client_data.get('business_name', 'Unknown Business')
        
        # For demo - simulate number purchase
        simulated_number = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
        
        phone_numbers[client_id] = {
            "number": simulated_number,
            "twilio_sid": f"simulated_{uuid.uuid4()}",
            "purchased_at": datetime.now().isoformat(),
            "client_name": business_name,
            "is_real": False,
            "capabilities": {"sms": True, "voice": True}
        }
        
        logger.info(f"Number created: {simulated_number} for {business_name}")
        
        return JSONResponse({
            "status": "success",
            "phone_number": simulated_number,
            "is_real_number": False,
            "message": "Demo number created successfully!"
        })
        
    except Exception as e:
        logger.error(f"Error buying number: {e}")
        return JSONResponse({"status": "error", "message": "Error buying number"})

@app.post("/skip_number")
async def skip_number_step(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    logger.info(f"Number purchase skipped for client {client_id}")
    return RedirectResponse(url="/clients_bots", status_code=303)

@app.get("/clients_bots", response_class=HTMLResponse)
async def bots_configuration(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    phone_data = phone_numbers.get(client_id, {})
    phone_number = phone_data.get("number", "Not purchased")
    
    kb_info = None
    if client_id in knowledge_bases:
        kb_info = knowledge_bases[client_id].get_info()
    
    bot_info = None
    if client_id in bots:
        bot_info = bots[client_id].get_info()
    
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
        "bot": bot_info,
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="https://e-z6j0.onrender.com/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    })

@app.post("/api/bot/activate_channel")
async def activate_bot_channel(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return JSONResponse({"status": "error", "message": "No session"})
    
    try:
        data = await request.json()
        channel = data.get("channel")
        
        if client_id not in bots:
            return JSONResponse({"status": "error", "message": "Bot not found"})
        
        bot = bots[client_id]
        
        if channel == "whatsapp":
            phone_data = phone_numbers.get(client_id, {})
            if not phone_data:
                return JSONResponse({"status": "error", "message": "No phone number available. Please buy a number first."})
            
            bot.activate_channel("whatsapp", number=phone_data["number"])
            message = f"WhatsApp bot activated with number: {phone_data['number']}"
            
        elif channel == "website":
            bot.activate_channel("website")
            message = "Website chat widget activated"
            
        elif channel == "telegram":
            bot.activate_channel("telegram", username=f"{client_id}_bot")
            message = "Telegram bot activated"
            
        else:
            return JSONResponse({"status": "error", "message": "Invalid channel"})
        
        logger.info(f"Activated {channel} channel for client {client_id}")
        
        return JSONResponse({
            "status": "success",
            "message": message,
            "bot": bot.get_info()
        })
        
    except Exception as e:
        logger.error(f"Error activating bot channel: {e}")
        return JSONResponse({"status": "error", "message": "Internal server error"})

@app.get("/complete", response_class=HTMLResponse)
async def complete_setup(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    bot_info = None
    if client_id in bots:
        bot_info = bots[client_id].get_info()
    
    kb_summary = None
    if client_id in knowledge_bases:
        kb = knowledge_bases[client_id]
        kb_summary = {
            "document_count": len(kb.documents),
            "documents": [doc["filename"] for doc in kb.documents[:5]]
        }
    
    # Clear session
    if session_id in sessions:
        del sessions[session_id]
    
    response = templates.TemplateResponse("complete.html", {
        "request": request,
        "client": client,
        "bot": bot_info,
        "knowledge_base": kb_summary,
        "has_phone": client_id in phone_numbers
    })
    response.delete_cookie("session_id")
    return response

@app.get("/bots", response_class=HTMLResponse)
async def bots_dashboard(request: Request):
    bot_data = []
    for client_id, bot in bots.items():
        bot_info = bot.get_info()
        client_info = clients.get(client_id, {})
        phone_info = phone_numbers.get(client_id, {})
        kb_info = knowledge_bases.get(client_id)
        
        bot_data.append({
            **bot_info,
            "client_info": client_info,
            "phone_number": phone_info.get("number", "Not set"),
            "document_count": len(kb_info.documents) if kb_info else 0
        })
    
    return templates.TemplateResponse("bots_dashboard.html", {
        "request": request,
        "bots": bot_data,
        "total_bots": len(bots),
        "active_bots": len([b for b in bots.values() if b.status == "active"])
    })

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "success",
        "message": "OwnBot running",
        "timestamp": datetime.now().isoformat(),
        "clients_count": len(clients),
        "bots_count": len(bots),
        "knowledge_bases_count": len(knowledge_bases)
    }

@app.get("/test")
async def test_page():
    return {"message": "Server is working", "status": "success"}

# Render.com configuration
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
