# main.py - Fixed version with proper template flow
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
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
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OwnBot", version="1.0.0")

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Persistent storage files
CLIENTS_FILE = "data/clients.pkl"
BOTS_FILE = "data/bots.pkl"
SESSIONS_FILE = "data/sessions.pkl"
PHONE_NUMBERS_FILE = "data/phone_numbers.pkl"

def load_data(filename, default=None):
    """Load data from pickle file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
    return default if default is not None else {}

def save_data(filename, data):
    """Save data to pickle file"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")

# Load persistent data
clients = load_data(CLIENTS_FILE, {})
bots = load_data(BOTS_FILE, {})
sessions = load_data(SESSIONS_FILE, {})
phone_numbers = load_data(PHONE_NUMBERS_FILE, {})

# Other storage
documents = {}
knowledge_bases = {}

class Bot:
    def __init__(self, client_id: str, client_name: str):
        self.client_id = client_id
        self.client_name = client_name
        self.created_at = datetime.now().isoformat()
        self.status = "inactive"
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
        # Update bot status
        active_channels = sum(1 for channel in self.channels.values() if channel["active"])
        self.status = "active" if active_channels > 0 else "inactive"
    
    def get_info(self):
        active_channels = sum(1 for channel in self.channels.values() if channel["active"])
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "created_at": self.created_at,
            "status": self.status,
            "active_channels": active_channels,
            "total_channels": len(self.channels),
            "channels": self.channels,
            "config": self.config
        }

# Root endpoint
@app.get("/")
async def dashboard(request: Request):
    total_clients = len(clients)
    active_bots = len([bot for bot in bots.values() if bot.status == "active"])
    
    return templates.TemplateResponse("base.html", {
        "request": request,
        "total_clients": total_clients,
        "active_clients": active_bots,
        "messages_today": 0,
        "clients": clients.values()
    })

# Clients Management
@app.get("/clients")
async def clients_page(request: Request):
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "clients": clients.values()
    })

# Add Client
@app.get("/clients/add")
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
        
        # Save to persistent storage
        save_data(CLIENTS_FILE, clients)
        
        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = client_id
        save_data(SESSIONS_FILE, sessions)
        
        response = RedirectResponse(url="/upload_documents", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
        
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(status_code=500, detail="Error creating client")

# Upload Documents
@app.get("/upload_documents")
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
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...)
):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Process uploaded files
    for file in files:
        if file.filename:
            file_path = f"uploads/{client_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Store document info
            if client_id not in documents:
                documents[client_id] = []
            documents[client_id].append({
                "filename": file.filename,
                "file_path": file_path,
                "uploaded_at": datetime.now().isoformat()
            })
    
    logger.info(f"Uploaded {len(files)} documents for client {client['business_name']}")
    
    # Redirect to buy number page
    return RedirectResponse(url="/buy_number", status_code=303)

# Buy Number
@app.get("/buy_number")
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

@app.post("/buy_number")
async def buy_number_post(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Generate a random phone number (in real app, this would be from Twilio)
    phone_number = f"+1{random.randint(200, 999)}{random.randint(200, 999)}{random.randint(1000, 9999)}"
    
    # Store phone number
    phone_numbers[client_id] = {
        "number": phone_number,
        "purchased_at": datetime.now().isoformat(),
        "status": "active"
    }
    save_data(PHONE_NUMBERS_FILE, phone_numbers)
    
    # Redirect to clients bots configuration
    return RedirectResponse(url="/clients_bots", status_code=303)

# Clients Bots Configuration
@app.get("/clients_bots")
async def clients_bots(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Create bot if not exists
    if client_id not in bots:
        bots[client_id] = Bot(client_id, client["business_name"])
        save_data(BOTS_FILE, bots)
    
    bot = bots[client_id]
    
    return templates.TemplateResponse("clients_bots.html", {
        "request": request,
        "client": client,
        "bot": bot.get_info(),
        "phone_number": phone_numbers.get(client_id, {}).get("number", "Not purchased")
    })

# Client Detail
@app.get("/clients/{client_id}")
async def client_detail(request: Request, client_id: str):
    client = clients.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    bot_info = bots.get(client_id)
    phone_info = phone_numbers.get(client_id)
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "bot": bot_info.get_info() if bot_info else None,
        "phone": phone_info
    })

# Bot Configuration API
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
                return JSONResponse({"status": "error", "message": "No phone number available"})
            
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
        
        # Save bot state
        save_data(BOTS_FILE, bots)
        
        return JSONResponse({
            "status": "success",
            "message": message,
            "bot": bot.get_info()
        })
        
    except Exception as e:
        logger.error(f"Error activating channel: {e}")
        return JSONResponse({"status": "error", "message": "Internal server error"})

# Complete Setup
@app.get("/complete")
async def complete_setup(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Get setup status
    bot_info = bots.get(client_id)
    phone_info = phone_numbers.get(client_id)
    doc_count = len(documents.get(client_id, []))
    
    # Clear session
    if session_id in sessions:
        del sessions[session_id]
        save_data(SESSIONS_FILE, sessions)
    
    return templates.TemplateResponse("complete.html", {
        "request": request,
        "client": client,
        "bot": bot_info.get_info() if bot_info else {},
        "phone": phone_info,
        "document_count": doc_count
    })

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "success",
        "message": "OwnBot running",
        "timestamp": datetime.now().isoformat(),
        "clients_count": len(clients),
        "bots_count": len(bots)
    }

@app.get("/test")
async def test_page():
    return {"message": "Server is working!", "status": "success"}

# Save data on shutdown
import atexit
@atexit.register
def save_on_exit():
    save_data(CLIENTS_FILE, clients)
    save_data(BOTS_FILE, bots)
    save_data(SESSIONS_FILE, sessions)
    save_data(PHONE_NUMBERS_FILE, phone_numbers)
