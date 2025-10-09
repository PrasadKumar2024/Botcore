# main.py - Complete Fixed Version
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
DOCUMENTS_FILE = "data/documents.pkl"

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
documents = load_data(DOCUMENTS_FILE, {})

# Bot Class
class Bot:
    def __init__(self, client_id: str, client_name: str):
        self.client_id = client_id
        self.client_name = client_name
        self.created_at = datetime.now().isoformat()
        self.status = "inactive"  # Start as inactive
        self.channels = {
            "whatsapp": {"active": False, "number": None},
            "voice": {"active": False, "number": None},
            "website": {"active": False, "widget_id": f"widget_{uuid.uuid4().hex[:8]}"}
        }
        self.subscriptions = {
            "whatsapp": {"start_date": None, "end_date": None, "active": False},
            "voice": {"start_date": None, "end_date": None, "active": False},
            "website": {"start_date": None, "end_date": None, "active": False}
        }
        self.config = {
            "welcome_message": f"Hello! Welcome to {client_name}. How can I help you today?",
            "response_mode": "auto",
            "business_hours": "24/7"
        }
    
    def activate_channel(self, channel: str, **kwargs):
        """Activate a bot channel"""
        if channel in self.channels:
            self.channels[channel]["active"] = True
            for key, value in kwargs.items():
                self.channels[channel][key] = value
            # Update overall bot status
            self._update_status()
    
    def deactivate_channel(self, channel: str):
        """Deactivate a bot channel"""
        if channel in self.channels:
            self.channels[channel]["active"] = False
            self._update_status()
    
    def _update_status(self):
        """Update bot status based on active channels"""
        active_channels = sum(1 for channel in self.channels.values() if channel["active"])
        self.status = "active" if active_channels > 0 else "inactive"
    
    def get_info(self):
        """Get bot information"""
        active_channels = sum(1 for channel in self.channels.values() if channel["active"])
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "created_at": self.created_at,
            "status": self.status,
            "active_channels": active_channels,
            "total_channels": len(self.channels),
            "channels": self.channels,
            "subscriptions": self.subscriptions,
            "config": self.config
        }

# ========== ROUTES ==========

# Root Dashboard
@app.get("/")
async def dashboard(request: Request):
    """Main dashboard - only shows completed clients"""
    # Filter out clients that are still in setup (in sessions)
    active_session_clients = set(sessions.values())
    completed_clients = [
        client for client in clients.values() 
        if client["id"] not in active_session_clients
    ]
    
    total_clients = len(completed_clients)
    active_bots = len([
        bot for bot in bots.values() 
        if bot.status == "active" and bot.client_id not in active_session_clients
    ])
    
    return templates.TemplateResponse("base.html", {
        "request": request,
        "total_clients": total_clients,
        "active_clients": active_bots,
        "messages_today": 0,
        "clients": completed_clients
    })

# Clients Page
@app.get("/clients")
async def clients_page(request: Request):
    """View all completed clients"""
    # Filter out clients still in setup
    active_session_clients = set(sessions.values())
    completed_clients = [
        client for client in clients.values() 
        if client["id"] not in active_session_clients
    ]
    
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "clients": completed_clients
    })

# Add Client Form
@app.get("/clients/add")
async def add_client_form(request: Request):
    """Show add client form"""
    return templates.TemplateResponse("add_client.html", {"request": request})

# Submit Add Client Form
@app.post("/clients/add")
async def submit_client_form(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...),
    industry: str = Form(""),
    description: str = Form("")
):
    """Create new client and start setup flow"""
    try:
        client_id = str(uuid.uuid4())
        
        # Create client record
        clients[client_id] = {
            "id": client_id,
            "business_name": business_name,
            "business_type": business_type,
            "industry": industry,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "setup_in_progress"
        }
        
        # Save to persistent storage
        save_data(CLIENTS_FILE, clients)
        
        # Create session for this client setup
        session_id = str(uuid.uuid4())
        sessions[session_id] = client_id
        save_data(SESSIONS_FILE, sessions)
        
        logger.info(f"New client created: {business_name} (ID: {client_id})")
        
        # Redirect to document upload
        response = RedirectResponse(url="/upload_documents", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
        
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(status_code=500, detail="Error creating client")

# Upload Documents Form
@app.get("/upload_documents")
async def upload_documents_form(request: Request):
    """Show document upload form"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    return templates.TemplateResponse("upload_documents.html", {
        "request": request,
        "client": client
    })

# Process Document Upload
@app.post("/upload_documents")
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """Process uploaded documents"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Create upload directory for client
    upload_dir = f"uploads/{client_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Process uploaded files
    uploaded_count = 0
    for file in files:
        if file.filename and file.filename.endswith('.pdf'):
            file_path = os.path.join(upload_dir, file.filename)
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
            uploaded_count += 1
    
    # Save documents
    save_data(DOCUMENTS_FILE, documents)
    
    logger.info(f"Uploaded {uploaded_count} documents for client {client['business_name']}")
    
    # Redirect to buy number page
    return RedirectResponse(url="/buy_number", status_code=303)

# Skip Documents
@app.post("/skip_documents")
async def skip_documents(request: Request):
    """Skip document upload step"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    logger.info(f"Documents skipped for client {client_id}")
    return RedirectResponse(url="/buy_number", status_code=303)

# Buy Number Form
@app.get("/buy_number")
async def buy_number_form(request: Request):
    """Show phone number purchase form"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    return templates.TemplateResponse("buy_number.html", {
        "request": request,
        "client": client
    })

# Process Number Purchase
@app.post("/buy_number")
async def buy_number_post(request: Request, country: str = Form("India")):
    """Purchase phone number for client"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Generate demo phone number
    country_codes = {
        "India": "+91",
        "USA": "+1",
        "UK": "+44",
        "Australia": "+61"
    }
    
    country_code = country_codes.get(country, "+91")
    phone_number = f"{country_code} {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
    
    # Store phone number
    phone_numbers[client_id] = {
        "number": phone_number,
        "country": country,
        "purchased_at": datetime.now().isoformat(),
        "status": "active",
        "is_demo": True
    }
    save_data(PHONE_NUMBERS_FILE, phone_numbers)
    
    logger.info(f"Phone number {phone_number} assigned to client {client['business_name']}")
    
    # Redirect to bot configuration
    return RedirectResponse(url="/clients_bots", status_code=303)

# Skip Number Purchase
@app.post("/skip_number")
async def skip_number(request: Request):
    """Skip phone number purchase step"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    logger.info(f"Number purchase skipped for client {client_id}")
    return RedirectResponse(url="/clients_bots", status_code=303)

# Bot Configuration Page
@app.get("/clients_bots")
async def clients_bots(request: Request):
    """Show bot configuration page"""
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
    phone_info = phone_numbers.get(client_id, {})
    doc_info = documents.get(client_id, [])
    
    return templates.TemplateResponse("clients_bots.html", {
        "request": request,
        "client": client,
        "bot": bot.get_info(),
        "phone_number": phone_info.get("number", "Not purchased"),
        "has_phone": client_id in phone_numbers,
        "document_count": len(doc_info),
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="https://yourdomain.com/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    })

# Activate Bot Channel API
@app.post("/api/bot/activate_channel")
async def activate_bot_channel(request: Request):
    """Activate a specific bot channel"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    try:
        data = await request.json()
        channel = data.get("channel")
        
        if client_id not in bots:
            return JSONResponse({"status": "error", "message": "Bot not found"})
        
        bot = bots[client_id]
        
        # Activate channel based on type
        if channel == "whatsapp" or channel == "voice":
            phone_data = phone_numbers.get(client_id, {})
            if not phone_data:
                return JSONResponse({
                    "status": "error", 
                    "message": "No phone number available. Please buy a number first."
                })
            
            bot.activate_channel(channel, number=phone_data["number"])
            message = f"{channel.title()} bot activated with number: {phone_data['number']}"
            
        elif channel == "website":
            bot.activate_channel("website")
            message = "Website chat widget activated"
            
        else:
            return JSONResponse({"status": "error", "message": "Invalid channel"})
        
        # Save bot state
        save_data(BOTS_FILE, bots)
        
        logger.info(f"Activated {channel} channel for client {client_id}")
        
        return JSONResponse({
            "status": "success",
            "message": message,
            "bot": bot.get_info()
        })
        
    except Exception as e:
        logger.error(f"Error activating channel: {e}")
        return JSONResponse({"status": "error", "message": "Internal server error"})

# Deactivate Bot Channel API
@app.post("/api/bot/deactivate_channel")
async def deactivate_bot_channel(request: Request):
    """Deactivate a specific bot channel"""
    try:
        data = await request.json()
        client_id = data.get("client_id")
        channel = data.get("channel")
        
        if not client_id or client_id not in bots:
            return JSONResponse({"status": "error", "message": "Bot not found"})
        
        bot = bots[client_id]
        bot.deactivate_channel(channel)
        
        # Save bot state
        save_data(BOTS_FILE, bots)
        
        return JSONResponse({
            "status": "success",
            "message": f"{channel.title()} bot deactivated",
            "bot": bot.get_info()
        })
        
    except Exception as e:
        logger.error(f"Error deactivating channel: {e}")
        return JSONResponse({"status": "error", "message": "Internal server error"})

# Complete Setup
@app.get("/complete")
async def complete_setup(request: Request):
    """Complete client setup and show summary"""
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Update client status to completed
    if client:
        client["status"] = "completed"
        save_data(CLIENTS_FILE, clients)
    
    # Get setup summary
    bot_info = bots.get(client_id)
    phone_info = phone_numbers.get(client_id)
    doc_count = len(documents.get(client_id, []))
    
    # Clear session - client setup is complete
    if session_id in sessions:
        del sessions[session_id]
        save_data(SESSIONS_FILE, sessions)
    
    response = templates.TemplateResponse("complete.html", {
        "request": request,
        "client": client,
        "bot": bot_info.get_info() if bot_info else None,
        "phone": phone_info,
        "document_count": doc_count
    })
    
    # Delete session cookie
    response.delete_cookie("session_id")
    
    logger.info(f"Client setup completed: {client['business_name']}")
    
    return response

# Client Detail Page
@app.get("/clients/{client_id}")
async def client_detail(request: Request, client_id: str):
    """View detailed client information"""
    client = clients.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    bot_info = bots.get(client_id)
    phone_info = phone_numbers.get(client_id)
    doc_info = documents.get(client_id, [])
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "bot": bot_info.get_info() if bot_info else None,
        "phone": phone_info,
        "documents": doc_info
    })

# Health Check
@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "success",
        "message": "OwnBot is running",
        "timestamp": datetime.now().isoformat(),
        "clients_count": len(clients),
        "bots_count": len(bots),
        "active_sessions": len(sessions)
    }

# Test Endpoint
@app.get("/test")
async def test_page():
    """Simple test endpoint"""
    return {
        "message": "Server is working!",
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }

# Save data on shutdown
import atexit

@atexit.register
def save_on_exit():
    """Save all data when application exits"""
    logger.info("Saving data on application exit...")
    save_data(CLIENTS_FILE, clients)
    save_data(BOTS_FILE, bots)
    save_data(SESSIONS_FILE, sessions)
    save_data(PHONE_NUMBERS_FILE, phone_numbers)
    save_data(DOCUMENTS_FILE, documents)
    logger.info("Data saved successfully")

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
