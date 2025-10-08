# main.py - Fixed version without template dependencies
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
logger = logging.getLogger.getLogger(__name__)

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

# HTML Templates as strings
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OwnBot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .card { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; }
        .stats { display: flex; gap: 20px; }
        .stat-card { background: white; padding: 15px; border-radius: 8px; flex: 1; }
    </style>
</head>
<body>
    <h1>OwnBot Dashboard</h1>
    <div class="stats">
        <div class="stat-card">
            <h3>Total Clients</h3>
            <p>{total_clients}</p>
        </div>
        <div class="stat-card">
            <h3>Active Bots</h3>
            <p>{active_bots}</p>
        </div>
        <div class="stat-card">
            <h3>Clients with Knowledge Base</h3>
            <p>{clients_with_kb}</p>
        </div>
    </div>
    <div class="card">
        <h2>Quick Actions</h2>
        <a href="/clients/add" class="btn">Add New Client</a>
        <a href="/clients" class="btn">View All Clients</a>
        <a href="/bots" class="btn">Manage Bots</a>
    </div>
    <div class="card">
        <h2>Recent Clients</h2>
        {recent_clients_html}
    </div>
</body>
</html>
"""

CLIENTS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Clients - OwnBot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .card { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .status-active { color: green; }
        .status-inactive { color: red; }
    </style>
</head>
<body>
    <h1>Client Management</h1>
    <a href="/clients/add" class="btn">Add New Client</a>
    <a href="/" class="btn">Dashboard</a>
    
    <div class="card">
        <h2>All Clients</h2>
        <table>
            <thead>
                <tr>
                    <th>Business Name</th>
                    <th>Type</th>
                    <th>Documents</th>
                    <th>Bot Status</th>
                    <th>Phone</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {clients_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

ADD_CLIENT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Add Client - OwnBot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .card { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; max-width: 500px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; border: none; cursor: pointer; }
        input, select, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Add New Client</h1>
    <div class="card">
        <form action="/clients/add" method="post">
            <div>
                <label>Business Name:</label>
                <input type="text" name="business_name" required>
            </div>
            <div>
                <label>Business Type:</label>
                <input type="text" name="business_type" required>
            </div>
            <div>
                <label>Industry:</label>
                <input type="text" name="industry">
            </div>
            <div>
                <label>Description:</label>
                <textarea name="description" rows="3"></textarea>
            </div>
            <button type="submit" class="btn">Create Client</button>
        </form>
    </div>
    <a href="/clients" class="btn">Back to Clients</a>
</body>
</html>
"""

CONFIGURE_BOT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Configure Bot - OwnBot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .card { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; border: none; cursor: pointer; margin: 5px; }
        .channel { display: flex; justify-content: space-between; align-items: center; padding: 15px; background: white; margin: 10px 0; border-radius: 4px; }
        .channel-active { border-left: 4px solid green; }
        .channel-inactive { border-left: 4px solid #ccc; }
    </style>
</head>
<body>
    <h1>Configure Bot for {client_name}</h1>
    
    <div class="card">
        <h2>Bot Channels</h2>
        
        <div class="channel {whatsapp_class}">
            <div>
                <h3>WhatsApp</h3>
                <p>Connect to customers via WhatsApp</p>
                {whatsapp_status}
            </div>
            <button class="btn" onclick="activateChannel('whatsapp')" {whatsapp_disabled}>
                {whatsapp_button}
            </button>
        </div>

        <div class="channel {website_class}">
            <div>
                <h3>Website Chat</h3>
                <p>Add chat widget to your website</p>
                {website_status}
            </div>
            <button class="btn" onclick="activateChannel('website')">
                {website_button}
            </button>
        </div>

        <div class="channel {telegram_class}">
            <div>
                <h3>Telegram</h3>
                <p>Connect to Telegram messenger</p>
                {telegram_status}
            </div>
            <button class="btn" onclick="activateChannel('telegram')">
                {telegram_button}
            </button>
        </div>
    </div>

    <div class="card">
        <h2>Embed Code for Website</h2>
        <textarea rows="4" style="width: 100%; font-family: monospace;" readonly>{embed_code}</textarea>
        <p>Copy and paste this code into your website's HTML</p>
    </div>

    <a href="/complete" class="btn">Finish Setup</a>

    <script>
    async function activateChannel(channel) {
        const response = await fetch('/api/bot/activate_channel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ channel: channel })
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            alert(result.message);
            location.reload();
        } else {
            alert('Error: ' + result.message);
        }
    }
    </script>
</body>
</html>
"""

COMPLETE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Setup Complete - OwnBot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
        .card { background: #f5f5f5; padding: 20px; margin: 20px auto; border-radius: 8px; max-width: 600px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; margin: 10px; }
        .success { color: green; font-size: 24px; }
    </style>
</head>
<body>
    <div class="success">✅</div>
    <h1>Setup Complete!</h1>
    
    <div class="card">
        <h2>Client: {client_name}</h2>
        <p>Your bot has been successfully configured and is ready to use.</p>
        
        <div style="text-align: left; margin: 20px 0;">
            <h3>Summary:</h3>
            <p>📊 Knowledge Base: {document_count} documents</p>
            <p>🤖 Bot Channels: {active_channels} activated</p>
            <p>📞 Phone Number: {phone_status}</p>
        </div>
    </div>

    <div class="card">
        <h3>Next Steps:</h3>
        <a href="/bots" class="btn">Manage Your Bots</a>
        <a href="/clients" class="btn">View All Clients</a>
        <a href="/" class="btn">Dashboard</a>
    </div>
</body>
</html>
"""

# Root endpoint - FIXED: No template dependency
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    total_clients = len(clients)
    active_bots = len([bot for bot in bots.values() if bot.status == "active"])
    clients_with_kb = len([kb for kb in knowledge_bases.values() if kb.documents])
    
    recent_clients = list(clients.values())[-5:]
    recent_clients_html = ""
    for client in recent_clients:
        recent_clients_html += f"<p>• {client['business_name']} ({client['business_type']})</p>"
    
    if not recent_clients:
        recent_clients_html = "<p>No clients yet. <a href='/clients/add'>Add your first client</a></p>"
    
    html_content = DASHBOARD_HTML.format(
        total_clients=total_clients,
        active_bots=active_bots,
        clients_with_kb=clients_with_kb,
        recent_clients_html=recent_clients_html
    )
    
    return HTMLResponse(content=html_content)

# Clients Management
@app.get("/clients", response_class=HTMLResponse)
async def clients_page():
    clients_html = ""
    for client in clients.values():
        has_kb = "✅" if client["id"] in knowledge_bases else "❌"
        has_bot = "✅" if client["id"] in bots else "❌"
        has_phone = "✅" if client["id"] in phone_numbers else "❌"
        
        clients_html += f"""
        <tr>
            <td>{client['business_name']}</td>
            <td>{client['business_type']}</td>
            <td>{has_kb}</td>
            <td>{has_bot}</td>
            <td>{has_phone}</td>
            <td>
                <a href="/configure_bot?client_id={client['id']}">Configure Bot</a>
            </td>
        </tr>
        """
    
    if not clients_html:
        clients_html = "<tr><td colspan='6'>No clients found. <a href='/clients/add'>Add your first client</a></td></tr>"
    
    return HTMLResponse(content=CLIENTS_HTML.format(clients_html=clients_html))

# Add Client
@app.get("/clients/add", response_class=HTMLResponse)
async def add_client_form():
    return HTMLResponse(content=ADD_CLIENT_HTML)

@app.post("/clients/add")
async def submit_client_form(
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
        
        response = RedirectResponse(url="/configure_bot", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": "Error creating client"})

# Configure Bot
@app.get("/configure_bot", response_class=HTMLResponse)
async def configure_bot(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Create bot if not exists
    if client_id not in bots:
        bots[client_id] = Bot(client_id, client["business_name"])
    
    bot = bots[client_id]
    bot_info = bot.get_info()
    
    # Prepare channel status
    whatsapp_active = bot_info["channels"]["whatsapp"]["active"]
    website_active = bot_info["channels"]["website"]["active"]
    telegram_active = bot_info["channels"]["telegram"]["active"]
    
    whatsapp_status = "Active" if whatsapp_active else "Inactive"
    website_status = "Active" if website_active else "Inactive" 
    telegram_status = "Active" if telegram_active else "Inactive"
    
    whatsapp_button = "Activated" if whatsapp_active else "Activate WhatsApp"
    website_button = "Activated" if website_active else "Activate Website Chat"
    telegram_button = "Activated" if telegram_active else "Activate Telegram"
    
    whatsapp_class = "channel-active" if whatsapp_active else "channel-inactive"
    website_class = "channel-active" if website_active else "channel-inactive"
    telegram_class = "channel-active" if telegram_active else "channel-inactive"
    
    whatsapp_disabled = "disabled" if whatsapp_active else ""
    
    embed_code = f'<script src="https://e-z6j0.onrender.com/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    
    html_content = CONFIGURE_BOT_HTML.format(
        client_name=client["business_name"],
        whatsapp_status=whatsapp_status,
        website_status=website_status,
        telegram_status=telegram_status,
        whatsapp_button=whatsapp_button,
        website_button=website_button,
        telegram_button=telegram_button,
        whatsapp_class=whatsapp_class,
        website_class=website_class,
        telegram_class=telegram_class,
        whatsapp_disabled=whatsapp_disabled,
        embed_code=embed_code
    )
    
    return HTMLResponse(content=html_content)

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
        
        return JSONResponse({
            "status": "success",
            "message": message,
            "bot": bot.get_info()
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": "Internal server error"})

# Complete Setup
@app.get("/complete", response_class=HTMLResponse)
async def complete_setup(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    
    # Get bot info
    active_channels = 0
    if client_id in bots:
        bot = bots[client_id]
        active_channels = sum(1 for channel in bot.channels.values() if channel["active"])
    
    # Get knowledge base info
    document_count = 0
    if client_id in knowledge_bases:
        document_count = len(knowledge_bases[client_id].documents)
    
    # Get phone status
    phone_status = "Not configured"
    if client_id in phone_numbers:
        phone_status = phone_numbers[client_id]["number"]
    
    # Clear session
    if session_id in sessions:
        del sessions[session_id]
    
    html_content = COMPLETE_HTML.format(
        client_name=client["business_name"],
        document_count=document_count,
        active_channels=active_channels,
        phone_status=phone_status
    )
    
    response = HTMLResponse(content=html_content)
    response.delete_cookie("session_id")
    return response

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

# Render.com configuration
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
