from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

# Initialize FastAPI app
app = FastAPI(title="OwnBot", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Data storage files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CLIENTS_FILE = os.path.join(DATA_DIR, "clients.json")
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json") 
PHONE_NUMBERS_FILE = os.path.join(DATA_DIR, "phone_numbers.json")
SUBSCRIPTIONS_FILE = os.path.join(DATA_DIR, "subscriptions.json")
WHATSAPP_PROFILES_FILE = os.path.join(DATA_DIR, "whatsapp_profiles.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

# In-memory storage (will persist to JSON files)
clients = {}
documents = {}
phone_numbers = {}
subscriptions = {}
whatsapp_profiles = {}
sessions = {}

# Load existing data
def load_data():
    global clients, documents, phone_numbers, subscriptions, whatsapp_profiles, sessions
    
    try:
        with open(CLIENTS_FILE, 'r') as f:
            clients = json.load(f)
    except FileNotFoundError:
        clients = {}
    
    try:
        with open(DOCUMENTS_FILE, 'r') as f:
            documents = json.load(f)
    except FileNotFoundError:
        documents = {}
    
    try:
        with open(PHONE_NUMBERS_FILE, 'r') as f:
            phone_numbers = json.load(f)
    except FileNotFoundError:
        phone_numbers = {}
    
    try:
        with open(SUBSCRIPTIONS_FILE, 'r') as f:
            subscriptions = json.load(f)
    except FileNotFoundError:
        subscriptions = {}
    
    try:
        with open(WHATSAPP_PROFILES_FILE, 'r') as f:
            whatsapp_profiles = json.load(f)
    except FileNotFoundError:
        whatsapp_profiles = {}
    
    try:
        with open(SESSIONS_FILE, 'r') as f:
            sessions = json.load(f)
    except FileNotFoundError:
        sessions = {}

# Save data to JSON files
def save_data():
    with open(CLIENTS_FILE, 'w') as f:
        json.dump(clients, f, indent=2)
    with open(DOCUMENTS_FILE, 'w') as f:
        json.dump(documents, f, indent=2)
    with open(PHONE_NUMBERS_FILE, 'w') as f:
        json.dump(phone_numbers, f, indent=2)
    with open(SUBSCRIPTIONS_FILE, 'w') as f:
        json.dump(subscriptions, f, indent=2)
    with open(WHATSAPP_PROFILES_FILE, 'w') as f:
        json.dump(whatsapp_profiles, f, indent=2)
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

# Load data on startup
load_data()

# Helper functions
def get_client_from_session(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    client_id = sessions[session_id]
    return clients.get(client_id)

def generate_phone_number(country_code="+91"):
    """Generate simulated phone number"""
    return f"{country_code} 9{random.randint(1000, 9999)} {random.randint(1000, 9999)}"

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing all completed clients"""
    completed_clients = {k: v for k, v in clients.items() if v.get("status") == "completed"}
    return templates.TemplateResponse("base.html", {
        "request": request,
        "clients": list(completed_clients.values()),
        "documents": documents,
        "phone_numbers": phone_numbers,
        "subscriptions": subscriptions
    })

@app.get("/clients", response_class=HTMLResponse)
async def clients_list(request: Request):
    """Client list page"""
    completed_clients = {k: v for k, v in clients.items() if v.get("status") == "completed"}
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "clients": list(completed_clients.values()),
        "documents": documents,
        "phone_numbers": phone_numbers,
        "subscriptions": subscriptions
    })

@app.get("/add_client", response_class=HTMLResponse)
async def add_client_form(request: Request):
    """Show add client form"""
    return templates.TemplateResponse("add_client.html", {"request": request})

@app.post("/clients/add")
async def clients_add(
    business_name: str = Form(...),
    business_type: str = Form(...)
):
    """Create new client and start setup session"""
    client_id = str(uuid.uuid4())
    
    clients[client_id] = {
        "id": client_id,
        "business_name": business_name,
        "business_type": business_type,
        "created_at": datetime.now().isoformat(),
        "status": "setup_in_progress"
    }
    
    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = client_id
    
    save_data()
    
    response = RedirectResponse(url="/upload_documents", status_code=303)
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.get("/upload_documents", response_class=HTMLResponse)
async def upload_documents_form(request: Request):
    """Show PDF upload form"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return templates.TemplateResponse("upload_documents.html", {
        "request": request,
        "client": client
    })

@app.post("/upload_documents")
async def upload_documents(
    request: Request,
    client_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Handle PDF upload"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if client_id not in documents:
        documents[client_id] = []
    
    if file.filename and file.filename.endswith('.pdf'):
        document_id = str(uuid.uuid4())
        # Read file content to get size
        content = await file.read()
        documents[client_id].append({
            "id": document_id,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "file_size": len(content),
            "processed": False
        })
    
    save_data()
    return RedirectResponse(url="/buy_number", status_code=303)

@app.post("/skip_documents")
async def skip_documents(request: Request):
    """Skip document upload step"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return RedirectResponse(url="/buy_number", status_code=303)

@app.get("/buy_number", response_class=HTMLResponse)
async def buy_number_form(request: Request):
    """Show phone number purchase form"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return templates.TemplateResponse("buy_number.html", {
        "request": request,
        "client": client
    })

@app.post("/buy_number")
async def buy_number(request: Request, country: str = Form(...)):
    """Buy phone number (simulated)"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    client_id = client["id"]
    
    # Generate simulated phone number
    phone_number = generate_phone_number()
    
    phone_numbers[client_id] = {
        "number": phone_number,
        "country": country,
        "purchased_at": datetime.now().isoformat(),
        "status": "active",
        "is_simulated": True
    }
    
    save_data()
    return RedirectResponse(url="/clients_bots", status_code=303)

@app.post("/skip_number")
async def skip_number(request: Request):
    """Skip phone number purchase"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return RedirectResponse(url="/clients_bots", status_code=303)

@app.get("/clients_bots", response_class=HTMLResponse)
async def clients_bots(request: Request):
    """Bot configuration page"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    client_id = client["id"]
    phone_info = phone_numbers.get(client_id, {})
    doc_info = documents.get(client_id, [])
    
    # Initialize bots if not exists
    if client_id not in subscriptions:
        subscriptions[client_id] = {
            "whatsapp": {"status": "inactive", "start_date": None, "expiry_date": None},
            "voice": {"status": "inactive", "start_date": None, "expiry_date": None},
            "web": {"status": "inactive", "start_date": None, "expiry_date": None}
        }
    
    # Initialize WhatsApp profile if not exists
    if client_id not in whatsapp_profiles:
        whatsapp_profiles[client_id] = {
            "business_name": client["business_name"],
            "address": "",
            "logo": "",
            "updated_at": datetime.now().isoformat()
        }
    
    save_data()
    
    return templates.TemplateResponse("client_bots.html", {
        "request": request,
        "client": client,
        "phone_number": phone_info.get("number", "Not purchased"),
        "has_phone": client_id in phone_numbers,
        "document_count": len(doc_info),
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="https://yourdomain.com/static/js/chat-widget.js" data-client-id="{client_id}"></script>',
        "subscriptions": subscriptions[client_id],
        "whatsapp_profile": whatsapp_profiles[client_id]
    })

@app.post("/complete_setup")
async def complete_setup(request: Request):
    """Mark client setup as completed"""
    client = get_client_from_session(request)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    client_id = client["id"]
    clients[client_id]["status"] = "completed"
    clients[client_id]["completed_at"] = datetime.now().isoformat()
    
    # Clear session
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    
    save_data()
    
    response = RedirectResponse(url="/clients", status_code=303)
    response.delete_cookie("session_id")
    return response

@app.get("/client/{client_id}", response_class=HTMLResponse)
async def client_detail(request: Request, client_id: str):
    """Client detail page with bots and data tabs"""
    if client_id not in clients:
        return RedirectResponse(url="/clients", status_code=303)
    
    client = clients[client_id]
    client_documents = documents.get(client_id, [])
    phone_info = phone_numbers.get(client_id, {})
    client_subscriptions = subscriptions.get(client_id, {})
    whatsapp_profile = whatsapp_profiles.get(client_id, {})
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "documents": client_documents,
        "phone_number": phone_info.get("number", "Not purchased"),
        "subscriptions": client_subscriptions,
        "whatsapp_profile": whatsapp_profile,
        "active_tab": "bots",
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="https://yourdomain.com/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    })

@app.get("/client/{client_id}/data", response_class=HTMLResponse)
async def client_data(request: Request, client_id: str):
    """Client data tab for PDF management"""
    if client_id not in clients:
        return RedirectResponse(url="/clients", status_code=303)
    
    client = clients[client_id]
    client_documents = documents.get(client_id, [])
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "documents": client_documents,
        "active_tab": "data"
    })

# API Routes for bot management
@app.post("/api/bot/add_months")
async def add_months(request: Request):
    """Add months to bot subscription"""
    client = get_client_from_session(request)
    if not client:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    try:
        data = await request.json()
        bot_type = data.get("bot_type")
        months = int(data.get("months", 1))
        client_id = client["id"]
        
        if client_id not in subscriptions:
            subscriptions[client_id] = {}
        
        if bot_type not in subscriptions[client_id] or not subscriptions[client_id][bot_type].get("start_date"):
            # First time subscription
            start_date = datetime.now()
            expiry_date = start_date + timedelta(days=30 * months)
            subscriptions[client_id][bot_type] = {
                "start_date": start_date.isoformat(),
                "expiry_date": expiry_date.isoformat(),
                "status": "active"
            }
        else:
            # Extend existing subscription
            current_expiry_str = subscriptions[client_id][bot_type]["expiry_date"]
            if current_expiry_str:
                current_expiry = datetime.fromisoformat(current_expiry_str)
                if current_expiry < datetime.now():
                    # Subscription expired, start from today
                    start_date = datetime.now()
                    expiry_date = start_date + timedelta(days=30 * months)
                    subscriptions[client_id][bot_type]["start_date"] = start_date.isoformat()
                else:
                    # Extend from current expiry
                    expiry_date = current_expiry + timedelta(days=30 * months)
            else:
                # No expiry date set, start from today
                start_date = datetime.now()
                expiry_date = start_date + timedelta(days=30 * months)
                subscriptions[client_id][bot_type]["start_date"] = start_date.isoformat()
            
            subscriptions[client_id][bot_type]["expiry_date"] = expiry_date.isoformat()
            subscriptions[client_id][bot_type]["status"] = "active"
        
        save_data()
        
        return JSONResponse({
            "status": "success", 
            "message": f"Added {months} months to {bot_type} bot",
            "start_date": subscriptions[client_id][bot_type]["start_date"],
            "expiry_date": subscriptions[client_id][bot_type]["expiry_date"]
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/bot/toggle")
async def toggle_bot(request: Request):
    """Activate/deactivate bot"""
    client = get_client_from_session(request)
    if not client:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    try:
        data = await request.json()
        bot_type = data.get("bot_type")
        action = data.get("action")
        client_id = client["id"]
        
        if client_id not in subscriptions or bot_type not in subscriptions[client_id]:
            return JSONResponse({"status": "error", "message": "No subscription found"})
        
        if action == "activate":
            # Check if subscription is valid
            expiry_date_str = subscriptions[client_id][bot_type].get("expiry_date")
            if not expiry_date_str:
                return JSONResponse({"status": "error", "message": "No expiry date set"})
            
            expiry_date = datetime.fromisoformat(expiry_date_str)
            if expiry_date < datetime.now():
                return JSONResponse({"status": "error", "message": "Subscription expired"})
            
            subscriptions[client_id][bot_type]["status"] = "active"
        else:
            subscriptions[client_id][bot_type]["status"] = "inactive"
        
        save_data()
        
        return JSONResponse({
            "status": "success", 
            "message": f"{bot_type} bot {action}d",
            "current_status": subscriptions[client_id][bot_type]["status"]
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/whatsapp/update_profile")
async def update_whatsapp_profile(request: Request):
    """Update WhatsApp business profile"""
    client = get_client_from_session(request)
    if not client:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    try:
        data = await request.json()
        business_name = data.get("business_name")
        address = data.get("address")
        client_id = client["id"]
        
        if client_id not in whatsapp_profiles:
            whatsapp_profiles[client_id] = {}
        
        update_data = {"updated_at": datetime.now().isoformat()}
        if business_name is not None:
            update_data["business_name"] = business_name
        if address is not None:
            update_data["address"] = address
            
        whatsapp_profiles[client_id].update(update_data)
        
        save_data()
        
        return JSONResponse({
            "status": "success", 
            "message": "WhatsApp profile updated successfully"
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# Chat API endpoint (for web chat bot)
@app.post("/api/chat/{client_id}")
async def chat_endpoint(client_id: str, request: Request):
    """Web chat bot endpoint"""
    if client_id not in clients:
        return JSONResponse({"status": "error", "message": "Client not found"})
    
    try:
        data = await request.json()
        message = data.get("message", "")
        
        # Simulated AI response (will be replaced with actual Gemini integration)
        response = f"I received your message: '{message}'. This is a simulated response from {clients[client_id]['business_name']}."
        
        return JSONResponse({
            "status": "success",
            "response": response,
            "client_id": client_id
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# Delete client API
@app.delete("/api/clients/{client_id}")
async def delete_client(client_id: str):
    """Delete client and all associated data"""
    try:
        if client_id in clients:
            del clients[client_id]
        
        if client_id in documents:
            del documents[client_id]
            
        if client_id in phone_numbers:
            del phone_numbers[client_id]
            
        if client_id in subscriptions:
            del subscriptions[client_id]
            
        if client_id in whatsapp_profiles:
            del whatsapp_profiles[client_id]
        
        # Remove from sessions
        sessions_to_remove = [sid for sid, cid in sessions.items() if cid == client_id]
        for session_id in sessions_to_remove:
            del sessions[session_id]
        
        save_data()
        
        return JSONResponse({
            "status": "success",
            "message": "Client deleted successfully"
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
