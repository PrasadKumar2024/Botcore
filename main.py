# main.py - OwnBot FastAPI Application Entry Point
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import logging
import os
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="OwnBot API",
    description="Comprehensive AI-powered chatbot management platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web chat widget
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# In-memory storage (replace with database later)
clients_data = {}
uploaded_documents = {}
purchased_numbers = {}
client_bots = {}

# Root endpoint - Serve HTML dashboard directly
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("clients.html", {"request": request})

# Clients dashboard
@app.get("/clients", response_class=HTMLResponse)
async def clients_dashboard(request: Request):
    return templates.TemplateResponse("clients.html", {"request": request})

# Step 1: Add Client Form
@app.get("/clients/add", response_class=HTMLResponse)
async def add_client_form(request: Request):
    return templates.TemplateResponse("add_client.html", {"request": request})

# Step 1: Process Client Form
@app.post("/clients/add")
async def submit_client_form(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...)
):
    # Store client data
    client_id = f"client_{len(clients_data) + 1}"
    clients_data[client_id] = {
        "id": client_id,
        "business_name": business_name,
        "business_type": business_type,
        "created_at": datetime.now().isoformat()
    }
    
    logger.info(f"New client created: {business_name} - {business_type}")
    
    # Store in session (for demo - use proper session management in production)
    request.session = {"current_client_id": client_id}
    
    # Redirect to Step 2 (Documents)
    return RedirectResponse(url="/upload_documents", status_code=303)

# Step 2: Upload Documents
@app.get("/upload_documents", response_class=HTMLResponse)
async def upload_documents_form(request: Request):
    return templates.TemplateResponse("upload_documents.html", {"request": request})

# Step 2: Process Document Uploads
@app.post("/upload_documents")
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...)
):
    # Get current client ID (in production, use proper session)
    client_id = getattr(request, 'session', {}).get('current_client_id', 'default_client')
    
    # Store uploaded files info
    uploaded_documents[client_id] = []
    for file in files:
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.now().isoformat()
        }
        uploaded_documents[client_id].append(file_info)
        logger.info(f"Uploaded document: {file.filename} for client {client_id}")
    
    # Redirect to Step 3 (Buy Number)
    return RedirectResponse(url="/buy_number", status_code=303)

# Step 2: Skip Documents
@app.post("/skip_documents")
async def skip_documents(request: Request):
    logger.info("Documents step skipped")
    # Redirect to Step 3 (Buy Number)
    return RedirectResponse(url="/buy_number", status_code=303)

# Step 3: Buy Number Page
@app.get("/buy_number", response_class=HTMLResponse)
async def buy_number_form(request: Request):
    return templates.TemplateResponse("buy_number.html", {"request": request})

# Step 3: Process Number Purchase
@app.post("/api/numbers/save")
async def save_purchased_number(request: Request):
    try:
        # Get current client ID
        client_id = getattr(request, 'session', {}).get('current_client_id', 'default_client')
        
        # In production, this would come from Twilio API
        # For demo, we'll generate a random number
        import random
        country_codes = {
            "US": "+1", "GB": "+44", "IN": "+91", 
            "CA": "+1", "AU": "+61", "DE": "+49"
        }
        
        # Generate random phone number
        country = "US"  # Default
        number = f"{country_codes.get(country, '+1')} {random.randint(200, 999)} {random.randint(200, 999)} {random.randint(1000, 9999)}"
        
        # Store purchased number
        purchased_numbers[client_id] = {
            "phone_number": number,
            "country": country,
            "purchased_at": datetime.now().isoformat()
        }
        
        logger.info(f"Number purchased for client {client_id}: {number}")
        
        return JSONResponse({
            "status": "success", 
            "message": "Number saved successfully",
            "phone_number": number
        })
        
    except Exception as e:
        logger.error(f"Error saving number: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error saving number: {str(e)}"}
        )

# Step 3: Skip Number Purchase
@app.post("/skip_number")
async def skip_number(request: Request):
    logger.info("Number purchase skipped")
    # Redirect to Step 4 (Bots)
    return RedirectResponse(url="/clients_bots", status_code=303)

# Step 4: Bots Configuration
@app.get("/clients_bots", response_class=HTMLResponse)
async def clients_bots_form(request: Request):
    # Get client data for the page
    client_id = getattr(request, 'session', {}).get('current_client_id', 'default_client')
    client_data = clients_data.get(client_id, {})
    phone_data = purchased_numbers.get(client_id, {})
    
    return templates.TemplateResponse(
        "clients_bots.html", 
        {
            "request": request,
            "client": client_data,
            "phone_number": phone_data.get('phone_number', 'Not purchased'),
            "has_phone": client_id in purchased_numbers
        }
    )

# Step 4: Activate Bot Subscription
@app.post("/api/bots/activate")
async def activate_bot_subscription(request: Request):
    try:
        client_id = getattr(request, 'session', {}).get('current_client_id', 'default_client')
        
        # Initialize bots for client if not exists
        if client_id not in client_bots:
            client_bots[client_id] = {
                "whatsapp_bot": {"active": False, "subscription_end": None},
                "voice_bot": {"active": False, "subscription_end": None},
                "web_bot": {"active": False, "subscription_end": None}
            }
        
        # In production, this would process payment and set actual dates
        # For demo, activate for 30 days
        from datetime import datetime, timedelta
        subscription_end = datetime.now() + timedelta(days=30)
        
        # Activate all bots for demo
        for bot_type in client_bots[client_id]:
            client_bots[client_id][bot_type] = {
                "active": True,
                "subscription_end": subscription_end.isoformat()
            }
        
        logger.info(f"Bots activated for client {client_id}")
        
        return JSONResponse({
            "status": "success",
            "message": "Bots activated successfully",
            "subscription_end": subscription_end.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error activating bots: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error activating bots: {str(e)}"}
        )

# Step 5: Complete Setup
@app.get("/complete", response_class=HTMLResponse)
async def complete_setup(request: Request):
    return templates.TemplateResponse("complete.html", {"request": request})

# Client Detail Page
@app.get("/clients/{client_id}", response_class=HTMLResponse)
async def client_detail(request: Request, client_id: str):
    client_data = clients_data.get(client_id, {})
    documents = uploaded_documents.get(client_id, [])
    phone_data = purchased_numbers.get(client_id, {})
    bots_data = client_bots.get(client_id, {})
    
    return templates.TemplateResponse(
        "client_detail.html",
        {
            "request": request,
            "client": client_data,
            "documents": documents,
            "phone_number": phone_data.get('phone_number', 'Not purchased'),
            "bots": bots_data
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "message": "OwnBot API is running",
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Test endpoint
@app.get("/api/test/simple")
async def test_simple():
    return {
        "message": "Simple test endpoint working",
        "status": "success", 
        "timestamp": datetime.now().isoformat()
    }

# Application information endpoint
@app.get("/api/info")
async def app_info():
    return {
        "app_name": "OwnBot",
        "version": "1.0.0",
        "description": "AI-powered chatbot management platform",
        "features": [
            "Multi-tenant client management",
            "PDF-based knowledge system", 
            "WhatsApp, Voice, and Web chat integration",
            "Subscription-based billing",
            "Twilio phone number management"
        ]
    }

# Get all clients (API endpoint)
@app.get("/api/clients")
async def get_all_clients():
    return {
        "status": "success",
        "clients": list(clients_data.values())
    }

# Get client documents (API endpoint)
@app.get("/api/clients/{client_id}/documents")
async def get_client_documents(client_id: str):
    documents = uploaded_documents.get(client_id, [])
    return {
        "status": "success",
        "client_id": client_id,
        "documents": documents
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
