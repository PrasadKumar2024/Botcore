# main.py - OwnBot FastAPI Application
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import logging
import os
from datetime import datetime, timedelta
import uuid

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

# Storage (simple dicts - replace with DB later)
clients = {}
documents = {}
phone_numbers = {}
subscriptions = {}
sessions = {}

# Dashboard - Show all clients
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "clients": list(clients.values())
    })

@app.get("/clients", response_class=HTMLResponse)
async def clients_page(request: Request):
    return templates.TemplateResponse("clients.html", {
        "request": request, 
        "clients": list(clients.values())
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
        
        logger.info(f"New client created: {business_name}")
        
        # Redirect to Step 2 with session cookie
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
        # Create upload directory
        os.makedirs(f"uploads/{client_id}", exist_ok=True)
        
        # Save files
        for file in files:
            if file.content_type == "application/pdf":
                file_path = f"uploads/{client_id}/{file.filename}"
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                if client_id not in documents:
                    documents[client_id] = []
                
                documents[client_id].append({
                    "filename": file.filename,
                    "uploaded_at": datetime.now().isoformat()
                })
        
        return RedirectResponse(url="/buy_number", status_code=303)
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Error uploading documents"}
        )

@app.post("/skip_documents")
async def skip_documents_step(request: Request):
    return RedirectResponse(url="/buy_number", status_code=303)

# Step 3: Buy Number
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
        # Generate phone number (replace with Twilio later)
        import random
        phone_number = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
        
        phone_numbers[client_id] = {
            "number": phone_number,
            "purchased_at": datetime.now().isoformat()
        }
        
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
    return RedirectResponse(url="/clients_bots", status_code=303)

# Step 4: Bots Configuration
@app.get("/clients_bots", response_class=HTMLResponse)
async def bots_configuration(request: Request):
    session_id = request.cookies.get("session_id")
    client_id = sessions.get(session_id) if session_id else None
    
    if not client_id:
        return RedirectResponse(url="/clients/add", status_code=303)
    
    client = clients.get(client_id, {})
    phone_number = phone_numbers.get(client_id, {}).get("number", "Not purchased")
    
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
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    })

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "success",
        "message": "OwnBot running",
        "timestamp": datetime.now().isoformat(),
        "clients_count": len(clients)
    }

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working", "status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
