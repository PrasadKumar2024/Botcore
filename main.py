from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime, timedelta
from typing import List, Optional
import random

#Import database and models
#import database and media:
# import database and models:
from app.database import get_db, engine, Base
from app import models, schema
from sqlalchemy import text

# Initialize FastAPI app
app = FastAPI(
    title="Ownbot",
    description="Multi-tenant Bot Management System",
    version="1.0.0"
)

# Add debug database code
@app.on_event("startup")
async def startup_event():
    print("🚀 Checking database connection...")
    
    # Test if we can connect to database
    try:
        with engine.connect() as conn:
            print("✅ Connected to Neon database!")
    except Exception as e:
        print(f"❌ Cannot connect to database: {e}")
        return
    
    # Create tables if they don't exist
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
    except Exception as e:
        print(f"❌ Cannot create tables: {e}")

# Add this debug route
@app.get("/debug-db")
async def debug_db():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
            tables = [row[0] for row in result]
        return {"tables": tables, "message": "Check Render logs for connection status"}
    except Exception as e:
        return {"error": str(e)}
# End debug code
# Initialize FastAPI app
app = FastAPI(
    title="OwnBot",
    description="Multi-tenant Bot Management System", 
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# In-memory session storage (will be replaced with database sessions)
sessions = {}

# Helper functions
def get_client_from_session(request: Request, db: Session):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    client_id = sessions[session_id]
    return db.query(models.Client).filter(models.Client.id == client_id).first()

def generate_phone_number(country_code="+91"):
    """Generate simulated phone number"""
    return f"{country_code} 9{random.randint(1000, 9999)} {random.randint(1000, 9999)}"

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Main dashboard showing all active clients"""
    clients = db.query(models.Client).filter(models.Client.status == models.ClientStatus.ACTIVE).all()
    return templates.TemplateResponse("base.html", {
        "request": request,
        "clients": clients
    })

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected", "service": "BotCore API"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

@app.get("/clients", response_class=HTMLResponse)
async def clients_list(request: Request, db: Session = Depends(get_db)):
    """Client list page"""
    clients = db.query(models.Client).all()
    return templates.TemplateResponse("clients.html", {
        "request": request,
        "clients": clients
    })

@app.get("/add_client", response_class=HTMLResponse)
async def add_client_form(request: Request):
    """Show add client form"""
    return templates.TemplateResponse("add_client.html", {"request": request})

@app.post("/clients/add")
async def clients_add(
    business_name: str = Form(...),
    business_type: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create new client and start setup session"""
    # Create client in database
    client = models.Client(
        name=business_name,  # Using business_name as name for now
        business_name=business_name,
        business_type=models.BusinessType(business_type),
        status=models.ClientStatus.ACTIVE
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    
    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = client.id
    
    response = RedirectResponse(url="/upload_documents", status_code=303)
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.get("/upload_documents", response_class=HTMLResponse)
async def upload_documents_form(request: Request, db: Session = Depends(get_db)):
    """Show PDF upload form"""
    client = get_client_from_session(request, db)
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
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Handle PDF upload during client creation"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            document = models.Document(
                client_id=client.id,
                filename=file.filename,
                stored_filename=f"{uuid.uuid4()}_{file.filename}",
                file_path=f"/uploads/{uuid.uuid4()}_{file.filename}",
                file_size=0,  # Would need actual file handling
                processed=False
            )
            db.add(document)
    
    db.commit()
    return RedirectResponse(url="/buy_number", status_code=303)

@app.post("/skip_documents")
async def skip_documents(request: Request, db: Session = Depends(get_db)):
    """Skip document upload step"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return RedirectResponse(url="/buy_number", status_code=303)

@app.get("/buy_number", response_class=HTMLResponse)
async def buy_number_form(request: Request, db: Session = Depends(get_db)):
    """Show phone number purchase form"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return templates.TemplateResponse("buy_number.html", {
        "request": request,
        "client": client
    })

@app.post("/buy_number")
async def buy_number(
    request: Request, 
    country: str = Form(...),
    db: Session = Depends(get_db)
):
    """Buy phone number (simulated)"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    # Generate simulated phone number
    phone_number = generate_phone_number()
    
    # Create phone number in database
    phone = models.PhoneNumber(
        client_id=client.id,
        number=phone_number,
        country=country,
        twilio_sid=f"SIM_{uuid.uuid4()}",  # Simulated Twilio SID
        is_active=True
    )
    db.add(phone)
    db.commit()
    
    return RedirectResponse(url="/clients_bots", status_code=303)

@app.post("/skip_number")
async def skip_number(request: Request, db: Session = Depends(get_db)):
    """Skip phone number purchase"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    return RedirectResponse(url="/clients_bots", status_code=303)

@app.get("/clients_bots", response_class=HTMLResponse)
async def clients_bots(request: Request, db: Session = Depends(get_db)):
    """Bot configuration page"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    # Get client data
    phone_number = db.query(models.PhoneNumber).filter(
        models.PhoneNumber.client_id == client.id
    ).first()
    
    documents = db.query(models.Document).filter(
        models.Document.client_id == client.id
    ).all()
    
    subscriptions = db.query(models.Subscription).filter(
        models.Subscription.client_id == client.id
    ).all()
    
    whatsapp_profile = db.query(models.WhatsAppProfile).filter(
        models.WhatsAppProfile.client_id == client.id
    ).first()
    
    # Create default subscriptions if not exist
    if not subscriptions:
        for bot_type in [models.BotType.WHATSAPP, models.BotType.VOICE, models.BotType.WEB]:
            subscription = models.Subscription(
                client_id=client.id,
                bot_type=bot_type,
                is_active=False
            )
            db.add(subscription)
        db.commit()
        # Refresh subscriptions
        subscriptions = db.query(models.Subscription).filter(
            models.Subscription.client_id == client.id
        ).all()
    
    # Create WhatsApp profile if not exists
    if not whatsapp_profile:
        whatsapp_profile = models.WhatsAppProfile(
            client_id=client.id,
            business_name=client.business_name
        )
        db.add(whatsapp_profile)
        db.commit()
        db.refresh(whatsapp_profile)
    
    return templates.TemplateResponse("client_bots.html", {
        "request": request,
        "client": client,
        "phone_number": phone_number.number if phone_number else "Not purchased",
        "has_phone": phone_number is not None,
        "document_count": len(documents),
        "chatbot_url": f"https://ownbot.chat/{client.id}",
        "embed_code": f'<script src="https://yourdomain.com/static/js/chat-widget.js" data-client-id="{client.id}"></script>',
        "subscriptions": {sub.bot_type.value: {
            "status": "active" if sub.is_active else "inactive",
            "start_date": sub.start_date,
            "expiry_date": sub.expiry_date
        } for sub in subscriptions},
        "whatsapp_profile": whatsapp_profile
    })

@app.post("/complete_setup")
async def complete_setup(request: Request, db: Session = Depends(get_db)):
    """Mark client setup as completed"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    # Clear session
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    
    response = RedirectResponse(url="/clients", status_code=303)
    response.delete_cookie("session_id")
    return response

@app.get("/client/{client_id}", response_class=HTMLResponse)
async def client_detail(request: Request, client_id: str, db: Session = Depends(get_db)):
    """Client detail page with bots and data tabs"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    documents = db.query(models.Document).filter(
        models.Document.client_id == client_id
    ).all()
    
    phone_number = db.query(models.PhoneNumber).filter(
        models.PhoneNumber.client_id == client_id
    ).first()
    
    subscriptions = db.query(models.Subscription).filter(
        models.Subscription.client_id == client_id
    ).all()
    
    whatsapp_profile = db.query(models.WhatsAppProfile).filter(
        models.WhatsAppProfile.client_id == client_id
    ).first()
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "documents": documents,
        "phone_number": phone_number.number if phone_number else "Not purchased",
        "subscriptions": {sub.bot_type.value: {
            "status": "active" if sub.is_active else "inactive",
            "start_date": sub.start_date,
            "expiry_date": sub.expiry_date
        } for sub in subscriptions},
        "whatsapp_profile": whatsapp_profile,
        "active_tab": "bots",
        "chatbot_url": f"https://ownbot.chat/{client_id}",
        "embed_code": f'<script src="https://yourdomain.com/static/js/chat-widget.js" data-client-id="{client_id}"></script>'
    })

@app.get("/client/{client_id}/data", response_class=HTMLResponse)
async def client_data(request: Request, client_id: str, db: Session = Depends(get_db)):
    """Client data tab for PDF management"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    documents = db.query(models.Document).filter(
        models.Document.client_id == client_id
    ).all()
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "documents": documents,
        "active_tab": "data"
    })

@app.post("/client/{client_id}/upload")
async def client_upload_documents(
    client_id: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload additional PDFs from Data tab"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    uploaded_count = 0
    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            document = models.Document(
                client_id=client_id,
                filename=file.filename,
                stored_filename=f"{uuid.uuid4()}_{file.filename}",
                file_path=f"/uploads/{uuid.uuid4()}_{file.filename}",
                file_size=0,  # Would need actual file handling
                processed=False
            )
            db.add(document)
            uploaded_count += 1
    
    db.commit()
    
    return JSONResponse({
        "status": "success",
        "message": f"Successfully uploaded {uploaded_count} document(s)",
        "uploaded_count": uploaded_count
    })

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a specific document"""
    try:
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        db.delete(document)
        db.commit()
        
        return JSONResponse({
            "status": "success",
            "message": "Document deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/documents/{client_id}/reprocess")
async def reprocess_documents(client_id: str, db: Session = Depends(get_db)):
    """Reprocess knowledge base for a client"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    documents = db.query(models.Document).filter(
        models.Document.client_id == client_id
    ).all()
    
    if not documents:
        return JSONResponse({
            "status": "success",
            "message": "No documents to process",
            "processed_count": 0
        })
    
    # Mark all documents as processed
    for document in documents:
        document.processed = True
        document.processed_at = datetime.utcnow()
    
    db.commit()
    
    return JSONResponse({
        "status": "success",
        "message": f"Successfully reprocessed {len(documents)} document(s)",
        "processed_count": len(documents)
    })

# API Routes for bot management
@app.post("/api/bot/add_months")
async def add_months(request: Request, db: Session = Depends(get_db)):
    """Add months to bot subscription"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    client_id = sessions[session_id]
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"})
    
    try:
        data = await request.json()
        bot_type = models.BotType(data.get("bot_type"))
        months = int(data.get("months", 1))
        
        subscription = db.query(models.Subscription).filter(
            models.Subscription.client_id == client_id,
            models.Subscription.bot_type == bot_type
        ).first()
        
        if not subscription:
            subscription = models.Subscription(
                client_id=client_id,
                bot_type=bot_type,
                is_active=True
            )
            db.add(subscription)
        
        # Calculate dates
        start_date = datetime.utcnow()
        if subscription.start_date and subscription.expiry_date:
            if subscription.expiry_date > datetime.utcnow():
                # Extend from current expiry
                start_date = subscription.start_date
                expiry_date = subscription.expiry_date + timedelta(days=30 * months)
            else:
                # Subscription expired, start fresh
                expiry_date = start_date + timedelta(days=30 * months)
        else:
            # First time subscription
            expiry_date = start_date + timedelta(days=30 * months)
        
        subscription.start_date = start_date
        subscription.expiry_date = expiry_date
        subscription.is_active = True
        subscription.bot_activated = True
        
        db.commit()
        
        return JSONResponse({
            "status": "success", 
            "message": f"Added {months} months to {bot_type.value} bot",
            "start_date": subscription.start_date.isoformat(),
            "expiry_date": subscription.expiry_date.isoformat()
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/bot/toggle")
async def toggle_bot(request: Request, db: Session = Depends(get_db)):
    """Activate/deactivate bot"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    client_id = sessions[session_id]
    
    try:
        data = await request.json()
        bot_type = models.BotType(data.get("bot_type"))
        action = data.get("action")
        
        subscription = db.query(models.Subscription).filter(
            models.Subscription.client_id == client_id,
            models.Subscription.bot_type == bot_type
        ).first()
        
        if not subscription:
            return JSONResponse({"status": "error", "message": "No subscription found"})
        
        if action == "activate":
            # Check if subscription is valid
            if not subscription.expiry_date or subscription.expiry_date < datetime.utcnow():
                return JSONResponse({"status": "error", "message": "Subscription expired or no expiry date set"})
            
            subscription.is_active = True
            subscription.bot_activated = True
        else:
            subscription.is_active = False
        
        db.commit()
        
        return JSONResponse({
            "status": "success", 
            "message": f"{bot_type.value} bot {action}d",
            "current_status": "active" if subscription.is_active else "inactive"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/whatsapp/update_profile")
async def update_whatsapp_profile(request: Request, db: Session = Depends(get_db)):
    """Update WhatsApp business profile"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return JSONResponse({"status": "error", "message": "No active session"})
    
    client_id = sessions[session_id]
    
    try:
        data = await request.json()
        business_name = data.get("business_name")
        address = data.get("address")
        
        whatsapp_profile = db.query(models.WhatsAppProfile).filter(
            models.WhatsAppProfile.client_id == client_id
        ).first()
        
        if not whatsapp_profile:
            whatsapp_profile = models.WhatsAppProfile(
                client_id=client_id,
                business_name=business_name or "",
                address=address or ""
            )
            db.add(whatsapp_profile)
        else:
            if business_name is not None:
                whatsapp_profile.business_name = business_name
            if address is not None:
                whatsapp_profile.address = address
        
        db.commit()
        
        return JSONResponse({
            "status": "success", 
            "message": "WhatsApp profile updated successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})

# Chat API endpoint (for web chat bot)
@app.post("/api/chat/{client_id}")
async def chat_endpoint(client_id: str, request: Request, db: Session = Depends(get_db)):
    """Web chat bot endpoint"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"})
    
    try:
        data = await request.json()
        message = data.get("message", "")
        session_id = data.get("session_id", str(uuid.uuid4()))
        
        # Log the message
        message_log = models.MessageLog(
            client_id=client_id,
            channel="web",
            message_text=message,
            response_text="",  # Will be filled after AI response
            timestamp=datetime.utcnow()
        )
        db.add(message_log)
        db.commit()
        
        # Simulated AI response (will be replaced with actual Gemini integration)
        response = f"I received your message: '{message}'. This is a simulated response from {client.business_name}."
        
        # Update message log with response
        message_log.response_text = response
        db.commit()
        
        return JSONResponse({
            "status": "success",
            "response": response,
            "session_id": session_id,
            "client_id": client_id
        })
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})

# Delete client API
@app.delete("/api/clients/{client_id}")
async def delete_client(client_id: str, db: Session = Depends(get_db)):
    """Delete client and all associated data"""
    try:
        client = db.query(models.Client).filter(models.Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Delete related records (cascade should handle most, but being explicit)
        db.query(models.Document).filter(models.Document.client_id == client_id).delete()
        db.query(models.PhoneNumber).filter(models.PhoneNumber.client_id == client_id).delete()
        db.query(models.Subscription).filter(models.Subscription.client_id == client_id).delete()
        db.query(models.WhatsAppProfile).filter(models.WhatsAppProfile.client_id == client_id).delete()
        db.query(models.MessageLog).filter(models.MessageLog.client_id == client_id).delete()
        db.query(models.KnowledgeChunk).filter(models.KnowledgeChunk.client_id == client_id).delete()
        db.query(models.BotSettings).filter(models.BotSettings.client_id == client_id).delete()
        
        # Delete client
        db.delete(client)
        db.commit()
        
        # Remove from sessions
        sessions_to_remove = [sid for sid, cid in sessions.items() if cid == client_id]
        for session_id in sessions_to_remove:
            del sessions[session_id]
        
        return JSONResponse({
            "status": "success",
            "message": "Client deleted successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})

# API Documentation endpoint
@app.get("/docs")
async def get_docs():
    """API documentation"""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
