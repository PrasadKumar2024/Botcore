from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime, timedelta
from typing import List, Optional
import random
import shutil
from pathlib import Path
import json

# Import database and models
from app.database import get_db, engine, Base
from app import models
from sqlalchemy import text

# Import AI Services
from app.services.gemini_service import GeminiService
from app.services.document_service import DocumentService

# Initialize FastAPI app
app = FastAPI(
    title="Ownbot",
    description="Multi-tenant Bot Management System",
    version="1.0.0"
)

# Initialize AI Services
gemini_service = GeminiService()
document_service = DocumentService()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    
    # Ensure uploads directory exists
    UPLOAD_DIR.mkdir(exist_ok=True)
    print(f"✅ Upload directory ready: {UPLOAD_DIR.absolute()}")

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

async def save_upload_file(upload_file: UploadFile, destination: Path):
    """Save uploaded file to disk"""
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes"""
    return file_path.stat().st_size if file_path.exists() else 0
'''
def create_db_session():
    """Create a new database session for background tasks"""
    from app.database import SessionLocal
    db = SessionLocal()
    return db '''

async def process_document_background(document_id: str, file_path: str, client_id: str):
    """Background task to process PDF and create knowledge chunks"""
    db = create_db_session()
    document = None
    try:
        # Safety check - ensure db is a session, not UUID
        if not hasattr(db, 'rollback'):
            print(f"❌ ERROR: 'db' parameter is not a database session. Type: {type(db)}")
            return

        # Get document from database
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            print(f"❌ Document {document_id} not found")
            return
        
        print(f"🔄 Processing document: {document.filename}")
        
        # Process document with DocumentService
        chunks = await document_service.process_document(file_path, client_id)
        
        # Save chunks to database
        for chunk_text, metadata in chunks:
            knowledge_chunk = models.KnowledgeChunk(
                client_id=client_id,
                document_id=document_id,
                chunk_text=chunk_text,
                chunk_index=metadata.get("chunk_index", 0),
                metadata=metadata
            )
            db.add(knowledge_chunk)
        
        # Mark document as processed
        document.processed = True
        document.processed_at = datetime.utcnow()
        document.processing_error = None
        db.commit()
        
        print(f"✅ Document processed: {len(chunks)} chunks created")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        # ROLLBACK FIRST!
        try:
            db.rollback()
        except Exception as rollback_error:
            print(f"❌ Rollback failed: {rollback_error}")
        
        # Update document status to indicate processing failure
        if document:
            try:
                document.processed = False
                document.processing_error = str(e)
                db.commit()
            except Exception as update_error:
                print(f"❌ Failed to update document status: {update_error}")
                try:
                    db.rollback()
                except:
                    pass
    finally:
        db.close()

def get_context_from_knowledge(client_id: str, query: str, db: Session, max_chunks: int = 5) -> str:
    """Retrieve relevant context from knowledge base for AI response"""
    try:
        # Simple keyword-based matching (can be enhanced with vector search)
        knowledge_chunks = db.query(models.KnowledgeChunk).filter(
            models.KnowledgeChunk.client_id == client_id
        ).all()
        
        if not knowledge_chunks:
            return ""
        
        # Simple relevance scoring based on keyword matching
        scored_chunks = []
        query_lower = query.lower()
        
        for chunk in knowledge_chunks:
            chunk_text_lower = chunk.chunk_text.lower()
            score = sum(1 for word in query_lower.split() if word in chunk_text_lower)
            if score > 0:
                scored_chunks.append((score, chunk.chunk_text))
        
        # Sort by relevance score and take top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        context_chunks = [chunk[1] for chunk in scored_chunks[:max_chunks]]
        
        return "\n\n".join(context_chunks)
    
    except Exception as e:
        print(f"Error getting context from knowledge base: {e}")
        return ""

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
        db.execute(text("SELECT 1"))
        
        # Check AI service
        ai_status = "configured" if gemini_service.is_configured() else "not configured"
        
        # Check file storage
        upload_dir_status = "exists" if UPLOAD_DIR.exists() else "missing"
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "ai_service": ai_status,
            "file_storage": upload_dir_status,
            "service": "BotCore API"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected", 
            "error": str(e)
        }

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
        name=business_name,
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
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Handle PDF upload during client creation with ACTUAL file storage"""
    client = get_client_from_session(request, db)
    if not client:
        return RedirectResponse(url="/clients", status_code=303)
    
    uploaded_files = []
    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            # Generate unique filename
            stored_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / stored_filename
            
            # Save file to disk
            await save_upload_file(file, file_path)
            
            # Get file size
            file_size = get_file_size(file_path)
            
            # Create document record
            document = models.Document(
                client_id=client.id,
                filename=file.filename,
                stored_filename=stored_filename,
                file_path=str(file_path),
                file_size=file_size,
                processed=False
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Schedule background processing - FIXED: Correct parameter order
            background_tasks.add_task(
                process_document_background,
                str(document.id),        # ✅ Positional argument
                str(file_path),            
                str(client.id)           # ✅ Positional argument
            )
            
            uploaded_files.append({
                "filename": file.filename,
                "size": file_size,
                "document_id": document.id
            })
            
            print(f"✅ File uploaded: {file.filename} ({file_size} bytes)")
    
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
        twilio_sid=f"SIM_{uuid.uuid4()}",
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
    
    # CREATE DEFAULT SUBSCRIPTIONS IF MISSING
    if not subscriptions:
        for bot_type in [models.BotType.WHATSAPP, models.BotType.VOICE, models.BotType.WEB]:
            subscription = models.Subscription(
                client_id=client_id,
                bot_type=bot_type,
                is_active=False
            )
            db.add(subscription)
        db.commit()
        subscriptions = db.query(models.Subscription).filter(
            models.Subscription.client_id == client_id
        ).all()
    
    whatsapp_profile = db.query(models.WhatsAppProfile).filter(
        models.WhatsAppProfile.client_id == client_id
    ).first()
    
    # CREATE WHATSAPP PROFILE IF MISSING
    if not whatsapp_profile:
        whatsapp_profile = models.WhatsAppProfile(
            client_id=client_id,
            business_name=client.business_name
        )
        db.add(whatsapp_profile)
        db.commit()
        db.refresh(whatsapp_profile)
    
    # BUILD SUBSCRIPTIONS DICT
    subscription_dict = {
        "whatsapp": {"status": "inactive", "start_date": None, "expiry_date": None},
        "voice": {"status": "inactive", "start_date": None, "expiry_date": None},
        "web": {"status": "inactive", "start_date": None, "expiry_date": None}
    }
    
    for sub in subscriptions:
        subscription_dict[sub.bot_type.value] = {
            "status": "active" if sub.is_active else "inactive",
            "start_date": sub.start_date,
            "expiry_date": sub.expiry_date
        }
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "documents": documents,
        "phone_number": phone_number.number if phone_number else "Not purchased",
        "has_phone": phone_number is not None,
        "document_count": len(documents),
        "subscriptions": subscription_dict,
        "whatsapp_profile": whatsapp_profile,
        "active_tab": "bots"
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
    
    phone_number = db.query(models.PhoneNumber).filter(
        models.PhoneNumber.client_id == client_id
    ).first()
    
    subscriptions = db.query(models.Subscription).filter(
        models.Subscription.client_id == client_id
    ).all()
    
    whatsapp_profile = db.query(models.WhatsAppProfile).filter(
        models.WhatsAppProfile.client_id == client_id
    ).first()
    
    subscription_dict = {
        "whatsapp": {"status": "inactive", "start_date": None, "expiry_date": None},
        "voice": {"status": "inactive", "start_date": None, "expiry_date": None},
        "web": {"status": "inactive", "start_date": None, "expiry_date": None}
    }
    
    for sub in subscriptions:
        subscription_dict[sub.bot_type.value] = {
            "status": "active" if sub.is_active else "inactive",
            "start_date": sub.start_date,
            "expiry_date": sub.expiry_date
        }
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "documents": documents,
        "phone_number": phone_number.number if phone_number else "Not purchased",
        "has_phone": phone_number is not None,
        "document_count": len(documents),
        "subscriptions": subscription_dict,
        "whatsapp_profile": whatsapp_profile,
        "active_tab": "data"
    })

@app.post("/client/{client_id}/upload")
async def client_upload_documents(
    client_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload additional PDFs from Data tab with ACTUAL file storage and processing"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    uploaded_count = 0
    uploaded_files = []
    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            # Generate unique filename
            stored_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / stored_filename
            
            # Save file to disk
            await save_upload_file(file, file_path)
            
            # Get file size
            file_size = get_file_size(file_path)
            
            # Create document record
            document = models.Document(
                client_id=client_id,
                filename=file.filename,
                stored_filename=stored_filename,
                file_path=str(file_path),
                file_size=file_size,
                processed=False
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Schedule background processing - FIXED: Correct parameter order
            background_tasks.add_task(
                process_document_background,
                str(document.id),        # ✅ Positional argument
                str(file_path),          # ✅ Positional argument
                str(client_id)           # ✅ Positional argument
            )
            
            uploaded_count += 1
            uploaded_files.append({
                "filename": file.filename,
                "size": file_size,
                "document_id": document.id
            })
            print(f"✅ File uploaded: {file.filename} ({file_size} bytes)")
    
    db.commit()
    
    return JSONResponse({
        "status": "success",
        "message": f"Successfully uploaded {uploaded_count} document(s)",
        "uploaded_count": uploaded_count,
        "files": uploaded_files
    })

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a specific document and its file"""
    try:
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete physical file
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ Deleted file: {file_path}")
        
        # Delete knowledge chunks
        db.query(models.KnowledgeChunk).filter(
            models.KnowledgeChunk.document_id == document_id
        ).delete()
        
        # Delete document record
        db.delete(document)
        db.commit()
        
        return JSONResponse({
            "status": "success",
            "message": "Document deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/documents/{client_id}/reprocess")
async def reprocess_documents(
    client_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
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
    
    # Delete existing knowledge chunks
    db.query(models.KnowledgeChunk).filter(
        models.KnowledgeChunk.client_id == client_id
    ).delete()
    db.commit()
    
    # Reprocess all documents - FIXED: Correct parameter order
    processed_count = 0
    for document in documents:
        document.processed = False
        document.processing_error = None
        background_tasks.add_task(
            process_document_background,
            str(document.id),           # ✅ Positional argument
            str(document.file_path),    # ✅ Positional argument
            str(client_id)              # ✅ Positional argument
        )
        processed_count += 1
    
    db.commit()
    
    return JSONResponse({
        "status": "success",
        "message": f"Reprocessing {processed_count} document(s) in background",
        "processed_count": processed_count
    })

# AI CHAT ENDPOINT - COMPLETE INTEGRATION
@app.post("/api/chat/{client_id}")
async def chat_with_bot(
    client_id: str, 
    request: Request, 
    db: Session = Depends(get_db)
):
    """Chat with bot using AI and knowledge base"""
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", str(uuid.uuid4()))
        
        if not user_message:
            return JSONResponse({"status": "error", "message": "Please provide a message."}, status_code=400)
        
        # Get client info for context
        client = db.query(models.Client).filter(models.Client.id == client_id).first()
        if not client:
            return JSONResponse({"status": "error", "message": "Client not found."}, status_code=404)
        
        # Retrieve relevant context from knowledge base
        context = get_context_from_knowledge(client_id, user_message, db)
        
        # Get conversation history (if implemented)
        conversation_history = []  # Can be enhanced to store actual conversation history
        
        # Generate AI response using Gemini
        ai_response = await gemini_service.generate_response(
            user_message=user_message,
            context=context,
            client_info={
                "business_name": client.business_name,
                "business_type": client.business_type.value
            },
            conversation_history=conversation_history
        )
        
        # Log the conversation to MessageLog table
        message_log = models.MessageLog(
            client_id=client_id,
            channel="web",
            message_text=user_message,
            response_text=ai_response,
            timestamp=datetime.utcnow()
        )
        db.add(message_log)
        db.commit()
        
        return JSONResponse({
            "status": "success",
            "response": ai_response,
            "session_id": session_id,
            "context_used": bool(context),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return JSONResponse({
            "status": "error",
            "response": "I'm having trouble responding right now. Please try again later.",
            "error": str(e)
        }, status_code=500)

# API Routes for bot management
@app.post("/api/bot/add_months")
async def add_months(request: Request, db: Session = Depends(get_db)):
    """Add months to bot subscription"""
    data = await request.json()
    client_id = data.get("client_id")
    
    if not client_id:
        session_id = request.cookies.get("session_id")
        if session_id and session_id in sessions:
            client_id = sessions[session_id]
    
    if not client_id:
        return JSONResponse({"status": "error", "message": "No client ID provided"})
    
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"})
    
    try:
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
                start_date = subscription.start_date
                expiry_date = subscription.expiry_date + timedelta(days=30 * months)
            else:
                expiry_date = start_date + timedelta(days=30 * months)
        else:
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
    data = await request.json()
    client_id = data.get("client_id")
    
    if not client_id:
        session_id = request.cookies.get("session_id")
        if session_id and session_id in sessions:
            client_id = sessions[session_id]
    
    if not client_id:
        return JSONResponse({"status": "error", "message": "No client ID provided"})
    
    try:
        bot_type = models.BotType(data.get("bot_type"))
        action = data.get("action")
        
        subscription = db.query(models.Subscription).filter(
            models.Subscription.client_id == client_id,
            models.Subscription.bot_type == bot_type
        ).first()
        
        if not subscription:
            return JSONResponse({"status": "error", "message": "No subscription found"})
        
        if action == "activate":
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
    data = await request.json()
    client_id = data.get("client_id")
    
    if not client_id:
        session_id = request.cookies.get("session_id")
        if session_id and session_id in sessions:
            client_id = sessions[session_id]
    
    if not client_id:
        return JSONResponse({"status": "error", "message": "No client ID provided"})
    
    try:
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
            "message": "WhatsApp profile updated successfully",
            "business_name": whatsapp_profile.business_name,
            "address": whatsapp_profile.address
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})

# File download endpoint
@app.get("/api/documents/{document_id}/download")
async def download_document(document_id: str, db: Session = Depends(get_db)):
    """Download a specific document"""
    document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = Path(document.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on server")
    
    return FileResponse(
        path=file_path,
        filename=document.filename,
        media_type='application/pdf'
    )

# Knowledge base status endpoint
@app.get("/api/client/{client_id}/knowledge_status")
async def get_knowledge_status(client_id: str, db: Session = Depends(get_db)):
    """Get knowledge base processing status for a client"""
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    documents = db.query(models.Document).filter(
        models.Document.client_id == client_id
    ).all()
    
    knowledge_chunks = db.query(models.KnowledgeChunk).filter(
        models.KnowledgeChunk.client_id == client_id
    ).count()
    
    processed_docs = [doc for doc in documents if doc.processed]
    processing_docs = [doc for doc in documents if not doc.processed and not doc.processing_error]
    error_docs = [doc for doc in documents if doc.processing_error]
    
    return JSONResponse({
        "status": "success",
        "total_documents": len(documents),
        "processed_documents": len(processed_docs),
        "processing_documents": len(processing_docs),
        "error_documents": len(error_docs),
        "knowledge_chunks": knowledge_chunks,
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "processed": doc.processed,
                "processing_error": doc.processing_error,
                "file_size": doc.file_size,
                "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None
            }
            for doc in documents
        ]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
