
from fastapi import APIRouter, Request, Depends, Form, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.services.client_service import ClientService
from app.services.document_service import DocumentService
from app.services.subscription_service import SubscriptionService
from app.services.twilio_service import TwilioService
from app.schemas import ClientCreate, DocumentCreate, SubscriptionCreate
from app.models import Client, BotType

router = APIRouter(prefix="/clients", tags=["clients"])
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def list_clients(request: Request, db: Session = Depends(get_db)):
    """Page 1: Dashboard with list of all clients"""
    clients = ClientService.get_all_clients(db)
    return templates.TemplateResponse("clients.html", {
        "request": request, 
        "clients": clients
    })

@router.get("/add", response_class=HTMLResponse)
async def add_client_form(request: Request):
    """Page 2: Form to add a new client"""
    return templates.TemplateResponse("add_client.html", {"request": request})

@router.post("/add", response_class=HTMLResponse)
async def create_client(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...),
    db: Session = Depends(get_db)
):
    """Process the new client form and redirect to document upload"""
    client_data = ClientCreate(
        business_name=business_name,
        business_type=business_type
    )
    client = ClientService.create_client(db, client_data)
    return RedirectResponse(url=f"/clients/{client.id}/documents", status_code=303)

@router.get("/{client_id}/documents", response_class=HTMLResponse)
async def upload_documents_form(request: Request, client_id: int, db: Session = Depends(get_db)):
    """Page 3: PDF upload page for a specific client"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return templates.TemplateResponse("upload_documents.html", {
        "request": request,
        "client": client
    })

@router.post("/{client_id}/documents", response_class=HTMLResponse)
async def upload_documents(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Process PDF uploads and redirect to number purchase"""
    # This would handle file uploads in a real implementation
    # For now, we'll just redirect
    return RedirectResponse(url=f"/clients/{client_id}/numbers", status_code=303)

@router.get("/{client_id}/numbers", response_class=HTMLResponse)
async def purchase_number_form(request: Request, client_id: int, db: Session = Depends(get_db)):
    """Page 4: Number purchase page"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return templates.TemplateResponse("buy_number.html", {
        "request": request,
        "client": client,
        "countries": TwilioService.get_available_countries()
    })

@router.post("/{client_id}/numbers", response_class=HTMLResponse)
async def purchase_number(
    request: Request,
    client_id: int,
    country_code: str = Form(...),
    db: Session = Depends(get_db)
):
    """Purchase a phone number and redirect to bot configuration"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Purchase the number
    number = TwilioService.buy_phone_number(country_code)
    
    # Save the number to the client
    ClientService.add_phone_number(db, client_id, number)
    
    return RedirectResponse(url=f"/clients/{client_id}/bots", status_code=303)

@router.get("/{client_id}/bots", response_class=HTMLResponse)
async def configure_bots(request: Request, client_id: int, db: Session = Depends(get_db)):
    """Page 5: Bot configuration page with all three bot types"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get client's phone number if exists
    phone_number = ClientService.get_phone_number(db, client_id)
    
    # Get subscriptions for each bot type
    whatsapp_sub = SubscriptionService.get_subscription(db, client_id, BotType.WHATSAPP)
    voice_sub = SubscriptionService.get_subscription(db, client_id, BotType.VOICE)
    web_sub = SubscriptionService.get_subscription(db, client_id, BotType.WEB)
    
    # Get WhatsApp profile if exists
    whatsapp_profile = ClientService.get_whatsapp_profile(db, client_id)
    
    return templates.TemplateResponse("client_bots.html", {
        "request": request,
        "client": client,
        "phone_number": phone_number,
        "whatsapp_sub": whatsapp_sub,
        "voice_sub": voice_sub,
        "web_sub": web_sub,
        "whatsapp_profile": whatsapp_profile,
        "chatbot_url": f"https://ownbot.chat/{client.business_name.lower().replace(' ', '-')}"
    })

@router.post("/{client_id}/bots/{bot_type}/subscribe", response_class=HTMLResponse)
async def add_subscription(
    request: Request,
    client_id: int,
    bot_type: str,
    months: int = Form(...),
    db: Session = Depends(get_db)
):
    """Add subscription months to a bot"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Create subscription data
    sub_data = SubscriptionCreate(
        client_id=client_id,
        bot_type=bot_type,
        months=months
    )
    
    # Add subscription
    SubscriptionService.add_subscription(db, sub_data)
    
    return RedirectResponse(url=f"/clients/{client_id}/bots", status_code=303)

@router.post("/{client_id}/whatsapp-profile", response_class=HTMLResponse)
async def update_whatsapp_profile(
    request: Request,
    client_id: int,
    business_name: str = Form(...),
    address: str = Form(...),
    db: Session = Depends(get_db)
):
    """Update WhatsApp business profile"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Update WhatsApp profile
    ClientService.update_whatsapp_profile(db, client_id, business_name, address)
    
    # Also update on Twilio
    phone_number = ClientService.get_phone_number(db, client_id)
    if phone_number:
        TwilioService.update_whatsapp_profile(phone_number.number, business_name, address)
    
    return RedirectResponse(url=f"/clients/{client_id}/bots", status_code=303)

@router.get("/{client_id}", response_class=HTMLResponse)
async def client_detail(
    request: Request, 
    client_id: int, 
    tab: str = "bots",
    db: Session = Depends(get_db)
):
    """Client detail page with tabs for Bots, Data, and Analytics"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get documents for the Data tab
    documents = []
    if tab == "data":
        documents = DocumentService.get_documents(db, client_id)
    
    # Get subscriptions for the Bots tab
    subscriptions = {}
    if tab == "bots":
        subscriptions = {
            "whatsapp": SubscriptionService.get_subscription(db, client_id, BotType.WHATSAPP),
            "voice": SubscriptionService.get_subscription(db, client_id, BotType.VOICE),
            "web": SubscriptionService.get_subscription(db, client_id, BotType.WEB)
        }
    
    return templates.TemplateResponse("client_detail.html", {
        "request": request,
        "client": client,
        "tab": tab,
        "documents": documents,
        "subscriptions": subscriptions,
        "chatbot_url": f"https://ownbot.chat/{client.business_name.lower().replace(' ', '-')}"
    })

@router.post("/{client_id}/documents/upload", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Upload a new document for an existing client"""
    # This would handle file upload in a real implementation
    # For now, we'll just redirect back to the data tab
    return RedirectResponse(url=f"/clients/{client_id}?tab=data", status_code=303)

@router.post("/{client_id}/documents/{document_id}/delete", response_class=HTMLResponse)
async def delete_document(
    request: Request,
    client_id: int,
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document"""
    DocumentService.delete_document(db, document_id)
    return RedirectResponse(url=f"/clients/{client_id}?tab=data", status_code=303)

@router.post("/{client_id}/documents/reprocess", response_class=HTMLResponse)
async def reprocess_documents(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Reprocess all documents for a client"""
    DocumentService.reprocess_documents(db, client_id)
    return RedirectResponse(url=f"/clients/{client_id}?tab=data", status_code=303)
