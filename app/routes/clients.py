from fastapi import APIRouter, Request, Depends, Form, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.services.client_service import ClientService
from app.schemas import ClientCreate

router = APIRouter(prefix="/clients", tags=["clients"])
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def list_clients(request: Request, db: Session = Depends(get_db)):
    """
    Page 1: Dashboard - Display all existing clients with +Add New Client button
    """
    try:
        clients = ClientService.get_all_clients(db)
        return templates.TemplateResponse("clients.html", {
            "request": request,
            "clients": clients
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clients: {str(e)}")

@router.get("/add", response_class=HTMLResponse)
async def add_client_form(request: Request):
    """
    Page 2: Show the form to add a new client
    """
    return templates.TemplateResponse("add_client.html", {"request": request})

@router.post("/add", response_class=HTMLResponse)
async def create_client(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Process the new client form and redirect to document upload
    """
    try:
        # Validate input
        if not business_name.strip():
            raise HTTPException(status_code=400, detail="Business name is required")
        
        if not business_type:
            raise HTTPException(status_code=400, detail="Business type is required")
        
        # Create client data
        client_data = ClientCreate(
            business_name=business_name.strip(),
            business_type=business_type
        )
        
        # Create client
        client = ClientService.create_client(db, client_data)
        
        # Redirect to document upload page
        return RedirectResponse(
            url=f"/clients/{client.id}/documents",
            status_code=303
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating client: {str(e)}")

@router.get("/{client_id}", response_class=HTMLResponse)
async def client_detail(
    request: Request,
    client_id: int,
    tab: str = "bots",
    db: Session = Depends(get_db)
):
    """
    Client detail page with tabs for Bots, Data, and Analytics
    """
    try:
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get additional data based on the selected tab
        context = {
            "request": request,
            "client": client,
            "tab": tab
        }
        
        if tab == "data":
            # Get documents for the data tab
            documents = ClientService.get_client_documents(db, client_id)
            context["documents"] = documents
            
        elif tab == "bots":
            # Get subscriptions and phone number for bots tab
            subscriptions = ClientService.get_client_subscriptions(db, client_id)
            phone_number = ClientService.get_client_phone_number(db, client_id)
            whatsapp_profile = ClientService.get_whatsapp_profile(db, client_id)
            
            context["subscriptions"] = subscriptions
            context["phone_number"] = phone_number
            context["whatsapp_profile"] = whatsapp_profile
            context["chatbot_url"] = f"https://ownbot.chat/{client.business_name.lower().replace(' ', '-')}"
        
        return templates.TemplateResponse("client_detail.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading client details: {str(e)}")

@router.get("/{client_id}/edit", response_class=HTMLResponse)
async def edit_client_form(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """
    Show form to edit client information
    """
    try:
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        return templates.TemplateResponse("edit_client.html", {
            "request": request,
            "client": client
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading edit form: {str(e)}")

@router.post("/{client_id}/edit", response_class=HTMLResponse)
async def update_client(
    request: Request,
    client_id: int,
    business_name: str = Form(...),
    business_type: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Update client information
    """
    try:
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Validate input
        if not business_name.strip():
            raise HTTPException(status_code=400, detail="Business name is required")
        
        # Update client
        updated_client = ClientService.update_client(
            db, client_id, business_name.strip(), business_type
        )
        
        # Redirect back to client detail page
        return RedirectResponse(
            url=f"/clients/{client_id}",
            status_code=303
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating client: {str(e)}")

@router.post("/{client_id}/delete", response_class=HTMLResponse)
async def delete_client(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a client and all associated data
    """
    try:
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Delete client and all associated data
        ClientService.delete_client(db, client_id)
        
        # Redirect to clients list
        return RedirectResponse(
            url="/clients",
            status_code=303
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting client: {str(e)}")

@router.get("/{client_id}/status", response_class=HTMLResponse)
async def update_client_status(
    request: Request,
    client_id: int,
    status: str,
    db: Session = Depends(get_db)
):
    """
    Update client status (active/inactive)
    """
    try:
        client = ClientService.get_client(db, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        if status not in ["active", "inactive"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        # Update client status
        ClientService.update_client_status(db, client_id, status)
        
        # Redirect back to client detail page
        return RedirectResponse(
            url=f"/clients/{client_id}",
            status_code=303
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating client status: {str(e)}")
