from fastapi import APIRouter, Request, Depends, Form, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.services.client_service import ClientService
from app.services.twilio_service import TwilioService
from app.schemas import PhoneNumberCreate

router = APIRouter(prefix="/clients/{client_id}", tags=["numbers"])
templates = Jinja2Templates(directory="templates")

@router.get("/numbers", response_class=HTMLResponse)
async def purchase_number_form(
    request: Request, 
    client_id: int, 
    db: Session = Depends(get_db)
):
    """Page 4: Phone number purchase page"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get available countries for phone number purchase
    available_countries = TwilioService.get_available_countries()
    
    # Check if client already has a phone number
    existing_number = ClientService.get_phone_number(db, client_id)
    
    return templates.TemplateResponse("buy_number.html", {
        "request": request,
        "client": client,
        "countries": available_countries,
        "existing_number": existing_number
    })

@router.post("/numbers", response_class=HTMLResponse)
async def purchase_number(
    request: Request,
    client_id: int,
    country_code: str = Form(...),
    area_code: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Purchase a phone number and redirect to bot configuration"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    try:
        # Purchase phone number from Twilio
        phone_number = TwilioService.buy_phone_number(country_code, area_code)
        
        # Save phone number to database
        phone_data = PhoneNumberCreate(
            number=phone_number["phone_number"],
            country=phone_number["country"],
            twilio_sid=phone_number["sid"],
            client_id=client_id
        )
        
        ClientService.add_phone_number(db, phone_data)
        
        # Redirect to bot configuration page
        return RedirectResponse(
            url=f"/clients/{client_id}/bots", 
            status_code=303
        )
        
    except Exception as e:
        # Handle errors (e.g., no numbers available, Twilio API error)
        available_countries = TwilioService.get_available_countries()
        
        return templates.TemplateResponse("buy_number.html", {
            "request": request,
            "client": client,
            "countries": available_countries,
            "error": f"Failed to purchase number: {str(e)}"
        })

@router.post("/numbers/skip", response_class=HTMLResponse)
async def skip_number_purchase(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Skip phone number purchase and redirect to bot configuration"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Redirect to bot configuration page
    return RedirectResponse(
        url=f"/clients/{client_id}/bots", 
        status_code=303
    )

@router.post("/numbers/release", response_class=HTMLResponse)
async def release_phone_number(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Release a phone number (from client detail page)"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get the phone number to release
    phone_number = ClientService.get_phone_number(db, client_id)
    if not phone_number:
        raise HTTPException(status_code=404, detail="No phone number found")
    
    try:
        # Release phone number from Twilio
        TwilioService.release_phone_number(phone_number.twilio_sid)
        
        # Remove phone number from database
        ClientService.remove_phone_number(db, client_id)
        
        # Redirect back to client detail page
        return RedirectResponse(
            url=f"/clients/{client_id}?tab=bots", 
            status_code=303
        )
        
    except Exception as e:
        # Handle errors
        return RedirectResponse(
            url=f"/clients/{client_id}?tab=bots&error={str(e)}", 
            status_code=303
        )

@router.get("/numbers/available", response_class=HTMLResponse)
async def check_available_numbers(
    request: Request,
    client_id: int,
    country_code: str,
    area_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Check available phone numbers for a country/area code"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    try:
        # Get available numbers from Twilio
        available_numbers = TwilioService.search_available_numbers(
            country_code, 
            area_code
        )
        
        return templates.TemplateResponse("_number_list.html", {
            "request": request,
            "numbers": available_numbers
        })
        
    except Exception as e:
        return templates.TemplateResponse("_number_list.html", {
            "request": request,
            "error": f"Failed to search numbers: {str(e)}",
            "numbers": []
        })
