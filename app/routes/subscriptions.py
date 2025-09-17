
from fastapi import APIRouter, Request, Depends, Form, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from app.database import get_db
from app.services.client_service import ClientService
from app.services.subscription_service import SubscriptionService
from app.schemas import SubscriptionCreate
from app.models import BotType

router = APIRouter(prefix="/clients/{client_id}", tags=["subscriptions"])
templates = Jinja2Templates(directory="templates")

@router.post("/subscriptions/{bot_type}/add-months", response_class=HTMLResponse)
async def add_subscription_months(
    request: Request,
    client_id: int,
    bot_type: str,
    months: int = Form(...),
    db: Session = Depends(get_db)
):
    """Add months to a bot's subscription"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Validate bot type
    if bot_type not in [bt.value for bt in BotType]:
        raise HTTPException(status_code=400, detail="Invalid bot type")
    
    # Create subscription data
    subscription_data = SubscriptionCreate(
        client_id=client_id,
        bot_type=bot_type,
        months=months
    )
    
    # Add subscription months
    subscription = SubscriptionService.add_subscription_months(db, subscription_data)
    
    # Redirect back to appropriate page
    if "bots" in str(request.url):
        return RedirectResponse(
            url=f"/clients/{client_id}/bots", 
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/clients/{client_id}?tab=bots", 
            status_code=303
        )

@router.post("/subscriptions/{bot_type}/deactivate", response_class=HTMLResponse)
async def deactivate_subscription(
    request: Request,
    client_id: int,
    bot_type: str,
    db: Session = Depends(get_db)
):
    """Deactivate a bot's subscription"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Validate bot type
    if bot_type not in [bt.value for bt in BotType]:
        raise HTTPException(status_code=400, detail="Invalid bot type")
    
    # Deactivate subscription
    SubscriptionService.deactivate_subscription(db, client_id, bot_type)
    
    # Redirect back to appropriate page
    if "bots" in str(request.url):
        return RedirectResponse(
            url=f"/clients/{client_id}/bots", 
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/clients/{client_id}?tab=bots", 
            status_code=303
        )

@router.post("/subscriptions/{bot_type}/reactivate", response_class=HTMLResponse)
async def reactivate_subscription(
    request: Request,
    client_id: int,
    bot_type: str,
    db: Session = Depends(get_db)
):
    """Reactivate a bot's subscription"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Validate bot type
    if bot_type not in [bt.value for bt in BotType]:
        raise HTTPException(status_code=400, detail="Invalid bot type")
    
    # Reactivate subscription
    SubscriptionService.reactivate_subscription(db, client_id, bot_type)
    
    # Redirect back to appropriate page
    if "bots" in str(request.url):
        return RedirectResponse(
            url=f"/clients/{client_id}/bots", 
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/clients/{client_id}?tab=bots", 
            status_code=303
        )

@router.get("/subscriptions/check-expiry", response_class=HTMLResponse)
async def check_subscription_expiry(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Check and update subscription expiry status (for cron jobs)"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Check and update expiry status for all bot types
    SubscriptionService.check_and_update_expiry(db, client_id)
    
    return HTMLResponse(content="Expiry status checked and updated")

@router.get("/subscriptions", response_class=HTMLResponse)
async def get_subscriptions(
    request: Request,
    client_id: int,
    db: Session = Depends(get_db)
):
    """Get all subscriptions for a client (for API)"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    subscriptions = SubscriptionService.get_all_subscriptions(db, client_id)
    
    # Return JSON response for API calls
    return {
        "client_id": client_id,
        "subscriptions": [
            {
                "bot_type": sub.bot_type,
                "start_date": sub.start_date.isoformat() if sub.start_date else None,
                "expiry_date": sub.expiry_date.isoformat() if sub.expiry_date else None,
                "active": sub.active,
                "status": "ACTIVE" if sub.active and sub.expiry_date and sub.expiry_date > datetime.now() else "EXPIRED" if sub.expiry_date and sub.expiry_date <= datetime.now() else "INACTIVE"
            }
            for sub in subscriptions
        ]
    }

@router.post("/subscriptions/{bot_type}/set-custom", response_class=HTMLResponse)
async def set_custom_subscription(
    request: Request,
    client_id: int,
    bot_type: str,
    start_date: str = Form(...),
    expiry_date: str = Form(...),
    db: Session = Depends(get_db)
):
    """Set custom subscription dates (for manual adjustments)"""
    client = ClientService.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Validate bot type
    if bot_type not in [bt.value for bt in BotType]:
        raise HTTPException(status_code=400, detail="Invalid bot type")
    
    # Parse dates
    try:
        start_dt = datetime.fromisoformat(start_date)
        expiry_dt = datetime.fromisoformat(expiry_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    # Set custom subscription
    SubscriptionService.set_custom_subscription(
        db, client_id, bot_type, start_dt, expiry_dt
    )
    
    # Redirect back to appropriate page
    if "bots" in str(request.url):
        return RedirectResponse(
            url=f"/clients/{client_id}/bots", 
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/clients/{client_id}?tab=bots", 
            status_code=303
  )
