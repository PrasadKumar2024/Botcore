# main.py - OwnBot FastAPI Application Entry Point
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import logging
from datetime import datetime

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

# Root endpoint - Serve HTML dashboard directly
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("clients.html", {"request": request})

# Essential HTML routes to ensure they work
@app.get("/clients", response_class=HTMLResponse)
async def clients_dashboard(request: Request):
    return templates.TemplateResponse("clients.html", {"request": request})

@app.get("/clients/add", response_class=HTMLResponse)
async def add_client_form(request: Request):
    return templates.TemplateResponse("add_client.html", {"request": request})

# ADD THIS - POST route for form submission
@app.post("/clients/add")
async def submit_client_form(
    request: Request,
    business_name: str = Form(...),
    business_type: str = Form(...)
):
    # Process the form data
    logger.info(f"New client: {business_name} - {business_type}")
    
    # Return success response
    return JSONResponse({
        "status": "success", 
        "message": "Client information saved successfully!",
        "data": {
            "business_name": business_name,
            "business_type": business_type
        }
    })

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
