# main.py - OwnBot FastAPI Application Entry Point
from fastapi import FastAPI, Request
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

# Root endpoint - Redirect to clients dashboard WITH trailing slash
@app.get("/")
async def root():
    return RedirectResponse(url="/clients/")

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

# ✅ Import your routes - they handle all specific paths
try:
    from app.routes import clients, documents, subscriptions, numbers, chat, voice
    
    # These routers handle their own specific routes:
    app.include_router(clients.router, tags=["Clients"])        # Handles /clients/*
    app.include_router(documents.router, tags=["Documents"])    # Handles /documents/*  
    app.include_router(subscriptions.router, tags=["Subscriptions"])
    app.include_router(numbers.router, tags=["Phone Numbers"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(voice.router, tags=["Voice"])
    
    logger.info("✅ All routes registered successfully")
    
except ImportError as e:
    logger.error(f"Route import error: {e}")

# ✅ Add a catch-all route for undefined paths to prevent 404 errors
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return RedirectResponse(url="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
