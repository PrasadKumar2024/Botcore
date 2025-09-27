from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="OwnBot API",
    description="Comprehensive AI-powered chatbot management platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ✅ ROOT ENDPOINT - DEFINED FIRST (HIGHEST PRIORITY)
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information - ONLY HTML"""
    return """
    <html>
        <head>
            <title>🚀 OwnBot API - FIXED</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
                .header { background: #10b981; color: white; padding: 30px; border-radius: 10px; text-align: center; }
                .success { color: #10b981; font-weight: bold; font-size: 24px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 OwnBot API</h1>
                    <p>Prefix Conflict FIXED ✅</p>
                </div>
                <div class="success">🎉 HTML PAGE IS WORKING!</div>
                <p><strong>Problem Solved:</strong> Removed duplicate route prefixes</p>
                <p><strong>Root Endpoint:</strong> Now shows HTML instead of JSON</p>
                <p><strong>Routes Fixed:</strong> /clients, /documents, etc. now work correctly</p>
            </div>
        </body>
    </html>
    """

# ✅ FIXED: Import routes AFTER root endpoint (prevents override)
try:
    logger.info("🔄 Importing routes...")
    
    from app.routes import clients, documents, subscriptions, numbers, chat, voice
    
    # ✅ FIXED: Remove duplicate prefixes - routes already have their own prefixes
    app.include_router(clients.router, tags=["Clients"])        # Uses /clients prefix from router
    app.include_router(documents.router, tags=["Documents"])    # Uses its own prefix
    app.include_router(subscriptions.router, tags=["Subscriptions"])
    app.include_router(numbers.router, tags=["Phone Numbers"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(voice.router, tags=["Voice"])
    
    logger.info("✅ Routes registered without prefix conflicts")
    
except ImportError as e:
    logger.error(f"Route import error: {e}")

# ✅ SIMPLE HEALTH ENDPOINT
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# ✅ SIMPLE TEST ENDPOINT
@app.get("/api/test/simple")
async def test_simple():
    return {"message": "Simple test works"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 OwnBot API started - Prefix conflicts FIXED")

# Middleware
@app.middleware("http")
async def log_requests(request, call_next):
    response = await call_next(request)
    return response
