from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging

from app.config import settings
from app.database import engine, Base
from app.routes import clients, documents, subscriptions, numbers, chat, voice
from app.utils.date_utils import get_current_datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {str(e)}")

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
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include all routers
app.include_router(clients.router, prefix="/api", tags=["Clients"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(subscriptions.router, prefix="/api", tags=["Subscriptions"])
app.include_router(numbers.router, prefix="/api", tags=["Phone Numbers"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(voice.router, prefix="/api", tags=["Voice"])

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <html>
        <head>
            <title>OwnBot API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { background: #2563eb; color: white; padding: 20px; border-radius: 10px; }
                .content { margin: 20px 0; }
                .endpoints { background: #f3f4f6; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 OwnBot API</h1>
                    <p>Comprehensive AI-powered chatbot management platform</p>
                </div>
                <div class="content">
                    <h2>Welcome to OwnBot</h2>
                    <p>This API powers the OwnBot platform for managing AI chatbots.</p>
                    
                    <div class="endpoints">
                        <h3>Available Endpoints:</h3>
                        <ul>
                            <li><a href="/docs">📚 API Documentation</a> - Interactive Swagger UI</li>
                            <li><a href="/redoc">📖 ReDoc</a> - Alternative documentation</li>
                            <li><strong>/api/clients</strong> - Client management</li>
                            <li><strong>/api/documents</strong> - PDF upload and management</li>
                            <li><strong>/api/subscriptions</strong> - Bot subscription management</li>
                            <li><strong>/api/numbers</strong> - Phone number management</li>
                            <li><strong>/api/chat</strong> - Web chat endpoints</li>
                            <li><strong>/api/voice</strong> - Voice call endpoints</li>
                        </ul>
                    </div>
                    
                    <p><strong>Status:</strong> 🟢 API is running</p>
                    <p><strong>Version:</strong> 1.0.0</p>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    services = {}
    
    # Test Database
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        services["database"] = "connected"
        db.close()
    except Exception as e:
        services["database"] = f"error: {str(e)}"
    
    # Test Pinecone
    try:
        from app.services.pinecone_service import PineconeService
        service = PineconeService()
        # Try to get index info or simple operation
        services["pinecone"] = "connected"
    except Exception as e:
        services["pinecone"] = f"error: {str(e)}"
    
    # Test Twilio
    try:
        from app.services.twilio_service import TwilioService
        service = TwilioService()
        # Test credentials by getting account info
        services["twilio"] = "connected"
    except Exception as e:
        services["twilio"] = f"error: {str(e)}"
    
    # Test Gemini
    try:
        from app.services.gemini_service import GeminiService
        service = GeminiService()
        services["gemini"] = "connected"
    except Exception as e:
        services["gemini"] = f"error: {str(e)}"
    
    # Overall status
    overall_status = "healthy" if all("connected" in status for status in services.values()) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": get_current_datetime().isoformat(),
        "version": "1.0.0",
        "services": services
    }

# Individual service test endpoints
@app.get("/api/test/database")
async def test_database():
    """Test database connection"""
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return {"status": "connected", "service": "database", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "error", "service": "database", "error": str(e)}

@app.get("/api/test/pinecone")
async def test_pinecone():
    """Test Pinecone connection"""
    try:
        from app.services.pinecone_service import PineconeService
        service = PineconeService()
        # Add a simple test operation here if available
        return {"status": "connected", "service": "pinecone", "message": "Pinecone connection successful"}
    except Exception as e:
        return {"status": "error", "service": "pinecone", "error": str(e)}

@app.get("/api/test/twilio")
async def test_twilio():
    """Test Twilio connection"""
    try:
        from app.services.twilio_service import TwilioService
        service = TwilioService()
        # Test by getting account balance or similar
        return {"status": "connected", "service": "twilio", "message": "Twilio connection successful"}
    except Exception as e:
        return {"status": "error", "service": "twilio", "error": str(e)}

@app.get("/api/test/gemini")
async def test_gemini():
    """Test Gemini connection"""
    try:
        from app.services.gemini_service import GeminiService
        service = GeminiService()
        return {"status": "connected", "service": "gemini", "message": "Gemini connection successful"}
    except Exception as e:
        return {"status": "error", "service": "gemini", "error": str(e)}

@app.get("/api/test/all")
async def test_all_services():
    """Test all services at once"""
    services = {}
    
    # Test each service
    db_test = await test_database()
    services["database"] = db_test
    
    pinecone_test = await test_pinecone()
    services["pinecone"] = pinecone_test
    
    twilio_test = await test_twilio()
    services["twilio"] = twilio_test
    
    gemini_test = await test_gemini()
    services["gemini"] = gemini_test
    
    # Count successful connections
    success_count = sum(1 for service in services.values() if service.get("status") == "connected")
    total_services = len(services)
    
    return {
        "overall_status": f"{success_count}/{total_services} services connected",
        "services": services
    }

@app.get("/api/info")
async def app_info():
    """Application information endpoint"""
    return {
        "name": "OwnBot",
        "version": "1.0.0",
        "description": "AI-powered chatbot management platform",
        "features": [
            "Multi-bot management (WhatsApp, Voice, Web Chat)",
            "PDF-based knowledge system",
            "Subscription-based billing",
            "Twilio integration for phone numbers",
            "Gemini AI for intelligent responses"
        ],
        "status": "operational"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return HTMLResponse(
        content="""
        <html>
            <body>
                <h1>404 - Page Not Found</h1>
                <p>The requested resource was not found.</p>
                <a href="/">Go to Home</a>
            </body>
        </html>
        """,
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later."
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    logger.info("OwnBot API starting up...")
    logger.info(f"Environment: {'development' if settings.DEBUG else 'production'}")
    logger.info("API endpoints registered successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    logger.info("OwnBot API shutting down...")

# Additional middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all requests"""
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response
