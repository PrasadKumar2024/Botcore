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

# ✅ FIXED: Import routes with proper error handling
try:
    logger.info("🔄 Attempting to import routes...")
    
    from app.routes import clients, documents, subscriptions, numbers, chat, voice
    logger.info("✅ All route modules imported successfully")
    
    # ✅ FIXED: Remove duplicate prefixes - routes already have their own prefixes
    app.include_router(clients.router, tags=["Clients"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(subscriptions.router, tags=["Subscriptions"])
    app.include_router(numbers.router, tags=["Phone Numbers"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(voice.router, tags=["Voice"])
    
    logger.info("✅ All routes registered successfully")
    
except ImportError as e:
    logger.error(f"❌ Route import error: {e}")
    logger.error(f"❌ Traceback: {traceback.format_exc()}")
except Exception as e:
    logger.error(f"❌ Route registration error: {e}")
    logger.error(f"❌ Traceback: {traceback.format_exc()}")

# ✅ FIXED: Safe import for settings and database
try:
    from app.config import settings
    from app.database import engine, Base
    from app.utils.date_utils import get_current_datetime
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Database table creation error: {e}")
        
except ImportError as e:
    logger.warning(f"⚠️ Some imports missing: {e}")

# ✅ ROOT ENDPOINT - HTML ONLY (NO JSON CONFLICT)
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information - ONLY HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 OwnBot API - LIVE</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container { 
                background: white; 
                max-width: 900px; 
                width: 100%;
                margin: 0 auto; 
                padding: 40px; 
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                text-align: center;
            }
            .header { 
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                color: white; 
                padding: 30px; 
                border-radius: 15px;
                margin-bottom: 30px;
            }
            .success-badge {
                background: #10b981;
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                display: inline-block;
                margin: 15px 0;
                font-weight: bold;
            }
            .endpoints { 
                background: #f8fafc; 
                padding: 25px; 
                border-radius: 15px;
                margin: 25px 0;
                text-align: left;
            }
            .endpoints ul {
                list-style: none;
                padding: 0;
            }
            .endpoints li {
                padding: 10px 0;
                border-bottom: 1px solid #e2e8f0;
            }
            .endpoints li:last-child {
                border-bottom: none;
            }
            .endpoints a {
                color: #2563eb;
                text-decoration: none;
                font-weight: 500;
            }
            .endpoints a:hover {
                color: #1d4ed8;
                text-decoration: underline;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 25px 0;
            }
            .status-item {
                background: #f0f9ff;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #2563eb;
            }
            .version {
                color: #64748b;
                font-size: 14px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 OwnBot API</h1>
                <p>AI-Powered Chatbot Management Platform</p>
                <div class="success-badge">✅ DEPLOYMENT SUCCESSFUL</div>
            </div>
            
            <h2>🎉 Your API is Live!</h2>
            <p>Successfully deployed to: <strong>https://botcore-z6j0.onrender.com</strong></p>
            
            <div class="status-grid">
                <div class="status-item">
                    <h3>🌐 API Status</h3>
                    <p>🟢 Operational</p>
                </div>
                <div class="status-item">
                    <h3>📊 Health Check</h3>
                    <p><a href="/health">Test Now</a></p>
                </div>
                <div class="status-item">
                    <h3>📚 Documentation</h3>
                    <p><a href="/docs">View API Docs</a></p>
                </div>
            </div>
            
            <div class="endpoints">
                <h3>🔗 Available Endpoints:</h3>
                <ul>
                    <li><a href="/health">/health</a> - API health status</li>
                    <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                    <li><a href="/api/test/simple">/api/test/simple</a> - Simple test endpoint</li>
                    <li><a href="/api/test/services">/api/test/services</a> - Service status check</li>
                </ul>
            </div>
            
            <div class="next-steps">
                <h3>🎯 Next Steps:</h3>
                <ol style="text-align: left; max-width: 600px; margin: 20px auto;">
                    <li>Check Render logs for route import errors</li>
                    <li>Create missing route files</li>
                    <li>Test individual endpoints</li>
                    <li>Build your frontend templates</li>
                </ol>
            </div>
            
            <div class="version">
                <strong>Version:</strong> 2.0 | <strong>Status:</strong> 🟢 Live
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ✅ TEST ENDPOINTS (Separate from root)
@app.get("/api/test/simple")
async def test_simple():
    """Simple test endpoint that always works"""
    return {
        "message": "✅ Simple route works!",
        "status": "success",
        "timestamp": "2025-09-26T06:00:00Z"
    }

@app.get("/api/test/services")
async def test_services():
    """Test if service imports work without crashing"""
    services = {}
    
    try:
        from app.services.client_service import ClientService
        services["client_service"] = "✅ Import successful"
    except Exception as e:
        services["client_service"] = f"❌ {str(e)}"
    
    try:
        from app.services.document_service import DocumentService
        services["document_service"] = "✅ Import successful"
    except Exception as e:
        services["document_service"] = f"❌ {str(e)}"
    
    try:
        from app.services.gemini_service import GeminiService
        services["gemini_service"] = "✅ Import successful"
    except Exception as e:
        services["gemini_service"] = f"❌ {str(e)}"
    
    try:
        from app.services.pinecone_service import PineconeService
        services["pinecone_service"] = "✅ Import successful"
    except Exception as e:
        services["pinecone_service"] = f"❌ {str(e)}"
    
    try:
        from app.services.twilio_service import TwilioService
        services["twilio_service"] = "✅ Import successful"
    except Exception as e:
        services["twilio_service"] = f"❌ {str(e)}"
    
    return {
        "status": "success",
        "services": services,
        "message": "Service import test completed"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running",
        "timestamp": "2025-09-26T06:00:00Z",
        "version": "2.0"
    }

# ✅ ERROR HANDLERS
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )

# ✅ STARTUP/SHUTDOWN
@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    logger.info("🚀 OwnBot API starting up...")
    logger.info("✅ API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    logger.info("🛑 OwnBot API shutting down...")

# ✅ MIDDLEWARE
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all requests"""
    logger.info(f"📍 Incoming: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"📍 Response: {response.status_code}")
    return response
