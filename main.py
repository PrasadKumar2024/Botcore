# main.py - OwnBot FastAPI Application Entry Point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
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

# HTML Dashboard - Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>OwnBot Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
                .header { background: #10b981; color: white; padding: 30px; border-radius: 10px; text-align: center; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
                a { color: #10b981; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 OwnBot Dashboard</h1>
                    <p>AI-powered chatbot management platform</p>
                </div>
                
                <div class="endpoint">
                    <h3>✅ System Status: Running</h3>
                    <p><strong>Version:</strong> 1.0.0</p>
                    <p><strong>Timestamp:</strong> """ + datetime.now().isoformat() + """</p>
                </div>

                <div class="endpoint">
                    <h3>📊 Available Endpoints:</h3>
                    <ul>
                        <li><a href="/health" target="_blank">/health</a> - Health check (JSON)</li>
                        <li><a href="/api/test/simple" target="_blank">/api/test/simple</a> - Test endpoint (JSON)</li>
                        <li><a href="/api/info" target="_blank">/api/info</a> - App information (JSON)</li>
                        <li><a href="/docs" target="_blank">/docs</a> - API Documentation</li>
                    </ul>
                </div>

                <div class="endpoint">
                    <h3>🔧 Features:</h3>
                    <ul>
                        <li>Multi-tenant client management</li>
                        <li>PDF-based knowledge system</li>
                        <li>WhatsApp, Voice, and Web chat integration</li>
                        <li>Subscription-based billing</li>
                        <li>Twilio phone number management</li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

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

# Import your routes
try:
    from app.routes import clients, documents, subscriptions, numbers, chat, voice
    
    app.include_router(clients.router, tags=["Clients"])
    app.include_router(documents.router, tags=["Documents"]) 
    app.include_router(subscriptions.router, tags=["Subscriptions"])
    app.include_router(numbers.router, tags=["Phone Numbers"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(voice.router, tags=["Voice"])
    
    logger.info("✅ All routes registered successfully")
    
except ImportError as e:
    logger.error(f"Route import error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
