# app/routes/__init__.py

from .clients import router as clients_router
from .documents import router as documents_router
#from .subscriptions import router as subscriptions_router
#from .numbers import router as numbers_router
from .chat import router as chat_router
#from .voice import router as voice_router

# Import all route modules to ensure they're registered
__all__ = [
    "clients_router",
    "documents_router", 
   # "subscriptions_router",
   # "numbers_router",
    "chat_router",
   # "voice_router"
]

# Optional: You can also create a function to register all routes
def register_routes(app):
    """Register all route blueprints with the main application"""
    app.include_router(clients_router, prefix="/clients", tags=["clients"])
    app.include_router(documents_router, prefix="/documents", tags=["documents"])
    app.include_router(subscriptions_router, prefix="/subscriptions", tags=["subscriptions"])
#    app.include_router(numbers_router, prefix="/numbers", tags=["numbers"])
   # app.include_router(chat_router, prefix="/chat", tags=["chat"])
#    app.include_router(voice_router, prefix="/voice", tags=["voice"])
