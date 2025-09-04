
from fastapi import FastAPI
from app.routes import clients, pdfs  # We will create these next
from app.database import engine, Base

# Create all database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="OwnBot API", version="0.1.0")

# Include our future routers
# app.include_router(clients.router, prefix="/api/clients", tags=["clients"])
# app.include_router(pdfs.router, prefix="/api/pdfs", tags=["pdfs"])

@app.get("/")
def read_root():
    return {"message": "OwnBot API is running!", "status": "success"}
# Add your other routes and functionality here
