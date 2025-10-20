from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL - using PostgreSQL (Neon)
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Create database engine with connection pooling and auto-reconnect
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them (fixes SSL drops)
    pool_recycle=3600,   # Recycle connections after 1 hour (3600 seconds)
    pool_size=10,        # Maximum number of persistent connections
    max_overflow=20,     # Maximum overflow connections beyond pool_size
    echo=False,          # Set to True for SQL query logging (debugging)
    connect_args={
        "keepalives": 1,           # Enable TCP keepalives
        "keepalives_idle": 30,     # Seconds before sending keepalive probes
        "keepalives_interval": 10, # Seconds between keepalive probes
        "keepalives_count": 5,     # Number of keepalives before connection drop
    }
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
