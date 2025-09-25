# tests/__init__.py
# This file makes the tests directory a Python package

# Import test modules to make them discoverable
from . import test_clients
from . import test_documents
from . import test_subscriptions

# Optional: You can define test fixtures or common utilities here
import pytest
from app.database import get_db
from app.main import app
from fastapi.testclient import TestClient

# Test client fixture for reuse across tests
@pytest.fixture(scope="module")
def test_client():
    """Create a test client for FastAPI application"""
    with TestClient(app) as client:
        yield client

# Database session fixture
@pytest.fixture(scope="function")
def db_session():
    """Create a database session for testing"""
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

# Common test data
TEST_CLIENT_DATA = {
    "name": "Test Client",
    "business_type": "Retail",
    "status": "active"
}

TEST_SUBSCRIPTION_DATA = {
    "bot_type": "whatsapp",
    "months": 3
}

TEST_DOCUMENT_DATA = {
    "filename": "test.pdf",
    "processed": False
}

# Export for easy importing in tests
__all__ = [
    "test_client",
    "db_session", 
    "TEST_CLIENT_DATA",
    "TEST_SUBSCRIPTION_DATA",
    "TEST_DOCUMENT_DATA"
]
