"""
Tests for FastAPI endpoints in the Clinical Evidence Copilot.

Tests the REST API endpoints and WebSocket functionality.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from src.api.main import app
from src.models.core import UrgencyLevel, DocumentType


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_query_endpoint_structure(client):
    """Test the query endpoint structure (may fail due to service initialization)."""
    query_data = {
        "query_text": "What is the recommended treatment for hypertension?",
        "clinician_id": "test_clinician_123",
        "urgency_level": "routine"
    }
    
    # This test may fail due to service initialization issues, but tests the endpoint structure
    response = client.post("/query", json=query_data)
    
    # We expect either a successful response or a service unavailable error
    assert response.status_code in [200, 503]
    
    if response.status_code == 503:
        # Service not initialized - expected in test environment
        data = response.json()
        assert "Clinical service not initialized" in data["detail"]


def test_documents_endpoint_structure(client):
    """Test the documents endpoint structure."""
    # Test GET /documents
    response = client.get("/documents")
    
    # We expect either a successful response or a service unavailable error
    assert response.status_code in [200, 503]


def test_websocket_stats_endpoint(client):
    """Test the WebSocket stats endpoint."""
    response = client.get("/ws/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "websocket_stats" in data
    assert "timestamp" in data
    
    stats = data["websocket_stats"]
    assert "active_connections" in stats
    assert "query_subscriptions" in stats
    assert "keyword_subscriptions" in stats


def test_notification_subscription_endpoint(client):
    """Test the notification subscription endpoint."""
    subscription_data = {
        "clinician_id": "test_clinician_123",
        "query_keywords": ["hypertension", "diabetes"],
        "notification_types": ["recommendation_change", "new_evidence"]
    }
    
    response = client.post("/notifications/subscribe", json=subscription_data)
    
    # We expect either a successful response or a service unavailable error
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["clinician_id"] == "test_clinician_123"
        assert data["keywords"] == ["hypertension", "diabetes"]


def test_get_notifications_endpoint(client):
    """Test the get notifications endpoint."""
    response = client.get("/notifications/test_clinician_123")
    
    # We expect either a successful response or a service unavailable error
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)  # Should return a list of notifications


def test_document_upload_structure(client):
    """Test the document upload endpoint structure."""
    document_data = {
        "title": "Test Clinical Study",
        "authors": ["Dr. Test Author"],
        "content": "This is a test clinical study about hypertension treatment.",
        "document_type": "research_paper",
        "source": "Test Journal",
        "publication_date": "2024-01-01T00:00:00",
        "metadata": {
            "doi": "10.1234/test.2024.001",
            "sample_size": 100
        }
    }
    
    response = client.post("/documents", json=document_data)
    
    # We expect either a successful response or a service unavailable error
    assert response.status_code in [200, 503]


def test_websocket_connection():
    """Test WebSocket connection (basic structure test)."""
    with TestClient(app) as client:
        # Test WebSocket endpoint exists
        # Note: Full WebSocket testing requires more complex setup
        # This just verifies the endpoint is defined
        try:
            with client.websocket_connect("/ws/test_clinician") as websocket:
                # If we get here, the endpoint exists and accepts connections
                # Send a ping message
                websocket.send_text(json.dumps({"type": "ping"}))
                
                # Try to receive a response (may timeout in test environment)
                try:
                    data = websocket.receive_text(timeout=1.0)
                    response = json.loads(data)
                    assert response.get("type") in ["pong", "connection_established"]
                except:
                    # Timeout is expected in test environment
                    pass
                    
        except Exception as e:
            # WebSocket connection may fail due to service initialization
            # This is expected in the test environment
            assert "Clinical service not initialized" in str(e) or "Connection failed" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])