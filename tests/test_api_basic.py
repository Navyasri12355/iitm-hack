"""
Basic tests for API components without full initialization.

Tests the API structure and models without requiring full service initialization.
"""

import pytest
from datetime import datetime

def test_api_models_import():
    """Test that API models can be imported."""
    try:
        from src.api.models import (
            QueryRequest, QueryResponse, DocumentUploadRequest, 
            DocumentResponse, HealthCheckResponse, ErrorResponse
        )
        assert True  # If we get here, imports work
    except ImportError as e:
        pytest.fail(f"Failed to import API models: {e}")


def test_websocket_manager_import():
    """Test that WebSocket manager can be imported."""
    try:
        from src.api.websocket import WebSocketManager, websocket_manager
        assert websocket_manager is not None
        assert isinstance(websocket_manager, WebSocketManager)
    except ImportError as e:
        pytest.fail(f"Failed to import WebSocket manager: {e}")


def test_query_request_model():
    """Test QueryRequest model validation."""
    from src.api.models import QueryRequest
    
    # Valid request
    request = QueryRequest(
        query_text="What is the treatment for hypertension?",
        clinician_id="test_clinician_123"
    )
    
    assert request.query_text == "What is the treatment for hypertension?"
    assert request.clinician_id == "test_clinician_123"
    assert request.urgency_level.value == "routine"  # Default value


def test_document_upload_request_model():
    """Test DocumentUploadRequest model validation."""
    from src.api.models import DocumentUploadRequest
    from src.models.core import DocumentType
    
    # Valid request
    request = DocumentUploadRequest(
        title="Test Study",
        authors=["Dr. Test"],
        content="Test content",
        document_type=DocumentType.RESEARCH_PAPER,
        source="Test Journal",
        publication_date=datetime(2024, 1, 1)
    )
    
    assert request.title == "Test Study"
    assert request.document_type == DocumentType.RESEARCH_PAPER


def test_websocket_manager_basic():
    """Test basic WebSocket manager functionality."""
    from src.api.websocket import WebSocketManager
    
    manager = WebSocketManager()
    
    # Test initial state
    assert len(manager.active_connections) == 0
    assert len(manager.query_subscriptions) == 0
    assert len(manager.keyword_subscriptions) == 0
    
    # Test stats
    stats = manager.get_connection_stats()
    assert stats["active_connections"] == 0
    assert stats["query_subscriptions"] == 0
    assert stats["keyword_subscriptions"] == 0


if __name__ == "__main__":
    pytest.main([__file__])