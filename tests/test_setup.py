"""Test basic project setup and configuration."""

import pytest
from src.config import get_settings, Settings


def test_settings_can_be_imported():
    """Test that settings can be imported and instantiated."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_default_configuration_values():
    """Test that default configuration values are set correctly."""
    settings = get_settings()
    
    # Test default values
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.debug is False
    assert settings.documents_path == "./data/documents"
    assert settings.max_document_size_mb == 50
    assert settings.response_timeout_seconds == 30
    assert settings.enable_hipaa_logging is True
    assert settings.log_level == "INFO"


def test_evidence_hierarchy_configuration():
    """Test that evidence hierarchy is properly configured."""
    settings = get_settings()
    
    hierarchy = settings.evidence_hierarchy
    assert "systematic_review" in hierarchy
    assert "randomized_controlled_trial" in hierarchy
    assert hierarchy["systematic_review"] < hierarchy["randomized_controlled_trial"]
    assert hierarchy["randomized_controlled_trial"] < hierarchy["case_report"]


def test_supported_formats():
    """Test that supported document formats are configured."""
    settings = get_settings()
    
    formats = settings.supported_formats
    assert "pdf" in formats
    assert "html" in formats
    assert "xml" in formats