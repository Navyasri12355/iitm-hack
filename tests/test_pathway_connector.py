"""
Tests for Pathway document connector functionality.

Tests the real-time document monitoring, parsing, and change detection
capabilities of the PathwayDocumentConnector.
"""

import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
import pytest

from src.ingestion.pathway_connector import PathwayDocumentConnector, create_pathway_connector
from src.models.core import DocumentType


class TestPathwayDocumentConnector:
    """Test suite for PathwayDocumentConnector."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.connector = create_pathway_connector(
            documents_path=self.temp_dir,
            supported_formats=['txt', 'html', 'pdf']
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_connector_initialization(self):
        """Test that connector initializes correctly."""
        assert self.connector.documents_path == self.temp_dir
        assert 'txt' in self.connector.supported_formats
        assert Path(self.temp_dir).exists()
    
    def test_supported_format_detection(self):
        """Test file format support detection."""
        assert self.connector._is_supported_format("test.txt")
        assert self.connector._is_supported_format("test.html")
        assert self.connector._is_supported_format("test.pdf")
        assert not self.connector._is_supported_format("test.doc")
        assert not self.connector._is_supported_format("test.xyz")
    
    def test_file_hash_generation(self):
        """Test file hash generation for change detection."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content for hashing")
        
        hash1 = self.connector._get_file_hash(str(test_file))
        assert hash1
        assert len(hash1) == 32  # MD5 hash length
        
        # Same content should produce same hash
        hash2 = self.connector._get_file_hash(str(test_file))
        assert hash1 == hash2
        
        # Different content should produce different hash
        test_file.write_text("Different content")
        hash3 = self.connector._get_file_hash(str(test_file))
        assert hash1 != hash3
    
    def test_document_type_detection(self):
        """Test document type detection heuristics."""
        # Test systematic review detection
        content_sr = "This is a systematic review of clinical trials"
        doc_type = self.connector._detect_document_type("test.pdf", content_sr)
        assert doc_type == DocumentType.SYSTEMATIC_REVIEW
        
        # Test RCT detection
        content_rct = "This randomized controlled trial examined"
        doc_type = self.connector._detect_document_type("test.pdf", content_rct)
        assert doc_type == DocumentType.CLINICAL_TRIAL
        
        # Test guideline detection
        content_guide = "Clinical practice guideline for treatment"
        doc_type = self.connector._detect_document_type("test.html", content_guide)
        assert doc_type == DocumentType.GUIDELINE
        
        # Test default case
        content_default = "This is a regular research paper"
        doc_type = self.connector._detect_document_type("test.pdf", content_default)
        assert doc_type == DocumentType.RESEARCH_PAPER
    
    def test_credibility_score_calculation(self):
        """Test credibility score calculation."""
        # High credibility document
        high_cred_doc = {
            'journal': 'Nature Medicine',
            'doi': '10.1038/s41591-2023-02543-7',
            'publication_date': datetime(2023, 6, 1),
            'authors': ['Smith, J.', 'Jones, M.']
        }
        score_high = self.connector._calculate_credibility_score(high_cred_doc)
        assert score_high > 0.7
        
        # Low credibility document
        low_cred_doc = {
            'publication_date': datetime(2010, 1, 1),
            'authors': []
        }
        score_low = self.connector._calculate_credibility_score(low_cred_doc)
        assert score_low < 0.7
        
        # Score should be between 0 and 1
        assert 0 <= score_high <= 1
        assert 0 <= score_low <= 1
    
    def test_document_deletion_detection(self):
        """Test detection of deleted documents."""
        # Create test files
        file1 = Path(self.temp_dir) / "doc1.txt"
        file2 = Path(self.temp_dir) / "doc2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        # Simulate processing both files
        self.connector._processed_documents[str(file1)] = datetime.now()
        self.connector._processed_documents[str(file2)] = datetime.now()
        
        # Delete one file
        file1.unlink()
        
        # Check deletion detection
        current_files = {str(file2)}
        deleted_files = self.connector.handle_document_deletions(current_files)
        
        assert str(file1) in deleted_files
        assert str(file2) not in deleted_files
        assert str(file1) not in self.connector._processed_documents
        assert str(file2) in self.connector._processed_documents
    
    def test_processing_stats(self):
        """Test processing statistics generation."""
        stats = self.connector.get_processing_stats()
        
        assert 'total_processed' in stats
        assert 'monitored_path' in stats
        assert 'supported_formats' in stats
        assert 'last_activity' in stats
        
        assert stats['total_processed'] == 0
        assert stats['monitored_path'] == self.temp_dir
        assert stats['supported_formats'] == ['txt', 'html', 'pdf']
        assert stats['last_activity'] is None
        
        # Add a processed document
        test_file = str(Path(self.temp_dir) / "test.txt")
        self.connector._processed_documents[test_file] = datetime.now()
        
        updated_stats = self.connector.get_processing_stats()
        assert updated_stats['total_processed'] == 1
        assert updated_stats['last_activity'] is not None


def test_create_pathway_connector():
    """Test factory function for creating connector."""
    connector = create_pathway_connector()
    assert isinstance(connector, PathwayDocumentConnector)
    assert connector.documents_path  # Should have default path
    assert connector.supported_formats  # Should have default formats


if __name__ == "__main__":
    pytest.main([__file__])