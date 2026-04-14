"""
Standalone tests for document validation functionality.

Tests the document authenticity validation and credibility assessment
without importing pathway-dependent modules.
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion.validation import DocumentValidator, create_document_validator
from models.core import ParsedDocument, DocumentType


class TestDocumentValidator:
    """Test suite for DocumentValidator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = create_document_validator()
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        assert self.validator is not None
        assert len(self.validator.trusted_journals) > 0
        assert len(self.validator.quality_indicators) > 0
    
    def test_high_quality_document_validation(self):
        """Test validation of high-quality medical document."""
        # Create high-quality document
        document = ParsedDocument(
            id="doc_001",
            title="Randomized Controlled Trial of ACE Inhibitors",
            authors=["Smith, J.", "Jones, M."],
            publication_date=datetime(2023, 6, 1),
            source="Nature Medicine",
            document_type=DocumentType.CLINICAL_TRIAL,
            content="""
            This randomized controlled trial examined the effectiveness of ACE inhibitors
            in treating hypertension. The study included 2,500 patients (n=2500) with
            essential hypertension. Statistical analysis showed p<0.05 with 95% confidence
            interval. The methodology followed double-blind protocols with peer review.
            Results demonstrated significant improvement in blood pressure control.
            """,
            metadata={
                'journal': 'Nature Medicine',
                'doi': '10.1038/s41591-2023-02543-7',
                'study_methodology': 'randomized controlled trial',
                'sample_size': 2500
            },
            credibility_score=0.8
        )
        
        is_valid, validation_details = self.validator.validate_document(document)
        
        assert is_valid is True
        assert validation_details['is_authentic'] is True
        assert validation_details['credibility_score'] > 0.7
        assert len(validation_details['suspicious_content']) == 0
        assert 'methodology' in validation_details['quality_indicators']
        assert validation_details['quality_indicators']['methodology'] is True
    
    def test_low_quality_document_validation(self):
        """Test validation of low-quality document."""
        # Create low-quality document
        document = ParsedDocument(
            id="doc_002",
            title="Miracle Cure for All Diseases",
            authors=[],
            publication_date=datetime(2010, 1, 1),
            source="Unknown Blog",
            document_type=DocumentType.RESEARCH_PAPER,
            content="""
            This miracle cure will solve all your health problems! Doctors hate this
            one weird trick that big pharma doesn't want you to know. 100% effective
            guaranteed results with no side effects.
            """,
            metadata={},
            credibility_score=0.2
        )
        
        is_valid, validation_details = self.validator.validate_document(document)
        
        assert is_valid is False
        assert validation_details['credibility_score'] < 0.5
        assert len(validation_details['suspicious_content']) > 0
        assert 'miracle cure' in ' '.join(validation_details['suspicious_content']).lower()


def test_create_document_validator():
    """Test factory function for creating validator."""
    validator = create_document_validator()
    assert isinstance(validator, DocumentValidator)


if __name__ == "__main__":
    pytest.main([__file__])