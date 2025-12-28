"""
Tests for document validation functionality.

Tests the document authenticity validation and credibility assessment
capabilities of the DocumentValidator.
"""

import pytest
from datetime import datetime, timedelta
from src.ingestion.validation import DocumentValidator, create_document_validator
from src.models.core import ParsedDocument, DocumentType


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
    
    def test_authenticity_scoring(self):
        """Test authenticity scoring mechanism."""
        # High authenticity document
        high_auth_doc = ParsedDocument(
            id="doc_003",
            title="Clinical Study",
            authors=["Dr. Smith"],
            publication_date=datetime(2023, 1, 1),
            source="test",
            document_type=DocumentType.CLINICAL_TRIAL,
            content="University hospital study with DOI and PMID references",
            metadata={
                'journal': 'NEJM',
                'doi': '10.1056/test',
                'pmid': '12345678'
            },
            credibility_score=0.8
        )
        
        authenticity_score = self.validator._check_authenticity(high_auth_doc)
        assert authenticity_score > 0.5
        
        # Low authenticity document
        low_auth_doc = ParsedDocument(
            id="doc_004",
            title="Blog Post",
            authors=[],
            publication_date=datetime(2010, 1, 1),
            source="test",
            document_type=DocumentType.RESEARCH_PAPER,
            content="Random blog post with no credentials",
            metadata={},
            credibility_score=0.2
        )
        
        authenticity_score = self.validator._check_authenticity(low_auth_doc)
        assert authenticity_score < 0.5
    
    def test_content_quality_assessment(self):
        """Test content quality assessment."""
        # High quality content
        high_quality_content = """
        This randomized controlled trial examined the effectiveness of treatment.
        The study methodology included double-blind protocols with statistical
        analysis showing p<0.05. Sample size was n=1000 with confidence intervals
        reported. Peer review process was followed with proper citations.
        """ * 20  # Make it substantial
        
        high_quality_doc = ParsedDocument(
            id="doc_005",
            title="Quality Study",
            authors=["Author"],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.CLINICAL_TRIAL,
            content=high_quality_content,
            metadata={'medical_term_counts': {'methodology': 5, 'statistics': 3}},
            credibility_score=0.8
        )
        
        quality_score = self.validator._assess_content_quality(high_quality_doc)
        assert quality_score > 0.5
    
    def test_suspicious_content_detection(self):
        """Test detection of suspicious content patterns."""
        suspicious_content = """
        This miracle cure is guaranteed to work 100% of the time!
        Doctors hate this one weird trick that big pharma doesn't want you to know.
        It's a conspiracy by the medical establishment to keep you sick.
        """
        
        suspicious_doc = ParsedDocument(
            id="doc_006",
            title="Suspicious Content",
            authors=[],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.RESEARCH_PAPER,
            content=suspicious_content,
            metadata={},
            credibility_score=0.1
        )
        
        suspicious_items = self.validator._detect_suspicious_content(suspicious_doc)
        assert len(suspicious_items) > 0
        assert any('miracle' in item.lower() for item in suspicious_items)
    
    def test_quality_indicators_detection(self):
        """Test detection of quality indicators."""
        quality_content = """
        This randomized controlled trial used double-blind methodology.
        Statistical analysis included p-values and confidence intervals.
        The study was peer-reviewed and published in a medical journal.
        Sample size was n=500 with proper statistical significance testing.
        """
        
        quality_doc = ParsedDocument(
            id="doc_007",
            title="Quality Indicators Test",
            authors=["Author"],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.CLINICAL_TRIAL,
            content=quality_content,
            metadata={},
            credibility_score=0.8
        )
        
        indicators = self.validator._get_quality_indicators(quality_doc)
        
        assert indicators['methodology'] is True
        assert indicators['statistics'] is True
        assert indicators['peer_review'] is True
        assert indicators['sample_size'] is True
    
    def test_credibility_score_calculation(self):
        """Test overall credibility score calculation."""
        document = ParsedDocument(
            id="doc_008",
            title="Test Document",
            authors=["Author"],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.SYSTEMATIC_REVIEW,  # High credibility type
            content="Quality medical content",
            metadata={},
            credibility_score=0.8
        )
        
        # Test with high scores and no suspicious content
        credibility = self.validator._calculate_credibility_score(
            document, 0.8, 0.7, 0
        )
        assert credibility > 0.7
        
        # Test with suspicious content penalty
        credibility_with_penalty = self.validator._calculate_credibility_score(
            document, 0.8, 0.7, 3
        )
        assert credibility_with_penalty < credibility
    
    def test_validation_recommendations(self):
        """Test generation of validation recommendations."""
        low_quality_doc = ParsedDocument(
            id="doc_009",
            title="Low Quality",
            authors=[],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.RESEARCH_PAPER,
            content="Short content",
            metadata={},
            credibility_score=0.3
        )
        
        recommendations = self.validator._generate_recommendations(
            low_quality_doc, 0.3, 0.2, ['suspicious item']
        )
        
        assert len(recommendations) > 0
        assert any('authenticity' in rec.lower() for rec in recommendations)
        assert any('author' in rec.lower() for rec in recommendations)


def test_create_document_validator():
    """Test factory function for creating validator."""
    validator = create_document_validator()
    assert isinstance(validator, DocumentValidator)


if __name__ == "__main__":
    pytest.main([__file__])