"""
Tests for core data models.

Tests basic validation and serialization functionality for all data models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models import (
    DocumentType,
    EvidenceLevel,
    UrgencyLevel,
    PatientContext,
    ParsedDocument,
    ClinicalQuery,
    Evidence,
    Contradiction,
    ClinicalRecommendation
)


class TestParsedDocument:
    """Test ParsedDocument model validation and functionality."""
    
    def test_valid_document_creation(self):
        """Test creating a valid ParsedDocument."""
        doc = ParsedDocument(
            id="doc_001",
            title="Efficacy of Drug X in Treatment Y",
            authors=["Dr. Smith", "Dr. Jones"],
            publication_date=datetime(2023, 1, 15),
            source="Journal of Medicine",
            document_type=DocumentType.RESEARCH_PAPER,
            content="This study examines the efficacy of Drug X...",
            metadata={"doi": "10.1234/example", "journal": "Journal of Medicine"},
            credibility_score=0.85
        )
        
        assert doc.id == "doc_001"
        assert doc.title == "Efficacy of Drug X in Treatment Y"
        assert len(doc.authors) == 2
        assert doc.document_type == DocumentType.RESEARCH_PAPER
        assert doc.credibility_score == 0.85
    
    def test_invalid_credibility_score(self):
        """Test that invalid credibility scores are rejected."""
        with pytest.raises(ValidationError):
            ParsedDocument(
                id="doc_001",
                title="Test Document",
                authors=["Dr. Smith"],
                publication_date=datetime.now(),
                source="Test Journal",
                document_type=DocumentType.RESEARCH_PAPER,
                content="Test content",
                credibility_score=1.5  # Invalid: > 1.0
            )
    
    def test_empty_title_validation(self):
        """Test that empty titles are rejected."""
        with pytest.raises(ValidationError):
            ParsedDocument(
                id="doc_001",
                title="",  # Invalid: empty title
                authors=["Dr. Smith"],
                publication_date=datetime.now(),
                source="Test Journal",
                document_type=DocumentType.RESEARCH_PAPER,
                content="Test content",
                credibility_score=0.8
            )


class TestClinicalQuery:
    """Test ClinicalQuery model validation and functionality."""
    
    def test_valid_query_creation(self):
        """Test creating a valid ClinicalQuery."""
        patient_context = PatientContext(
            age_range="45-65",
            gender="female",
            conditions=["diabetes", "hypertension"]
        )
        
        query = ClinicalQuery(
            id="query_001",
            query_text="What is the best treatment for diabetes in middle-aged women?",
            clinician_id="clinician_123",
            patient_context=patient_context,
            urgency_level=UrgencyLevel.ROUTINE
        )
        
        assert query.id == "query_001"
        assert query.urgency_level == UrgencyLevel.ROUTINE
        assert query.patient_context.age_range == "45-65"
        assert len(query.patient_context.conditions) == 2
    
    def test_empty_query_text_validation(self):
        """Test that empty query text is rejected."""
        with pytest.raises(ValidationError):
            ClinicalQuery(
                id="query_001",
                query_text="   ",  # Invalid: whitespace only
                clinician_id="clinician_123"
            )
    
    def test_query_text_trimming(self):
        """Test that query text is properly trimmed."""
        query = ClinicalQuery(
            id="query_001",
            query_text="  What is the treatment?  ",
            clinician_id="clinician_123"
        )
        
        assert query.query_text == "What is the treatment?"


class TestEvidence:
    """Test Evidence model validation and functionality."""
    
    def test_valid_evidence_creation(self):
        """Test creating valid Evidence."""
        evidence = Evidence(
            document_id="doc_001",
            relevance_score=0.92,
            evidence_level=EvidenceLevel.RCT,
            excerpt="The study showed significant improvement in patient outcomes...",
            confidence_interval="95% CI: 0.8-1.2",
            sample_size=500
        )
        
        assert evidence.document_id == "doc_001"
        assert evidence.relevance_score == 0.92
        assert evidence.evidence_level == EvidenceLevel.RCT
        assert evidence.sample_size == 500
    
    def test_invalid_relevance_score(self):
        """Test that invalid relevance scores are rejected."""
        with pytest.raises(ValidationError):
            Evidence(
                document_id="doc_001",
                relevance_score=1.5,  # Invalid: > 1.0
                evidence_level=EvidenceLevel.RCT,
                excerpt="Test excerpt"
            )
    
    def test_invalid_sample_size(self):
        """Test that invalid sample sizes are rejected."""
        with pytest.raises(ValidationError):
            Evidence(
                document_id="doc_001",
                relevance_score=0.8,
                evidence_level=EvidenceLevel.RCT,
                excerpt="Test excerpt",
                sample_size=0  # Invalid: must be >= 1
            )


class TestClinicalRecommendation:
    """Test ClinicalRecommendation model validation and functionality."""
    
    def test_valid_recommendation_creation(self):
        """Test creating a valid ClinicalRecommendation."""
        evidence = Evidence(
            document_id="doc_001",
            relevance_score=0.9,
            evidence_level=EvidenceLevel.SYSTEMATIC_REVIEW,
            excerpt="Meta-analysis shows strong evidence for treatment effectiveness"
        )
        
        recommendation = ClinicalRecommendation(
            id="rec_001",
            query_id="query_001",
            recommendation_text="Recommend Drug X as first-line treatment",
            supporting_evidence=[evidence],
            confidence_score=0.88
        )
        
        assert recommendation.id == "rec_001"
        assert recommendation.query_id == "query_001"
        assert recommendation.confidence_score == 0.88
        assert len(recommendation.supporting_evidence) == 1
    
    def test_empty_supporting_evidence_validation(self):
        """Test that recommendations without supporting evidence are rejected."""
        with pytest.raises(ValidationError):
            ClinicalRecommendation(
                id="rec_001",
                query_id="query_001",
                recommendation_text="Recommend Drug X",
                supporting_evidence=[],  # Invalid: empty evidence list
                confidence_score=0.8
            )
    
    def test_empty_recommendation_text_validation(self):
        """Test that empty recommendation text is rejected."""
        evidence = Evidence(
            document_id="doc_001",
            relevance_score=0.9,
            evidence_level=EvidenceLevel.RCT,
            excerpt="Test evidence"
        )
        
        with pytest.raises(ValidationError):
            ClinicalRecommendation(
                id="rec_001",
                query_id="query_001",
                recommendation_text="",  # Invalid: empty text
                supporting_evidence=[evidence],
                confidence_score=0.8
            )


class TestEnums:
    """Test enum definitions and values."""
    
    def test_document_type_enum(self):
        """Test DocumentType enum values."""
        assert DocumentType.RESEARCH_PAPER == "research_paper"
        assert DocumentType.CLINICAL_TRIAL == "clinical_trial"
        assert DocumentType.SYSTEMATIC_REVIEW == "systematic_review"
    
    def test_evidence_level_enum(self):
        """Test EvidenceLevel enum values."""
        assert EvidenceLevel.SYSTEMATIC_REVIEW == "systematic_review"
        assert EvidenceLevel.RCT == "randomized_controlled_trial"
        assert EvidenceLevel.OBSERVATIONAL == "observational"
    
    def test_urgency_level_enum(self):
        """Test UrgencyLevel enum values."""
        assert UrgencyLevel.EMERGENCY == "emergency"
        assert UrgencyLevel.URGENT == "urgent"
        assert UrgencyLevel.ROUTINE == "routine"


class TestSerialization:
    """Test JSON serialization and deserialization."""
    
    def test_document_json_serialization(self):
        """Test that ParsedDocument can be serialized to JSON."""
        doc = ParsedDocument(
            id="doc_001",
            title="Test Document",
            authors=["Dr. Smith"],
            publication_date=datetime(2023, 1, 15),
            source="Test Journal",
            document_type=DocumentType.RESEARCH_PAPER,
            content="Test content",
            credibility_score=0.8
        )
        
        json_data = doc.model_dump_json()
        assert "doc_001" in json_data
        assert "Test Document" in json_data
        assert "2023-01-15T00:00:00" in json_data
    
    def test_query_json_serialization(self):
        """Test that ClinicalQuery can be serialized to JSON."""
        query = ClinicalQuery(
            id="query_001",
            query_text="Test query",
            clinician_id="clinician_123"
        )
        
        json_data = query.model_dump_json()
        assert "query_001" in json_data
        assert "Test query" in json_data
        assert "clinician_123" in json_data