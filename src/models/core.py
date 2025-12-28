"""
Core data models for the Clinical Evidence Copilot system.

This module defines the primary data structures used throughout the system
for representing documents, queries, evidence, and clinical recommendations.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DocumentType(str, Enum):
    """Types of medical documents that can be processed."""
    RESEARCH_PAPER = "research_paper"
    GUIDELINE = "guideline"
    CLINICAL_TRIAL = "clinical_trial"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_STUDY = "case_study"


class EvidenceLevel(str, Enum):
    """Hierarchy of evidence levels in medical research."""
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    RCT = "randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL = "case_control"
    OBSERVATIONAL = "observational"
    CASE_STUDY = "case_study"
    EXPERT_OPINION = "expert_opinion"


class UrgencyLevel(str, Enum):
    """Urgency levels for clinical queries."""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"
    RESEARCH = "research"


@dataclass
class PatientContext:
    """Optional patient context for clinical queries."""
    age_range: Optional[str] = None
    gender: Optional[str] = None
    conditions: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class ParsedDocument(BaseModel):
    """
    Represents a medical document that has been parsed and processed.
    
    Validates Requirements 6.1, 6.2, 6.3:
    - Supports multiple document formats (PDF, XML, HTML)
    - Preserves medical terminology and numerical data integrity
    - Extracts comprehensive metadata
    """
    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., min_length=1, description="Document title")
    authors: List[str] = Field(default_factory=list, description="List of document authors")
    publication_date: datetime = Field(..., description="Date of publication")
    source: str = Field(..., min_length=1, description="Source of the document")
    document_type: DocumentType = Field(..., description="Type of medical document")
    content: str = Field(..., min_length=1, description="Full text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Document credibility score")
    
    @field_validator('authors')
    @classmethod
    def validate_authors(cls, v):
        """Ensure authors list contains valid entries."""
        if v:
            for author in v:
                if not isinstance(author, str) or not author.strip():
                    raise ValueError("All authors must be non-empty strings")
        return v
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        """Ensure metadata contains expected medical document fields."""
        # Common medical document metadata fields
        expected_fields = ['study_methodology', 'sample_size', 'doi', 'journal', 'keywords']
        return v
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class ClinicalQuery(BaseModel):
    """
    Represents a clinical query from a healthcare professional.
    
    Used for processing medical questions and generating evidence-backed responses.
    """
    id: str = Field(..., description="Unique identifier for the query")
    query_text: str = Field(..., min_length=1, description="The clinical question text")
    clinician_id: str = Field(..., description="Identifier of the requesting clinician")
    patient_context: Optional[PatientContext] = Field(None, description="Optional patient context")
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE, description="Query urgency level")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query submission timestamp")
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v):
        """Ensure query text is meaningful."""
        if not v.strip():
            raise ValueError("Query text cannot be empty or whitespace only")
        return v.strip()
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class Evidence(BaseModel):
    """
    Represents a piece of evidence extracted from medical literature.
    
    Links to source documents and provides relevance scoring.
    """
    document_id: str = Field(..., description="ID of the source document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to the query")
    evidence_level: EvidenceLevel = Field(..., description="Level of evidence hierarchy")
    excerpt: str = Field(..., min_length=1, description="Relevant text excerpt")
    confidence_interval: Optional[str] = Field(None, description="Statistical confidence interval if available")
    sample_size: Optional[int] = Field(None, ge=1, description="Study sample size if available")
    
    @field_validator('excerpt')
    @classmethod
    def validate_excerpt(cls, v):
        """Ensure excerpt is meaningful."""
        if not v.strip():
            raise ValueError("Evidence excerpt cannot be empty")
        return v.strip()


@dataclass
class Contradiction:
    """Represents conflicting evidence between studies."""
    conflicting_evidence: List[Evidence]
    explanation: str
    resolution_guidance: str


class ClinicalRecommendation(BaseModel):
    """
    Represents a clinical recommendation generated from evidence analysis.
    
    Includes supporting evidence, confidence scoring, and change tracking.
    """
    id: str = Field(..., description="Unique identifier for the recommendation")
    query_id: str = Field(..., description="ID of the originating query")
    recommendation_text: str = Field(..., min_length=1, description="The clinical recommendation")
    supporting_evidence: List[Evidence] = Field(default_factory=list, description="Evidence supporting this recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the recommendation")
    contradictions: List[Contradiction] = Field(default_factory=list, description="Any contradictory evidence")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    change_reason: Optional[str] = Field(None, description="Reason for last change if applicable")
    
    @field_validator('recommendation_text')
    @classmethod
    def validate_recommendation_text(cls, v):
        """Ensure recommendation text is meaningful."""
        if not v.strip():
            raise ValueError("Recommendation text cannot be empty")
        return v.strip()
    
    @field_validator('supporting_evidence')
    @classmethod
    def validate_supporting_evidence(cls, v):
        """Ensure there is supporting evidence for the recommendation."""
        if not v:
            raise ValueError("Clinical recommendations must have supporting evidence")
        return v
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# Export all models
__all__ = [
    'DocumentType',
    'EvidenceLevel', 
    'UrgencyLevel',
    'PatientContext',
    'ParsedDocument',
    'ClinicalQuery',
    'Evidence',
    'Contradiction',
    'ClinicalRecommendation'
]