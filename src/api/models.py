"""
Pydantic models for API request/response schemas.

Defines the data structures for FastAPI endpoints to ensure
proper validation and documentation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..models.core import (
    ClinicalRecommendation, PatientContext, UrgencyLevel, 
    DocumentType, EvidenceLevel
)


class QueryRequest(BaseModel):
    """Request model for clinical queries."""
    query_text: str = Field(..., min_length=1, max_length=2000, description="The clinical question")
    clinician_id: str = Field(..., description="Identifier of the requesting clinician")
    patient_context: Optional[PatientContext] = Field(None, description="Optional patient context")
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE, description="Query urgency level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "What is the recommended first-line treatment for hypertension in elderly patients?",
                "clinician_id": "dr_smith_123",
                "patient_context": {
                    "age_range": "65-75",
                    "gender": "female",
                    "conditions": ["diabetes", "mild_kidney_disease"],
                    "medications": ["metformin"],
                    "allergies": ["penicillin"]
                },
                "urgency_level": "routine"
            }
        }


class CitationInfo(BaseModel):
    """Citation information for evidence sources."""
    citation_number: int = Field(..., description="Citation number in the response")
    title: str = Field(..., description="Document title")
    authors: List[str] = Field(default_factory=list, description="Document authors")
    journal: str = Field(..., description="Journal or source")
    publication_date: str = Field(..., description="Publication date")
    doi: Optional[str] = Field(None, description="DOI if available")
    evidence_level: str = Field(..., description="Level of evidence")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to query")


class ReasoningStep(BaseModel):
    """Reasoning step in the recommendation process."""
    step_number: int = Field(..., description="Step number in the process")
    step_type: str = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Description of the step")
    reasoning: str = Field(..., description="Detailed reasoning explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this step")


class QueryResponse(BaseModel):
    """Response model for clinical queries."""
    query_id: str = Field(..., description="Unique identifier for the query")
    recommendation: ClinicalRecommendation = Field(..., description="Clinical recommendation")
    processing_time_seconds: float = Field(..., description="Time taken to process the query")
    citations: List[CitationInfo] = Field(default_factory=list, description="Citations for evidence sources")
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list, description="Step-by-step reasoning")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "query_20241228_143022_dr_smith_123",
                "recommendation": {
                    "id": "rec_query_20241228_143022_dr_smith_123_20241228_143025",
                    "query_id": "query_20241228_143022_dr_smith_123",
                    "recommendation_text": "Based on current evidence, ACE inhibitors are recommended as first-line treatment...",
                    "confidence_score": 0.85,
                    "last_updated": "2024-12-28T14:30:25"
                },
                "processing_time_seconds": 3.2,
                "citations": [
                    {
                        "citation_number": 1,
                        "title": "Hypertension Management Guidelines 2024",
                        "authors": ["Smith, J.", "Johnson, M."],
                        "journal": "Journal of Hypertension",
                        "publication_date": "2024",
                        "evidence_level": "Systematic Review",
                        "relevance_score": 0.92
                    }
                ]
            }
        }


class DocumentUploadRequest(BaseModel):
    """Request model for document uploads."""
    title: str = Field(..., min_length=1, description="Document title")
    authors: List[str] = Field(default_factory=list, description="Document authors")
    content: str = Field(..., min_length=1, description="Document content")
    document_type: DocumentType = Field(..., description="Type of medical document")
    source: str = Field(..., description="Source of the document")
    publication_date: datetime = Field(..., description="Publication date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Efficacy of ACE Inhibitors in Elderly Hypertensive Patients",
                "authors": ["Dr. Jane Smith", "Dr. Robert Johnson"],
                "content": "Abstract: This systematic review examines...",
                "document_type": "systematic_review",
                "source": "Journal of Hypertension",
                "publication_date": "2024-01-15T00:00:00",
                "metadata": {
                    "doi": "10.1234/jh.2024.001",
                    "sample_size": 5000,
                    "study_methodology": "systematic_review"
                }
            }
        }


class DocumentResponse(BaseModel):
    """Response model for document information."""
    id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    authors: List[str] = Field(default_factory=list, description="Document authors")
    publication_date: datetime = Field(..., description="Publication date")
    document_type: DocumentType = Field(..., description="Type of medical document")
    source: str = Field(..., description="Source of the document")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Document credibility score")
    indexed_at: datetime = Field(..., description="When the document was indexed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_20241228_143000_001",
                "title": "Efficacy of ACE Inhibitors in Elderly Hypertensive Patients",
                "authors": ["Dr. Jane Smith", "Dr. Robert Johnson"],
                "publication_date": "2024-01-15T00:00:00",
                "document_type": "systematic_review",
                "source": "Journal of Hypertension",
                "credibility_score": 0.92,
                "indexed_at": "2024-12-28T14:30:00"
            }
        }


class RecommendationHistoryResponse(BaseModel):
    """Response model for recommendation history."""
    query_id: str = Field(..., description="Query identifier")
    recommendations: List[ClinicalRecommendation] = Field(..., description="Historical recommendations")
    total_changes: int = Field(..., description="Total number of changes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "query_20241228_143022_dr_smith_123",
                "recommendations": [
                    {
                        "id": "rec_query_20241228_143022_dr_smith_123_20241228_143025",
                        "recommendation_text": "Initial recommendation based on available evidence...",
                        "confidence_score": 0.75,
                        "last_updated": "2024-12-28T14:30:25",
                        "change_reason": None
                    },
                    {
                        "id": "rec_query_20241228_143022_dr_smith_123_20241228_150000",
                        "recommendation_text": "Updated recommendation with new evidence...",
                        "confidence_score": 0.85,
                        "last_updated": "2024-12-28T15:00:00",
                        "change_reason": "New systematic review added to knowledge base"
                    }
                ],
                "total_changes": 1
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-12-28T14:30:00",
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Query text cannot be empty",
                "timestamp": "2024-12-28T14:30:00"
            }
        }


class NotificationRequest(BaseModel):
    """Request model for notification subscriptions."""
    clinician_id: str = Field(..., description="Clinician identifier")
    query_keywords: List[str] = Field(default_factory=list, description="Keywords to monitor")
    notification_types: List[str] = Field(
        default=["recommendation_change", "new_evidence"],
        description="Types of notifications to receive"
    )


class NotificationResponse(BaseModel):
    """Response model for notifications."""
    notification_id: str = Field(..., description="Notification identifier")
    clinician_id: str = Field(..., description="Target clinician")
    notification_type: str = Field(..., description="Type of notification")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    related_query_id: Optional[str] = Field(None, description="Related query if applicable")
    timestamp: datetime = Field(..., description="Notification timestamp")
    read: bool = Field(default=False, description="Whether notification has been read")


# Export all models
__all__ = [
    'QueryRequest',
    'QueryResponse', 
    'DocumentUploadRequest',
    'DocumentResponse',
    'RecommendationHistoryResponse',
    'HealthCheckResponse',
    'ErrorResponse',
    'CitationInfo',
    'ReasoningStep',
    'NotificationRequest',
    'NotificationResponse'
]