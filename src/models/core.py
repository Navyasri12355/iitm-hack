"""Core data models for the Clinical Evidence Copilot."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DocumentType(str, Enum):
    RESEARCH_PAPER = "research_paper"
    GUIDELINE = "guideline"
    CLINICAL_TRIAL = "clinical_trial"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_STUDY = "case_study"


class EvidenceLevel(str, Enum):
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    RCT = "randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL = "case_control"
    OBSERVATIONAL = "observational"
    CASE_STUDY = "case_study"
    EXPERT_OPINION = "expert_opinion"


class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"
    RESEARCH = "research"


class ParsedDocument(BaseModel):
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    publication_date: datetime
    source: str
    document_type: DocumentType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class Evidence(BaseModel):
    document_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    evidence_level: EvidenceLevel
    excerpt: str
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    source: str = ""
    publication_date: Optional[str] = None
    confidence_interval: Optional[str] = None
    sample_size: Optional[int] = None
    contradiction_flag: bool = False


@dataclass
class Contradiction:
    conflicting_evidence: List[Evidence]
    explanation: str
    resolution_guidance: str


class ClinicalQuery(BaseModel):
    id: str
    query_text: str
    clinician_id: str
    patient_context: Optional[Dict[str, Any]] = None
    urgency_level: UrgencyLevel = UrgencyLevel.ROUTINE
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class ClinicalRecommendation(BaseModel):
    id: str
    query_id: str
    recommendation_text: str
    supporting_evidence: List[Evidence] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)
    change_reason: Optional[str] = None
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


__all__ = [
    "DocumentType", "EvidenceLevel", "UrgencyLevel",
    "ParsedDocument", "Evidence", "Contradiction",
    "ClinicalQuery", "ClinicalRecommendation"
]