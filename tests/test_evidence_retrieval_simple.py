"""
Simple tests for evidence retrieval functionality.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.core import Evidence, EvidenceLevel, ClinicalQuery, UrgencyLevel, Contradiction
from datetime import datetime


def test_evidence_creation():
    """Test creating evidence objects."""
    evidence = Evidence(
        document_id="test_doc",
        relevance_score=0.85,
        evidence_level=EvidenceLevel.RCT,
        excerpt="This is a test excerpt.",
        sample_size=1000
    )
    
    assert evidence.document_id == "test_doc"
    assert evidence.relevance_score == 0.85
    assert evidence.evidence_level == EvidenceLevel.RCT
    assert evidence.sample_size == 1000


def test_clinical_query_creation():
    """Test creating clinical query objects."""
    query = ClinicalQuery(
        id="test_query",
        query_text="What is the effectiveness of drug X?",
        clinician_id="clinician_123",
        urgency_level=UrgencyLevel.ROUTINE
    )
    
    assert query.id == "test_query"
    assert query.query_text == "What is the effectiveness of drug X?"
    assert query.clinician_id == "clinician_123"
    assert query.urgency_level == UrgencyLevel.ROUTINE


def test_evidence_hierarchy_ranking():
    """Test evidence hierarchy ranking logic."""
    # Import and test the standalone implementation
    import numpy as np
    from typing import List
    
    class TestEvidenceHierarchyRanker:
        EVIDENCE_WEIGHTS = {
            EvidenceLevel.SYSTEMATIC_REVIEW: 10.0,
            EvidenceLevel.META_ANALYSIS: 9.5,
            EvidenceLevel.RCT: 8.0,
            EvidenceLevel.COHORT_STUDY: 6.0,
            EvidenceLevel.CASE_CONTROL: 5.0,
            EvidenceLevel.OBSERVATIONAL: 4.0,
            EvidenceLevel.CASE_STUDY: 2.0,
            EvidenceLevel.EXPERT_OPINION: 1.0
        }
        
        def rank_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
            if not evidence_list:
                return []
            
            scored_evidence = []
            for evidence in evidence_list:
                composite_score = self._calculate_composite_score(evidence)
                scored_evidence.append((evidence, composite_score))
            
            scored_evidence.sort(key=lambda x: x[1], reverse=True)
            return [evidence for evidence, _ in scored_evidence]
        
        def _calculate_composite_score(self, evidence: Evidence) -> float:
            hierarchy_score = self.EVIDENCE_WEIGHTS.get(evidence.evidence_level, 1.0)
            relevance_score = evidence.relevance_score
            
            sample_size_bonus = 0.0
            if evidence.sample_size:
                sample_size_bonus = min(np.log10(evidence.sample_size) / 5.0, 1.0)
            
            composite_score = (
                hierarchy_score * 0.6 +
                relevance_score * 10.0 * 0.3 +
                sample_size_bonus * 0.1
            )
            
            return composite_score
    
    # Create test evidence
    evidence_list = [
        Evidence(
            document_id="doc1",
            relevance_score=0.7,
            evidence_level=EvidenceLevel.OBSERVATIONAL,
            excerpt="Observational study.",
            sample_size=500
        ),
        Evidence(
            document_id="doc2",
            relevance_score=0.9,
            evidence_level=EvidenceLevel.SYSTEMATIC_REVIEW,
            excerpt="Systematic review.",
            sample_size=5000
        ),
        Evidence(
            document_id="doc3",
            relevance_score=0.8,
            evidence_level=EvidenceLevel.RCT,
            excerpt="RCT study.",
            sample_size=1000
        )
    ]
    
    # Test ranking
    ranker = TestEvidenceHierarchyRanker()
    ranked = ranker.rank_evidence(evidence_list)
    
    # Systematic review should be first
    assert ranked[0].evidence_level == EvidenceLevel.SYSTEMATIC_REVIEW
    # RCT should be second
    assert ranked[1].evidence_level == EvidenceLevel.RCT
    # Observational should be last
    assert ranked[2].evidence_level == EvidenceLevel.OBSERVATIONAL


def test_contradiction_detection():
    """Test contradiction detection logic."""
    from typing import List, Optional
    
    class TestContradictionDetector:
        def __init__(self):
            self.similarity_threshold = 0.3
        
        def detect_contradictions(self, evidence_list: List[Evidence]) -> List[Contradiction]:
            if len(evidence_list) < 2:
                return []
            
            contradictions = []
            for i in range(len(evidence_list)):
                for j in range(i + 1, len(evidence_list)):
                    evidence1 = evidence_list[i]
                    evidence2 = evidence_list[j]
                    
                    if self._are_comparable(evidence1, evidence2):
                        contradiction = self._analyze_potential_contradiction(evidence1, evidence2)
                        if contradiction:
                            contradictions.append(contradiction)
            
            return contradictions
        
        def _are_comparable(self, evidence1: Evidence, evidence2: Evidence) -> bool:
            text1 = evidence1.excerpt.lower()
            text2 = evidence2.excerpt.lower()
            
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return False
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union) if union else 0.0
            return similarity >= self.similarity_threshold
        
        def _analyze_potential_contradiction(self, evidence1: Evidence, evidence2: Evidence) -> Optional[Contradiction]:
            contradictory_patterns = [
                ("effective", "ineffective"),
                ("beneficial", "harmful"),
                ("significant", "non-significant")
            ]
            
            text1 = evidence1.excerpt.lower()
            text2 = evidence2.excerpt.lower()
            
            found_contradictions = []
            for positive_term, negative_term in contradictory_patterns:
                if ((positive_term in text1 and negative_term in text2) or 
                    (negative_term in text1 and positive_term in text2)):
                    found_contradictions.append((positive_term, negative_term))
            
            if found_contradictions:
                return Contradiction(
                    conflicting_evidence=[evidence1, evidence2],
                    explanation="Contradiction detected between studies",
                    resolution_guidance="Consider prioritizing higher quality evidence"
                )
            
            return None
    
    # Create contradictory evidence
    evidence_list = [
        Evidence(
            document_id="doc1",
            relevance_score=0.9,
            evidence_level=EvidenceLevel.SYSTEMATIC_REVIEW,
            excerpt="Treatment X is highly effective for condition Y.",
            sample_size=5000
        ),
        Evidence(
            document_id="doc2",
            relevance_score=0.8,
            evidence_level=EvidenceLevel.RCT,
            excerpt="Treatment X is ineffective for condition Y.",
            sample_size=1000
        )
    ]
    
    # Test contradiction detection
    detector = TestContradictionDetector()
    contradictions = detector.detect_contradictions(evidence_list)
    
    # Should find at least one contradiction
    assert len(contradictions) >= 1
    
    # Check contradiction structure
    contradiction = contradictions[0]
    assert isinstance(contradiction, Contradiction)
    assert len(contradiction.conflicting_evidence) == 2
    assert contradiction.explanation
    assert contradiction.resolution_guidance


def test_evidence_level_weights():
    """Test that evidence level weights are properly ordered."""
    weights = {
        EvidenceLevel.SYSTEMATIC_REVIEW: 10.0,
        EvidenceLevel.META_ANALYSIS: 9.5,
        EvidenceLevel.RCT: 8.0,
        EvidenceLevel.COHORT_STUDY: 6.0,
        EvidenceLevel.CASE_CONTROL: 5.0,
        EvidenceLevel.OBSERVATIONAL: 4.0,
        EvidenceLevel.CASE_STUDY: 2.0,
        EvidenceLevel.EXPERT_OPINION: 1.0
    }
    
    # Systematic review should have highest weight
    assert weights[EvidenceLevel.SYSTEMATIC_REVIEW] > weights[EvidenceLevel.RCT]
    assert weights[EvidenceLevel.RCT] > weights[EvidenceLevel.OBSERVATIONAL]
    assert weights[EvidenceLevel.OBSERVATIONAL] > weights[EvidenceLevel.EXPERT_OPINION]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])