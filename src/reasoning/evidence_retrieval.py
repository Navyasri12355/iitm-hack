"""
Evidence retrieval and ranking system for the Clinical Evidence Copilot.

This module implements:
- Vector similarity search with medical context
- Evidence hierarchy ranking (systematic reviews > RCTs > observational)
- Contradiction detection between studies

Validates Requirements 1.3, 1.4, 3.5:
- Rank recommendations based on current evidence strength
- Flag contradictions and explain differences
- Use appropriate medical tools and databases for comprehensive analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from ..models.core import (
    Evidence, EvidenceLevel, ClinicalQuery, ParsedDocument, 
    Contradiction, DocumentType
)

# Import PathwayVectorStore conditionally to avoid import errors
try:
    from ..ingestion.vector_store import PathwayVectorStore
except ImportError:
    # Create a mock class if PathwayVectorStore is not available
    class PathwayVectorStore:
        def __init__(self, *args, **kwargs):
            pass
        
        def similarity_search(self, query_text: str, top_k: int = 10, filters=None):
            return []

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """Context information for medical evidence search."""
    medical_specialty: Optional[str] = None
    patient_demographics: Optional[Dict[str, Any]] = None
    urgency_level: str = "routine"
    evidence_recency_preference: int = 5  # years
    minimum_evidence_level: Optional[EvidenceLevel] = None


class EvidenceHierarchyRanker:
    """
    Ranks medical evidence according to established evidence hierarchy.
    
    Implements the standard medical evidence pyramid:
    1. Systematic Reviews & Meta-analyses (highest)
    2. Randomized Controlled Trials (RCTs)
    3. Cohort Studies
    4. Case-Control Studies
    5. Observational Studies
    6. Case Studies
    7. Expert Opinion (lowest)
    """
    
    # Evidence level weights (higher = better evidence)
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
    
    def __init__(self):
        """Initialize the evidence hierarchy ranker."""
        logger.info("Initialized EvidenceHierarchyRanker")
    
    def rank_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """
        Rank evidence according to medical evidence hierarchy.
        
        Args:
            evidence_list: List of Evidence objects to rank
            
        Returns:
            List of Evidence objects sorted by hierarchy and relevance
        """
        if not evidence_list:
            return []
        
        # Calculate composite scores for each evidence
        scored_evidence = []
        for evidence in evidence_list:
            composite_score = self._calculate_composite_score(evidence)
            scored_evidence.append((evidence, composite_score))
        
        # Sort by composite score (descending)
        scored_evidence.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked evidence
        ranked_evidence = [evidence for evidence, _ in scored_evidence]
        
        logger.info(f"Ranked {len(evidence_list)} evidence items by hierarchy")
        return ranked_evidence
    
    def _calculate_composite_score(self, evidence: Evidence) -> float:
        """
        Calculate composite score combining evidence level, relevance, and other factors.
        
        Args:
            evidence: Evidence object to score
            
        Returns:
            Composite score for ranking
        """
        # Base score from evidence hierarchy
        hierarchy_score = self.EVIDENCE_WEIGHTS.get(evidence.evidence_level, 1.0)
        
        # Relevance score (0.0 to 1.0)
        relevance_score = evidence.relevance_score
        
        # Sample size bonus (if available)
        sample_size_bonus = 0.0
        if evidence.sample_size:
            # Logarithmic bonus for larger sample sizes
            sample_size_bonus = min(np.log10(evidence.sample_size) / 5.0, 1.0)
        
        # Combine scores with weights
        composite_score = (
            hierarchy_score * 0.6 +          # Evidence level is most important
            relevance_score * 10.0 * 0.3 +   # Relevance is second most important
            sample_size_bonus * 0.1           # Sample size provides small bonus
        )
        
        return composite_score


class ContradictionDetector:
    """
    Detects and analyzes contradictions between medical studies.
    
    Identifies conflicting findings, analyzes potential reasons for contradictions,
    and provides guidance for resolution.
    """
    
    def __init__(self):
        """Initialize the contradiction detector."""
        self.similarity_threshold = 0.7  # Threshold for considering studies similar enough to compare
        logger.info("Initialized ContradictionDetector")
    
    def detect_contradictions(self, evidence_list: List[Evidence]) -> List[Contradiction]:
        """
        Detect contradictions between evidence items.
        
        Args:
            evidence_list: List of Evidence objects to analyze
            
        Returns:
            List of Contradiction objects found
        """
        if len(evidence_list) < 2:
            return []
        
        contradictions = []
        
        # Compare each pair of evidence items
        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                evidence1 = evidence_list[i]
                evidence2 = evidence_list[j]
                
                # Check if these evidence items are comparable
                if self._are_comparable(evidence1, evidence2):
                    contradiction = self._analyze_potential_contradiction(evidence1, evidence2)
                    if contradiction:
                        contradictions.append(contradiction)
        
        logger.info(f"Detected {len(contradictions)} contradictions among {len(evidence_list)} evidence items")
        return contradictions
    
    def _are_comparable(self, evidence1: Evidence, evidence2: Evidence) -> bool:
        """
        Determine if two evidence items are comparable for contradiction analysis.
        
        Args:
            evidence1: First evidence item
            evidence2: Second evidence item
            
        Returns:
            True if evidence items are comparable
        """
        # For now, use a simple text similarity approach
        # In a full implementation, this would use more sophisticated medical concept matching
        
        text1 = evidence1.excerpt.lower()
        text2 = evidence2.excerpt.lower()
        
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        return similarity >= self.similarity_threshold
    
    def _analyze_potential_contradiction(self, evidence1: Evidence, evidence2: Evidence) -> Optional[Contradiction]:
        """
        Analyze two evidence items for potential contradictions.
        
        Args:
            evidence1: First evidence item
            evidence2: Second evidence item
            
        Returns:
            Contradiction object if contradiction found, None otherwise
        """
        # Look for contradictory keywords and phrases
        contradictory_patterns = [
            ("effective", "ineffective"),
            ("beneficial", "harmful"),
            ("recommended", "not recommended"),
            ("safe", "unsafe"),
            ("increases", "decreases"),
            ("improves", "worsens"),
            ("positive", "negative"),
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
            # Generate explanation
            explanation = self._generate_contradiction_explanation(
                evidence1, evidence2, found_contradictions
            )
            
            # Generate resolution guidance
            resolution_guidance = self._generate_resolution_guidance(evidence1, evidence2)
            
            return Contradiction(
                conflicting_evidence=[evidence1, evidence2],
                explanation=explanation,
                resolution_guidance=resolution_guidance
            )
        
        return None
    
    def _generate_contradiction_explanation(
        self, 
        evidence1: Evidence, 
        evidence2: Evidence, 
        contradictions: List[Tuple[str, str]]
    ) -> str:
        """Generate explanation for detected contradiction."""
        contradiction_terms = [f"'{pos}' vs '{neg}'" for pos, neg in contradictions]
        terms_str = ", ".join(contradiction_terms)
        
        explanation = (
            f"Contradiction detected between studies with conflicting findings: {terms_str}. "
            f"Study 1 (Evidence Level: {evidence1.evidence_level.value}) "
            f"conflicts with Study 2 (Evidence Level: {evidence2.evidence_level.value}). "
        )
        
        # Add sample size information if available
        if evidence1.sample_size and evidence2.sample_size:
            explanation += (
                f"Sample sizes differ significantly: {evidence1.sample_size} vs {evidence2.sample_size}. "
            )
        
        return explanation
    
    def _generate_resolution_guidance(self, evidence1: Evidence, evidence2: Evidence) -> str:
        """Generate guidance for resolving contradictions."""
        # Determine which evidence has higher quality
        ranker = EvidenceHierarchyRanker()
        score1 = ranker._calculate_composite_score(evidence1)
        score2 = ranker._calculate_composite_score(evidence2)
        
        if score1 > score2:
            higher_quality = evidence1
            lower_quality = evidence2
        else:
            higher_quality = evidence2
            lower_quality = evidence1
        
        guidance = (
            f"Consider prioritizing the {higher_quality.evidence_level.value} evidence "
            f"(relevance: {higher_quality.relevance_score:.2f}) over the "
            f"{lower_quality.evidence_level.value} evidence "
            f"(relevance: {lower_quality.relevance_score:.2f}). "
            f"However, examine study methodologies, populations, and contexts for differences "
            f"that might explain the conflicting results. Consider seeking additional "
            f"high-quality evidence to resolve the contradiction."
        )
        
        return guidance


class MedicalContextualRetriever:
    """
    Retrieves evidence with medical context awareness.
    
    Combines vector similarity search with medical domain knowledge
    to provide contextually relevant evidence retrieval.
    """
    
    def __init__(self, vector_store: PathwayVectorStore):
        """
        Initialize the medical contextual retriever.
        
        Args:
            vector_store: PathwayVectorStore instance for similarity search
        """
        self.vector_store = vector_store
        self.hierarchy_ranker = EvidenceHierarchyRanker()
        self.contradiction_detector = ContradictionDetector()
        
        logger.info("Initialized MedicalContextualRetriever")
    
    def retrieve_evidence(
        self, 
        query: ClinicalQuery, 
        search_context: Optional[SearchContext] = None,
        max_results: int = 20
    ) -> Tuple[List[Evidence], List[Contradiction]]:
        """
        Retrieve and rank evidence for a clinical query.
        
        Args:
            query: ClinicalQuery to search for
            search_context: Optional search context for filtering
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (ranked_evidence_list, contradictions_list)
        """
        try:
            # Prepare search filters
            filters = self._prepare_search_filters(search_context)
            
            # Perform vector similarity search
            search_results = self.vector_store.similarity_search(
                query_text=query.query_text,
                top_k=max_results * 2,  # Get more results for better filtering
                filters=filters
            )
            
            # Convert search results to Evidence objects
            evidence_list = self._convert_to_evidence(search_results, query)
            
            # Apply medical context filtering
            filtered_evidence = self._apply_medical_context_filtering(
                evidence_list, search_context
            )
            
            # Rank evidence by hierarchy and relevance
            ranked_evidence = self.hierarchy_ranker.rank_evidence(filtered_evidence)
            
            # Limit to requested number of results
            final_evidence = ranked_evidence[:max_results]
            
            # Detect contradictions
            contradictions = self.contradiction_detector.detect_contradictions(final_evidence)
            
            logger.info(
                f"Retrieved {len(final_evidence)} evidence items with "
                f"{len(contradictions)} contradictions for query: {query.query_text[:50]}..."
            )
            
            return final_evidence, contradictions
            
        except Exception as e:
            logger.error(f"Error retrieving evidence for query {query.id}: {e}")
            return [], []
    
    def _prepare_search_filters(self, search_context: Optional[SearchContext]) -> Dict[str, Any]:
        """Prepare search filters based on search context."""
        filters = {}
        
        if search_context:
            # Filter by evidence recency
            if search_context.evidence_recency_preference:
                max_age_days = search_context.evidence_recency_preference * 365
                filters['max_age_days'] = max_age_days
            
            # Filter by minimum evidence level
            if search_context.minimum_evidence_level:
                # This would need to be implemented in the vector store
                filters['min_evidence_level'] = search_context.minimum_evidence_level.value
        
        return filters
    
    def _convert_to_evidence(
        self, 
        search_results: List[Tuple[str, float, Dict[str, Any]]], 
        query: ClinicalQuery
    ) -> List[Evidence]:
        """Convert vector search results to Evidence objects."""
        evidence_list = []
        
        for document_id, similarity_score, metadata in search_results:
            try:
                # Determine evidence level from document type
                evidence_level = self._infer_evidence_level(metadata)
                
                # Create Evidence object
                evidence = Evidence(
                    document_id=document_id,
                    relevance_score=similarity_score,
                    evidence_level=evidence_level,
                    excerpt=f"Relevant content from {metadata.get('title', 'Unknown Document')}",
                    sample_size=metadata.get('sample_size'),
                    confidence_interval=metadata.get('confidence_interval')
                )
                
                evidence_list.append(evidence)
                
            except Exception as e:
                logger.warning(f"Error converting search result to evidence: {e}")
                continue
        
        return evidence_list
    
    def _infer_evidence_level(self, metadata: Dict[str, Any]) -> EvidenceLevel:
        """Infer evidence level from document metadata."""
        doc_type = metadata.get('document_type', '').lower()
        title = metadata.get('title', '').lower()
        
        # Map document types to evidence levels
        if 'systematic_review' in doc_type or 'systematic review' in title:
            return EvidenceLevel.SYSTEMATIC_REVIEW
        elif 'meta_analysis' in doc_type or 'meta-analysis' in title:
            return EvidenceLevel.META_ANALYSIS
        elif 'clinical_trial' in doc_type or 'randomized' in title or 'rct' in title:
            return EvidenceLevel.RCT
        elif 'cohort' in title:
            return EvidenceLevel.COHORT_STUDY
        elif 'case_control' in doc_type or 'case-control' in title:
            return EvidenceLevel.CASE_CONTROL
        elif 'case_study' in doc_type or 'case study' in title or 'case report' in title:
            return EvidenceLevel.CASE_STUDY
        elif 'guideline' in doc_type:
            return EvidenceLevel.SYSTEMATIC_REVIEW  # Guidelines are typically high-level evidence
        else:
            return EvidenceLevel.OBSERVATIONAL  # Default for research papers
    
    def _apply_medical_context_filtering(
        self, 
        evidence_list: List[Evidence], 
        search_context: Optional[SearchContext]
    ) -> List[Evidence]:
        """Apply medical context-specific filtering to evidence."""
        if not search_context:
            return evidence_list
        
        filtered_evidence = []
        
        for evidence in evidence_list:
            # Apply minimum evidence level filter
            if search_context.minimum_evidence_level:
                evidence_weight = self.hierarchy_ranker.EVIDENCE_WEIGHTS.get(
                    evidence.evidence_level, 0
                )
                min_weight = self.hierarchy_ranker.EVIDENCE_WEIGHTS.get(
                    search_context.minimum_evidence_level, 0
                )
                
                if evidence_weight < min_weight:
                    continue
            
            # Apply other context-specific filters here
            # (e.g., medical specialty, patient demographics)
            
            filtered_evidence.append(evidence)
        
        return filtered_evidence


# Export main classes
__all__ = [
    'EvidenceHierarchyRanker',
    'ContradictionDetector', 
    'MedicalContextualRetriever',
    'SearchContext'
]