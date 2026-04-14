"""
Document validation module for clinical evidence copilot.

This module implements document authenticity validation and credibility assessment
for medical literature ingestion.

Validates Requirements 2.4:
- Validate document authenticity and source credibility before indexing
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..models.core import ParsedDocument, DocumentType, EvidenceLevel

logger = logging.getLogger(__name__)


class DocumentValidator:
    """
    Validates document authenticity and credibility for medical literature.
    
    Performs comprehensive validation including source verification,
    content quality assessment, and credibility scoring.
    """
    
    def __init__(self):
        """Initialize the document validator."""
        # Trusted medical journal patterns
        self.trusted_journals = {
            'nature medicine', 'new england journal of medicine', 'nejm',
            'lancet', 'jama', 'bmj', 'cochrane', 'pubmed', 'medline',
            'journal of clinical medicine', 'clinical medicine',
            'american journal of medicine', 'european heart journal',
            'circulation', 'hypertension research', 'diabetes care'
        }
        
        # Suspicious content patterns that indicate low quality
        self.suspicious_patterns = [
            re.compile(r'\b(miracle cure|guaranteed|100% effective)\b', re.IGNORECASE),
            re.compile(r'\b(doctors hate|one weird trick|secret)\b', re.IGNORECASE),
            re.compile(r'\b(big pharma|conspiracy|cover.?up)\b', re.IGNORECASE),
        ]
        
        # Quality indicators for medical content
        self.quality_indicators = {
            'methodology': re.compile(r'\b(randomized|controlled|double.?blind|placebo|methodology)\b', re.IGNORECASE),
            'statistics': re.compile(r'\b(p.?value|confidence interval|statistical|significance)\b', re.IGNORECASE),
            'sample_size': re.compile(r'\bn\s*=\s*\d+\b', re.IGNORECASE),
            'citations': re.compile(r'\b(references|bibliography|doi|pmid)\b', re.IGNORECASE),
            'peer_review': re.compile(r'\b(peer.?review|reviewed|journal)\b', re.IGNORECASE)
        }
    
    def validate_document(self, document: ParsedDocument) -> Tuple[bool, Dict[str, any]]:
        """
        Perform comprehensive document validation.
        
        Args:
            document: ParsedDocument to validate
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_results = {
            'is_authentic': False,
            'credibility_score': 0.0,
            'quality_indicators': {},
            'suspicious_content': [],
            'validation_errors': [],
            'recommendations': []
        }
        
        try:
            # Check document authenticity
            authenticity_score = self._check_authenticity(document)
            validation_results['is_authentic'] = authenticity_score > 0.5
            
            # Assess content quality
            quality_score = self._assess_content_quality(document)
            validation_results['quality_indicators'] = self._get_quality_indicators(document)
            
            # Check for suspicious content
            suspicious_content = self._detect_suspicious_content(document)
            validation_results['suspicious_content'] = suspicious_content
            
            # Calculate overall credibility score
            credibility_score = self._calculate_credibility_score(
                document, authenticity_score, quality_score, len(suspicious_content)
            )
            validation_results['credibility_score'] = credibility_score
            
            # Determine if document should be accepted
            is_valid = self._determine_validity(
                authenticity_score, quality_score, suspicious_content, document
            )
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(
                document, authenticity_score, quality_score, suspicious_content
            )
            
            logger.info(f"Document {document.id} validation completed: valid={is_valid}, credibility={credibility_score:.2f}")
            return is_valid, validation_results
            
        except Exception as e:
            logger.error(f"Error validating document {document.id}: {e}")
            validation_results['validation_errors'].append(str(e))
            return False, validation_results
    
    def _check_authenticity(self, document: ParsedDocument) -> float:
        """
        Check document authenticity based on source and metadata.
        
        Returns:
            Authenticity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check for trusted journal
        source_lower = document.source.lower()
        metadata = document.metadata or {}
        
        # Check journal field in metadata
        journal = metadata.get('journal', '').lower()
        if any(trusted in journal for trusted in self.trusted_journals):
            score += 0.3
        
        # Check for DOI
        if metadata.get('doi') or 'doi' in document.content.lower():
            score += 0.2
        
        # Check for PMID or PubMed reference
        if any(term in document.content.lower() for term in ['pmid', 'pubmed']):
            score += 0.2
        
        # Check for proper author attribution
        if document.authors and len(document.authors) > 0:
            score += 0.1
        
        # Check publication date recency (more recent = potentially more credible)
        if document.publication_date:
            days_old = (datetime.now() - document.publication_date).days
            if days_old < 365 * 2:  # Within 2 years
                score += 0.1
            elif days_old < 365 * 5:  # Within 5 years
                score += 0.05
        
        # Check for institutional affiliation
        if any(term in document.content.lower() for term in ['university', 'hospital', 'medical center', 'institute']):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_content_quality(self, document: ParsedDocument) -> float:
        """
        Assess the quality of document content.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        content_lower = document.content.lower()
        
        # Check for quality indicators
        for indicator, pattern in self.quality_indicators.items():
            if pattern.search(document.content):
                score += 0.15
        
        # Check document length (substantial content)
        if len(document.content) > 5000:
            score += 0.1
        elif len(document.content) > 2000:
            score += 0.05
        
        # Check for structured content
        if document.metadata and document.metadata.get('element_types'):
            score += 0.1
        
        # Check for medical terminology density
        medical_terms = document.metadata.get('medical_term_counts', {})
        if medical_terms and len(medical_terms) > 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_suspicious_content(self, document: ParsedDocument) -> List[str]:
        """
        Detect suspicious content patterns that indicate low credibility.
        
        Returns:
            List of suspicious content found
        """
        suspicious_found = []
        
        for pattern in self.suspicious_patterns:
            matches = pattern.findall(document.content)
            if matches:
                suspicious_found.extend(matches)
        
        return suspicious_found
    
    def _get_quality_indicators(self, document: ParsedDocument) -> Dict[str, bool]:
        """Get quality indicators present in the document."""
        indicators = {}
        
        for indicator, pattern in self.quality_indicators.items():
            indicators[indicator] = bool(pattern.search(document.content))
        
        return indicators
    
    def _calculate_credibility_score(
        self, 
        document: ParsedDocument, 
        authenticity_score: float, 
        quality_score: float, 
        suspicious_count: int
    ) -> float:
        """
        Calculate overall credibility score.
        
        Args:
            document: The document being evaluated
            authenticity_score: Authenticity assessment score
            quality_score: Content quality score
            suspicious_count: Number of suspicious content patterns found
            
        Returns:
            Overall credibility score between 0.0 and 1.0
        """
        # Base score from authenticity and quality
        base_score = (authenticity_score * 0.6) + (quality_score * 0.4)
        
        # Penalty for suspicious content
        suspicious_penalty = min(0.3, suspicious_count * 0.1)
        
        # Bonus for document type (systematic reviews and RCTs are more credible)
        type_bonus = 0.0
        if document.document_type == DocumentType.SYSTEMATIC_REVIEW:
            type_bonus = 0.1
        elif document.document_type == DocumentType.CLINICAL_TRIAL:
            type_bonus = 0.08
        elif document.document_type == DocumentType.META_ANALYSIS:
            type_bonus = 0.09
        
        final_score = base_score + type_bonus - suspicious_penalty
        return max(0.0, min(1.0, final_score))
    
    def _determine_validity(
        self, 
        authenticity_score: float, 
        quality_score: float, 
        suspicious_content: List[str], 
        document: ParsedDocument
    ) -> bool:
        """
        Determine if document should be accepted for indexing.
        
        Args:
            authenticity_score: Document authenticity score
            quality_score: Content quality score
            suspicious_content: List of suspicious content found
            document: The document being evaluated
            
        Returns:
            True if document should be accepted, False otherwise
        """
        # Minimum thresholds
        min_authenticity = 0.3
        min_quality = 0.2
        max_suspicious = 2
        
        # Basic validity checks
        if authenticity_score < min_authenticity:
            logger.warning(f"Document {document.id} rejected: low authenticity score {authenticity_score:.2f}")
            return False
        
        if quality_score < min_quality:
            logger.warning(f"Document {document.id} rejected: low quality score {quality_score:.2f}")
            return False
        
        if len(suspicious_content) > max_suspicious:
            logger.warning(f"Document {document.id} rejected: too much suspicious content ({len(suspicious_content)} items)")
            return False
        
        # Additional checks for very short content
        if len(document.content) < 500:
            logger.warning(f"Document {document.id} rejected: content too short ({len(document.content)} chars)")
            return False
        
        return True
    
    def _generate_recommendations(
        self, 
        document: ParsedDocument, 
        authenticity_score: float, 
        quality_score: float, 
        suspicious_content: List[str]
    ) -> List[str]:
        """Generate recommendations for document handling."""
        recommendations = []
        
        if authenticity_score < 0.5:
            recommendations.append("Consider manual review for authenticity verification")
        
        if quality_score < 0.4:
            recommendations.append("Document may need additional quality assessment")
        
        if suspicious_content:
            recommendations.append(f"Review suspicious content: {', '.join(suspicious_content[:3])}")
        
        if not document.authors:
            recommendations.append("Missing author information - verify source")
        
        if document.credibility_score < 0.5:
            recommendations.append("Low credibility score - use with caution")
        
        return recommendations


def create_document_validator() -> DocumentValidator:
    """Factory function to create a document validator."""
    return DocumentValidator()


# Export main classes
__all__ = ['DocumentValidator', 'create_document_validator']