"""
Query decomposition and routing logic for the Clinical Evidence Copilot.

This module implements agentic reasoning capabilities including:
- Query analysis and sub-task breakdown
- Search, filter, rank, summarize pipeline
- Ambiguity detection and clarification requests

Validates Requirements 3.1, 3.2:
- Complex medical queries are broken into logical sub-tasks
- Ambiguous queries trigger clarification requests
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.core import ClinicalQuery, UrgencyLevel, PatientContext


class SubTaskType(str, Enum):
    """Types of sub-tasks in the query processing pipeline."""
    SEARCH = "search"
    FILTER = "filter"
    RANK = "rank"
    SUMMARIZE = "summarize"
    CLARIFY = "clarify"


class QueryComplexity(str, Enum):
    """Complexity levels for clinical queries."""
    SIMPLE = "simple"          # Single condition/treatment query
    MODERATE = "moderate"      # Multiple related conditions
    COMPLEX = "complex"        # Multiple conditions with interactions
    AMBIGUOUS = "ambiguous"    # Unclear or incomplete query


@dataclass
class SubTask:
    """Represents a sub-task in the query processing pipeline."""
    task_type: SubTaskType
    description: str
    search_terms: List[str]
    filters: Dict[str, Any]
    priority: int
    requires_clarification: bool = False
    clarification_question: Optional[str] = None


@dataclass
class QueryAnalysis:
    """Results of query analysis and decomposition."""
    original_query: ClinicalQuery
    complexity: QueryComplexity
    medical_entities: List[str]
    sub_tasks: List[SubTask]
    ambiguities: List[str]
    clarification_needed: bool
    confidence_score: float


class QueryRouter:
    """
    Handles query decomposition and routing for agentic reasoning.
    
    Implements the search → filter → rank → summarize pipeline
    with ambiguity detection and clarification requests.
    """
    
    def __init__(self):
        """Initialize the query router with medical domain knowledge."""
        self.medical_keywords = self._load_medical_keywords()
        self.ambiguity_patterns = self._load_ambiguity_patterns()
        self.complexity_indicators = self._load_complexity_indicators()
    
    def analyze_query(self, query: ClinicalQuery) -> QueryAnalysis:
        """
        Analyze a clinical query and decompose it into sub-tasks.
        
        Args:
            query: The clinical query to analyze
            
        Returns:
            QueryAnalysis with decomposed sub-tasks and complexity assessment
        """
        # Extract medical entities from the query
        medical_entities = self._extract_medical_entities(query.query_text)
        
        # Assess query complexity
        complexity = self._assess_complexity(query.query_text, medical_entities)
        
        # Detect ambiguities
        ambiguities = self._detect_ambiguities(query.query_text)
        
        # Generate sub-tasks based on complexity and content
        sub_tasks = self._generate_subtasks(query, complexity, medical_entities, ambiguities)
        
        # Calculate confidence in our analysis
        confidence_score = self._calculate_analysis_confidence(
            complexity, len(ambiguities), len(medical_entities)
        )
        
        return QueryAnalysis(
            original_query=query,
            complexity=complexity,
            medical_entities=medical_entities,
            sub_tasks=sub_tasks,
            ambiguities=ambiguities,
            clarification_needed=len(ambiguities) > 0 or complexity == QueryComplexity.AMBIGUOUS,
            confidence_score=confidence_score
        )
    
    def _extract_medical_entities(self, query_text: str) -> List[str]:
        """Extract medical entities (conditions, treatments, drugs) from query text."""
        entities = []
        query_lower = query_text.lower()
        
        # Look for medical keywords
        for keyword in self.medical_keywords:
            if keyword.lower() in query_lower:
                entities.append(keyword)
        
        # Extract potential drug names (often capitalized or end with common suffixes)
        drug_patterns = [
            r'\b[A-Z][a-z]+(?:ine|ide|ate|ium|cin|zole|pril|sartan|olol)\b',
            r'\b(?:mg|mcg|units?)\b'
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, query_text)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)
        
        return unique_entities
    
    def _assess_complexity(self, query_text: str, medical_entities: List[str]) -> QueryComplexity:
        """Assess the complexity of a clinical query."""
        query_lower = query_text.lower()
        
        # Check for ambiguity indicators first
        for pattern in self.ambiguity_patterns:
            if re.search(pattern, query_lower):
                return QueryComplexity.AMBIGUOUS
        
        # Count complexity indicators
        complexity_score = 0
        
        # Multiple medical entities increase complexity
        complexity_score += len(medical_entities)
        
        # Look for complexity indicators
        for indicator in self.complexity_indicators:
            if indicator in query_lower:
                complexity_score += 2
        
        # Multiple questions or conditions
        question_count = query_text.count('?') + len(re.findall(r'\b(?:and|or|but|however)\b', query_lower))
        complexity_score += question_count
        
        # Classify based on score
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 5:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _detect_ambiguities(self, query_text: str) -> List[str]:
        """Detect ambiguous elements in the query that need clarification."""
        ambiguities = []
        query_lower = query_text.lower()
        
        # Check for vague terms
        vague_terms = [
            "best", "better", "good", "bad", "effective", "safe", "normal",
            "high", "low", "recent", "old", "young", "elderly"
        ]
        
        for term in vague_terms:
            if f" {term} " in f" {query_lower} ":
                ambiguities.append(f"Vague term '{term}' needs specification")
        
        # Check for missing context
        if any(word in query_lower for word in ["patient", "case", "condition"]):
            if not any(word in query_lower for word in ["age", "gender", "history"]):
                ambiguities.append("Patient context (age, gender, medical history) not specified")
        
        # Check for incomplete comparisons
        comparison_words = ["versus", "vs", "compared to", "better than", "worse than"]
        for comp in comparison_words:
            if comp in query_lower and query_text.count(comp) == 1:
                ambiguities.append(f"Incomplete comparison using '{comp}'")
        
        # Check for dosage/timing ambiguities
        if any(word in query_lower for word in ["dose", "dosage", "frequency", "duration"]):
            if not re.search(r'\d+\s*(?:mg|mcg|g|ml|times?|daily|weekly)', query_lower):
                ambiguities.append("Dosage or timing information is vague")
        
        return ambiguities
    
    def _generate_subtasks(
        self, 
        query: ClinicalQuery, 
        complexity: QueryComplexity,
        medical_entities: List[str],
        ambiguities: List[str]
    ) -> List[SubTask]:
        """Generate sub-tasks based on query analysis."""
        sub_tasks = []
        
        # If ambiguous, start with clarification
        if complexity == QueryComplexity.AMBIGUOUS or ambiguities:
            clarification_questions = self._generate_clarification_questions(ambiguities, query.query_text)
            sub_tasks.append(SubTask(
                task_type=SubTaskType.CLARIFY,
                description="Request clarification for ambiguous query elements",
                search_terms=[],
                filters={},
                priority=1,
                requires_clarification=True,
                clarification_question="; ".join(clarification_questions)
            ))
            return sub_tasks  # Don't proceed until clarification is received
        
        # Standard pipeline: Search → Filter → Rank → Summarize
        
        # 1. Search sub-task
        search_terms = medical_entities + self._extract_search_terms(query.query_text)
        sub_tasks.append(SubTask(
            task_type=SubTaskType.SEARCH,
            description=f"Search for evidence related to: {', '.join(search_terms)}",
            search_terms=search_terms,
            filters={},
            priority=1
        ))
        
        # 2. Filter sub-task
        filters = self._generate_filters(query, complexity)
        sub_tasks.append(SubTask(
            task_type=SubTaskType.FILTER,
            description="Filter evidence based on relevance and quality criteria",
            search_terms=search_terms,
            filters=filters,
            priority=2
        ))
        
        # 3. Rank sub-task
        sub_tasks.append(SubTask(
            task_type=SubTaskType.RANK,
            description="Rank evidence by hierarchy and relevance to clinical question",
            search_terms=search_terms,
            filters=filters,
            priority=3
        ))
        
        # 4. Summarize sub-task
        sub_tasks.append(SubTask(
            task_type=SubTaskType.SUMMARIZE,
            description="Synthesize evidence into clinical recommendation with reasoning",
            search_terms=search_terms,
            filters=filters,
            priority=4
        ))
        
        return sub_tasks
    
    def _generate_clarification_questions(self, ambiguities: List[str], query_text: str) -> List[str]:
        """Generate specific clarification questions based on detected ambiguities."""
        questions = []
        
        for ambiguity in ambiguities:
            if "patient context" in ambiguity.lower():
                questions.append("Could you provide patient demographics (age, gender) and relevant medical history?")
            elif "vague term" in ambiguity.lower():
                term = ambiguity.split("'")[1] if "'" in ambiguity else "term"
                questions.append(f"Could you be more specific about what you mean by '{term}'?")
            elif "incomplete comparison" in ambiguity.lower():
                questions.append("What specific treatments or interventions are you comparing?")
            elif "dosage" in ambiguity.lower():
                questions.append("What specific dosage, frequency, or duration are you asking about?")
            else:
                questions.append(f"Could you clarify: {ambiguity}")
        
        return questions
    
    def _extract_search_terms(self, query_text: str) -> List[str]:
        """Extract additional search terms from query text."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how'
        }
        
        # Extract words, filter stop words, and keep meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_text.lower())
        search_terms = [word for word in words if word not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Limit to top 10 terms
    
    def _generate_filters(self, query: ClinicalQuery, complexity: QueryComplexity) -> Dict[str, Any]:
        """Generate filters based on query characteristics."""
        filters = {}
        
        # Patient context filters
        if query.patient_context:
            if query.patient_context.age_range:
                filters['age_range'] = query.patient_context.age_range
            if query.patient_context.gender:
                filters['gender'] = query.patient_context.gender
            if query.patient_context.conditions:
                filters['conditions'] = query.patient_context.conditions
        
        # Urgency-based filters
        if query.urgency_level == UrgencyLevel.EMERGENCY:
            filters['evidence_level'] = ['systematic_review', 'meta_analysis', 'randomized_controlled_trial']
            filters['recency_years'] = 5
        elif query.urgency_level == UrgencyLevel.URGENT:
            filters['recency_years'] = 10
        
        # Complexity-based filters
        if complexity == QueryComplexity.COMPLEX:
            filters['min_sample_size'] = 100
            filters['evidence_level'] = ['systematic_review', 'meta_analysis', 'randomized_controlled_trial']
        
        return filters
    
    def _calculate_analysis_confidence(
        self, 
        complexity: QueryComplexity, 
        ambiguity_count: int, 
        entity_count: int
    ) -> float:
        """Calculate confidence score for the query analysis."""
        base_confidence = 0.8
        
        # Reduce confidence for ambiguous queries
        if complexity == QueryComplexity.AMBIGUOUS:
            base_confidence -= 0.3
        
        # Reduce confidence for each ambiguity
        base_confidence -= (ambiguity_count * 0.1)
        
        # Increase confidence for queries with clear medical entities
        if entity_count > 0:
            base_confidence += min(entity_count * 0.05, 0.2)
        
        return max(0.0, min(1.0, base_confidence))
    
    def _load_medical_keywords(self) -> List[str]:
        """Load medical keywords for entity extraction."""
        # In a real implementation, this would load from a medical ontology
        return [
            # Common conditions
            "hypertension", "diabetes", "asthma", "copd", "pneumonia", "sepsis",
            "myocardial infarction", "stroke", "heart failure", "arrhythmia",
            "depression", "anxiety", "schizophrenia", "bipolar", "dementia",
            "cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
            
            # Common treatments
            "surgery", "chemotherapy", "radiation", "immunotherapy",
            "antibiotic", "antiviral", "antifungal", "steroid",
            "beta blocker", "ace inhibitor", "statin", "insulin",
            
            # Medical specialties
            "cardiology", "oncology", "neurology", "psychiatry", "surgery",
            "emergency medicine", "internal medicine", "pediatrics",
            
            # Diagnostic terms
            "diagnosis", "prognosis", "screening", "biomarker", "imaging",
            "laboratory", "biopsy", "endoscopy", "ultrasound", "mri", "ct scan"
        ]
    
    def _load_ambiguity_patterns(self) -> List[str]:
        """Load regex patterns that indicate query ambiguity."""
        return [
            r'\b(?:maybe|perhaps|possibly|might be|could be)\b',
            r'\b(?:some|any|several|various|different)\b.*\b(?:options|choices|alternatives)\b',
            r'\?.*\?',  # Multiple question marks
            r'\b(?:not sure|uncertain|unclear|confused)\b',
            r'\b(?:what about|how about|what if)\b'
        ]
    
    def _load_complexity_indicators(self) -> List[str]:
        """Load terms that indicate query complexity."""
        return [
            "interaction", "contraindication", "side effect", "adverse event",
            "combination therapy", "polypharmacy", "comorbidity", "differential diagnosis",
            "risk stratification", "personalized medicine", "precision medicine",
            "multidisciplinary", "interdisciplinary", "complex case"
        ]


# Export the main class
__all__ = ['QueryRouter', 'QueryAnalysis', 'SubTask', 'SubTaskType', 'QueryComplexity']