"""
Recommendation generation with reasoning for the Clinical Evidence Copilot.

This module implements:
- LLM-based recommendation synthesis
- Reasoning explanation and citation generation
- Confidence scoring and change tracking

Validates Requirements 1.2, 3.4, 4.2, 4.3:
- Generate responses with specific medical literature sources and publication dates
- Show reasoning process and evidence evaluation steps
- Explain why recommendations changed
- Maintain history of previous recommendations with timestamps
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import json
import openai

from ..models.core import (
    ClinicalQuery, ClinicalRecommendation, Evidence, Contradiction,
    EvidenceLevel, UrgencyLevel
)
from ..config import get_settings
from .evidence_retrieval import MedicalContextualRetriever, SearchContext
from .query_router import QueryRouter, QueryAnalysis, SubTaskType

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process."""
    step_number: int
    step_type: str
    description: str
    evidence_used: List[str]  # Document IDs
    reasoning: str
    confidence: float


@dataclass
class RecommendationContext:
    """Context information for recommendation generation."""
    query_analysis: QueryAnalysis
    evidence_list: List[Evidence]
    contradictions: List[Contradiction]
    search_context: Optional[SearchContext] = None
    previous_recommendation: Optional[ClinicalRecommendation] = None


class ConfidenceCalculator:
    """
    Calculates confidence scores for clinical recommendations.
    
    Considers evidence quality, quantity, consistency, and recency.
    """
    
    def __init__(self):
        """Initialize the confidence calculator."""
        self.evidence_weights = {
            EvidenceLevel.SYSTEMATIC_REVIEW: 1.0,
            EvidenceLevel.META_ANALYSIS: 0.95,
            EvidenceLevel.RCT: 0.8,
            EvidenceLevel.COHORT_STUDY: 0.6,
            EvidenceLevel.CASE_CONTROL: 0.5,
            EvidenceLevel.OBSERVATIONAL: 0.4,
            EvidenceLevel.CASE_STUDY: 0.2,
            EvidenceLevel.EXPERT_OPINION: 0.1
        }
        logger.info("Initialized ConfidenceCalculator")
    
    def calculate_confidence(
        self, 
        evidence_list: List[Evidence], 
        contradictions: List[Contradiction],
        query_complexity: str = "moderate"
    ) -> float:
        """
        Calculate confidence score for a recommendation.
        
        Args:
            evidence_list: Supporting evidence
            contradictions: Any contradictory evidence
            query_complexity: Complexity of the original query
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not evidence_list:
            return 0.0
        
        # Base confidence from evidence quality
        quality_score = self._calculate_quality_score(evidence_list)
        
        # Quantity bonus (more evidence = higher confidence, with diminishing returns)
        quantity_score = min(len(evidence_list) / 10.0, 0.3)
        
        # Consistency penalty (contradictions reduce confidence)
        consistency_penalty = len(contradictions) * 0.15
        
        # Complexity penalty (complex queries are harder to answer confidently)
        complexity_penalties = {
            "simple": 0.0,
            "moderate": 0.05,
            "complex": 0.1,
            "ambiguous": 0.2
        }
        complexity_penalty = complexity_penalties.get(query_complexity, 0.05)
        
        # Calculate final confidence
        confidence = quality_score + quantity_score - consistency_penalty - complexity_penalty
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug(f"Calculated confidence: {confidence:.3f} (quality: {quality_score:.3f}, "
                    f"quantity: {quantity_score:.3f}, contradictions: -{consistency_penalty:.3f}, "
                    f"complexity: -{complexity_penalty:.3f})")
        
        return confidence
    
    def _calculate_quality_score(self, evidence_list: List[Evidence]) -> float:
        """Calculate quality score based on evidence levels and relevance."""
        if not evidence_list:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for evidence in evidence_list:
            # Evidence level weight
            level_weight = self.evidence_weights.get(evidence.evidence_level, 0.1)
            
            # Relevance weight
            relevance_weight = evidence.relevance_score
            
            # Sample size bonus (if available)
            sample_bonus = 1.0
            if evidence.sample_size:
                # Logarithmic bonus for larger samples
                sample_bonus = 1.0 + min(0.2, evidence.sample_size / 10000.0)
            
            # Combined weight
            combined_weight = level_weight * relevance_weight * sample_bonus
            
            total_weighted_score += combined_weight
            total_weight += level_weight
        
        # Normalize by total possible weight
        quality_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        return min(quality_score, 1.0)


class ReasoningExplainer:
    """
    Generates explanations for the reasoning process behind recommendations.
    
    Creates step-by-step explanations of how evidence was evaluated
    and how the recommendation was derived.
    """
    
    def __init__(self):
        """Initialize the reasoning explainer."""
        logger.info("Initialized ReasoningExplainer")
    
    def generate_reasoning_steps(
        self, 
        context: RecommendationContext
    ) -> List[ReasoningStep]:
        """
        Generate step-by-step reasoning explanation.
        
        Args:
            context: Recommendation context with query, evidence, and contradictions
            
        Returns:
            List of reasoning steps
        """
        steps = []
        step_number = 1
        
        # Step 1: Query Analysis
        steps.append(ReasoningStep(
            step_number=step_number,
            step_type="query_analysis",
            description="Analyzed clinical query and identified key medical concepts",
            evidence_used=[],
            reasoning=self._explain_query_analysis(context.query_analysis),
            confidence=context.query_analysis.confidence_score
        ))
        step_number += 1
        
        # Step 2: Evidence Search and Retrieval
        if context.evidence_list:
            steps.append(ReasoningStep(
                step_number=step_number,
                step_type="evidence_retrieval",
                description=f"Retrieved {len(context.evidence_list)} relevant evidence items",
                evidence_used=[e.document_id for e in context.evidence_list],
                reasoning=self._explain_evidence_retrieval(context.evidence_list),
                confidence=0.8  # Fixed confidence for retrieval step
            ))
            step_number += 1
        
        # Step 3: Evidence Evaluation and Ranking
        if context.evidence_list:
            steps.append(ReasoningStep(
                step_number=step_number,
                step_type="evidence_evaluation",
                description="Evaluated and ranked evidence according to medical hierarchy",
                evidence_used=[e.document_id for e in context.evidence_list[:5]],  # Top 5
                reasoning=self._explain_evidence_evaluation(context.evidence_list),
                confidence=0.9
            ))
            step_number += 1
        
        # Step 4: Contradiction Analysis (if any)
        if context.contradictions:
            steps.append(ReasoningStep(
                step_number=step_number,
                step_type="contradiction_analysis",
                description=f"Identified and analyzed {len(context.contradictions)} contradictions",
                evidence_used=[e.document_id for c in context.contradictions for e in c.conflicting_evidence],
                reasoning=self._explain_contradiction_analysis(context.contradictions),
                confidence=0.7  # Lower confidence when contradictions exist
            ))
            step_number += 1
        
        # Step 5: Recommendation Synthesis
        steps.append(ReasoningStep(
            step_number=step_number,
            step_type="recommendation_synthesis",
            description="Synthesized evidence into clinical recommendation",
            evidence_used=[e.document_id for e in context.evidence_list[:3]],  # Top 3
            reasoning=self._explain_recommendation_synthesis(context),
            confidence=0.85
        ))
        
        return steps
    
    def _explain_query_analysis(self, analysis: QueryAnalysis) -> str:
        """Explain the query analysis step."""
        explanation = f"Identified query complexity as '{analysis.complexity.value}' "
        
        if hasattr(analysis, 'medical_entities') and analysis.medical_entities:
            explanation += f"with {len(analysis.medical_entities)} medical entities: "
            explanation += ", ".join(analysis.medical_entities[:5])  # Limit to first 5
            if len(analysis.medical_entities) > 5:
                explanation += f" and {len(analysis.medical_entities) - 5} others"
        
        if hasattr(analysis, 'ambiguities') and analysis.ambiguities:
            explanation += f". Detected {len(analysis.ambiguities)} ambiguities requiring clarification."
        
        return explanation
    
    def _explain_evidence_retrieval(self, evidence_list: List[Evidence]) -> str:
        """Explain the evidence retrieval step."""
        if not evidence_list:
            return "No relevant evidence found in the knowledge base."
        
        # Count evidence by level
        level_counts = {}
        for evidence in evidence_list:
            level = evidence.evidence_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        explanation = f"Retrieved {len(evidence_list)} evidence items: "
        level_descriptions = []
        for level, count in sorted(level_counts.items()):
            level_descriptions.append(f"{count} {level.replace('_', ' ')}")
        
        explanation += ", ".join(level_descriptions)
        
        # Add relevance information
        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)
        explanation += f". Average relevance score: {avg_relevance:.2f}"
        
        return explanation
    
    def _explain_evidence_evaluation(self, evidence_list: List[Evidence]) -> str:
        """Explain the evidence evaluation step."""
        if not evidence_list:
            return "No evidence to evaluate."
        
        # Get top evidence
        top_evidence = evidence_list[0]
        
        explanation = f"Ranked evidence using medical hierarchy. "
        explanation += f"Highest quality evidence: {top_evidence.evidence_level.value.replace('_', ' ')} "
        explanation += f"(relevance: {top_evidence.relevance_score:.2f})"
        
        if top_evidence.sample_size:
            explanation += f" with sample size of {top_evidence.sample_size}"
        
        # Mention if there are multiple high-quality sources
        high_quality_count = sum(1 for e in evidence_list 
                                if e.evidence_level in [EvidenceLevel.SYSTEMATIC_REVIEW, 
                                                       EvidenceLevel.META_ANALYSIS, 
                                                       EvidenceLevel.RCT])
        if high_quality_count > 1:
            explanation += f". Found {high_quality_count} high-quality evidence sources supporting the analysis."
        
        return explanation
    
    def _explain_contradiction_analysis(self, contradictions: List[Contradiction]) -> str:
        """Explain the contradiction analysis step."""
        if not contradictions:
            return "No contradictions found between evidence sources."
        
        explanation = f"Identified {len(contradictions)} contradiction(s) between studies. "
        
        # Describe the first contradiction in detail
        first_contradiction = contradictions[0]
        explanation += f"Primary contradiction: {first_contradiction.explanation} "
        explanation += f"Resolution approach: {first_contradiction.resolution_guidance}"
        
        if len(contradictions) > 1:
            explanation += f" Additional {len(contradictions) - 1} contradiction(s) also considered."
        
        return explanation
    
    def _explain_recommendation_synthesis(self, context: RecommendationContext) -> str:
        """Explain the recommendation synthesis step."""
        explanation = "Synthesized evidence into clinical recommendation by: "
        
        synthesis_factors = []
        
        # Evidence quality consideration
        if context.evidence_list:
            top_level = context.evidence_list[0].evidence_level
            synthesis_factors.append(f"prioritizing {top_level.value.replace('_', ' ')} evidence")
        
        # Contradiction handling
        if context.contradictions:
            synthesis_factors.append("resolving contradictions through evidence hierarchy")
        
        # Query complexity consideration
        complexity = context.query_analysis.complexity.value
        if complexity in ["complex", "ambiguous"]:
            synthesis_factors.append(f"accounting for {complexity} query nature")
        
        explanation += ", ".join(synthesis_factors)
        explanation += ". Recommendation reflects current best evidence while acknowledging limitations."
        
        return explanation


class CitationGenerator:
    """
    Generates proper citations for medical literature sources.
    
    Creates formatted citations with publication dates and source information.
    """
    
    def __init__(self):
        """Initialize the citation generator."""
        logger.info("Initialized CitationGenerator")
    
    def generate_citations(
        self, 
        evidence_list: List[Evidence],
        document_metadata: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate formatted citations for evidence sources.
        
        Args:
            evidence_list: Evidence items to cite
            document_metadata: Optional metadata for documents
            
        Returns:
            List of formatted citations
        """
        citations = []
        
        for i, evidence in enumerate(evidence_list, 1):
            citation = self._format_citation(evidence, i, document_metadata)
            citations.append(citation)
        
        return citations
    
    def _format_citation(
        self, 
        evidence: Evidence, 
        citation_number: int,
        document_metadata: Dict[str, Dict[str, Any]] = None
    ) -> str:
        """Format a single citation."""
        # Get metadata if available
        metadata = {}
        if document_metadata and evidence.document_id in document_metadata:
            metadata = document_metadata[evidence.document_id]
        
        # Extract citation components
        title = metadata.get('title', f'Document {evidence.document_id}')
        authors = metadata.get('authors', ['Unknown Author'])
        journal = metadata.get('journal', 'Unknown Journal')
        pub_date = metadata.get('publication_date', 'Unknown Date')
        doi = metadata.get('doi', '')
        
        # Format authors
        if isinstance(authors, list) and authors:
            if len(authors) == 1:
                author_str = authors[0]
            elif len(authors) <= 3:
                author_str = ', '.join(authors[:-1]) + f' and {authors[-1]}'
            else:
                author_str = f'{authors[0]} et al.'
        else:
            author_str = 'Unknown Author'
        
        # Format date
        if isinstance(pub_date, datetime):
            date_str = pub_date.strftime('%Y')
        elif isinstance(pub_date, str):
            date_str = pub_date[:4] if len(pub_date) >= 4 else pub_date
        else:
            date_str = 'Unknown Date'
        
        # Create citation
        citation = f"[{citation_number}] {author_str}. {title}. {journal}. {date_str}."
        
        if doi:
            citation += f" DOI: {doi}"
        
        # Add evidence level and relevance information
        citation += f" (Evidence Level: {evidence.evidence_level.value.replace('_', ' ').title()}, "
        citation += f"Relevance: {evidence.relevance_score:.2f})"
        
        return citation


class RecommendationGenerator:
    """
    Main class for generating clinical recommendations with reasoning.
    
    Integrates evidence retrieval, reasoning explanation, citation generation,
    and confidence scoring to produce comprehensive clinical recommendations.
    """
    
    def __init__(self, evidence_retriever: MedicalContextualRetriever):
        """
        Initialize the recommendation generator.
        
        Args:
            evidence_retriever: Evidence retrieval system
        """
        self.evidence_retriever = evidence_retriever
        self.query_router = QueryRouter()
        self.confidence_calculator = ConfidenceCalculator()
        self.reasoning_explainer = ReasoningExplainer()
        self.citation_generator = CitationGenerator()
        
        # Initialize OpenAI client
        settings = get_settings()
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.openai_model = settings.openai_model
        else:
            logger.warning("OpenAI API key not configured - LLM features will be limited")
            self.openai_model = None
        
        # Store for tracking recommendation changes
        self.recommendation_history: Dict[str, List[ClinicalRecommendation]] = {}
        
        logger.info("Initialized RecommendationGenerator")
    
    def generate_recommendation(
        self, 
        query: ClinicalQuery,
        search_context: Optional[SearchContext] = None,
        document_metadata: Dict[str, Dict[str, Any]] = None
    ) -> ClinicalRecommendation:
        """
        Generate a comprehensive clinical recommendation with reasoning.
        
        Args:
            query: Clinical query to answer
            search_context: Optional search context
            document_metadata: Optional document metadata for citations
            
        Returns:
            Complete clinical recommendation with reasoning and citations
        """
        try:
            # Step 1: Analyze the query
            query_analysis = self.query_router.analyze_query(query)
            
            # Step 2: Handle ambiguous queries
            if query_analysis.clarification_needed:
                return self._create_clarification_recommendation(query, query_analysis)
            
            # Step 3: Retrieve evidence
            evidence_list, contradictions = self.evidence_retriever.retrieve_evidence(
                query, search_context
            )
            
            # Step 4: Create recommendation context
            context = RecommendationContext(
                query_analysis=query_analysis,
                evidence_list=evidence_list,
                contradictions=contradictions,
                search_context=search_context,
                previous_recommendation=self._get_previous_recommendation(query.id)
            )
            
            # Step 5: Generate reasoning steps
            reasoning_steps = self.reasoning_explainer.generate_reasoning_steps(context)
            
            # Step 6: Calculate confidence
            confidence_score = self.confidence_calculator.calculate_confidence(
                evidence_list, contradictions, query_analysis.complexity.value
            )
            
            # Step 7: Generate citations
            citations = self.citation_generator.generate_citations(
                evidence_list, document_metadata
            )
            
            # Step 8: Generate recommendation text using LLM
            recommendation_text = self._generate_recommendation_text(context, reasoning_steps)
            
            # Step 9: Determine change reason (if updating existing recommendation)
            change_reason = self._determine_change_reason(context)
            
            # Step 10: Create final recommendation
            recommendation = ClinicalRecommendation(
                id=f"rec_{query.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                query_id=query.id,
                recommendation_text=recommendation_text,
                supporting_evidence=evidence_list if evidence_list else [Evidence(
                    document_id="fallback",
                    relevance_score=0.0,
                    evidence_level=EvidenceLevel.EXPERT_OPINION,
                    excerpt="No specific evidence found - recommendation based on general clinical knowledge"
                )],
                confidence_score=confidence_score,
                contradictions=contradictions,
                last_updated=datetime.now(),
                change_reason=change_reason
            )
            
            # Step 11: Store in history for change tracking
            self._store_recommendation_history(recommendation)
            
            logger.info(f"Generated recommendation for query {query.id} with confidence {confidence_score:.3f}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation for query {query.id}: {e}")
            
            # Return a fallback recommendation
            return ClinicalRecommendation(
                id=f"rec_{query.id}_error",
                query_id=query.id,
                recommendation_text="Unable to generate recommendation due to system error. Please consult with colleagues or refer to current clinical guidelines.",
                supporting_evidence=[Evidence(
                    document_id="error_fallback",
                    relevance_score=0.0,
                    evidence_level=EvidenceLevel.EXPERT_OPINION,
                    excerpt="System error - no evidence available"
                )],
                confidence_score=0.0,
                contradictions=[],
                last_updated=datetime.now(),
                change_reason="System error during recommendation generation"
            )
    
    def _create_clarification_recommendation(
        self, 
        query: ClinicalQuery, 
        analysis: QueryAnalysis
    ) -> ClinicalRecommendation:
        """Create a recommendation requesting clarification."""
        clarification_questions = []
        
        if hasattr(analysis, 'sub_tasks') and analysis.sub_tasks:
            for subtask in analysis.sub_tasks:
                if hasattr(subtask, 'requires_clarification') and subtask.requires_clarification and hasattr(subtask, 'clarification_question') and subtask.clarification_question:
                    clarification_questions.append(subtask.clarification_question)
        
        if not clarification_questions:
            clarification_questions = ["Could you provide more specific details about your clinical question?"]
        
        recommendation_text = (
            "Your query requires clarification before I can provide evidence-based recommendations. "
            "Please provide additional information:\n\n" +
            "\n".join(f"• {q}" for q in clarification_questions)
        )
        
        return ClinicalRecommendation(
            id=f"rec_{query.id}_clarification",
            query_id=query.id,
            recommendation_text=recommendation_text,
            supporting_evidence=[Evidence(
                document_id="clarification_needed",
                relevance_score=0.0,
                evidence_level=EvidenceLevel.EXPERT_OPINION,
                excerpt="Clarification required before providing evidence-based recommendation"
            )],
            confidence_score=0.0,
            contradictions=[],
            last_updated=datetime.now(),
            change_reason="Clarification required for ambiguous query"
        )
    
    def _generate_recommendation_text(
        self, 
        context: RecommendationContext, 
        reasoning_steps: List[ReasoningStep]
    ) -> str:
        """Generate the main recommendation text using LLM or fallback logic."""
        if not self.openai_model:
            return self._generate_fallback_recommendation_text(context)
        
        try:
            # Prepare prompt for LLM
            prompt = self._create_llm_prompt(context, reasoning_steps)
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a clinical evidence expert providing evidence-based medical recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for more consistent medical advice
            )
            
            recommendation_text = response.choices[0].message.content.strip()
            
            # Add reasoning explanation
            reasoning_summary = self._create_reasoning_summary(reasoning_steps)
            recommendation_text += f"\n\n**Reasoning Process:**\n{reasoning_summary}"
            
            return recommendation_text
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return self._generate_fallback_recommendation_text(context)
    
    def _create_llm_prompt(
        self, 
        context: RecommendationContext, 
        reasoning_steps: List[ReasoningStep]
    ) -> str:
        """Create prompt for LLM recommendation generation."""
        query_text = context.query_analysis.original_query.query_text
        
        prompt = f"Clinical Query: {query_text}\n\n"
        
        # Add evidence summary
        if context.evidence_list:
            prompt += "Available Evidence:\n"
            for i, evidence in enumerate(context.evidence_list[:5], 1):
                prompt += f"{i}. {evidence.evidence_level.value.replace('_', ' ').title()}: {evidence.excerpt[:200]}...\n"
            prompt += "\n"
        
        # Add contradictions if any
        if context.contradictions:
            prompt += "Contradictory Evidence:\n"
            for i, contradiction in enumerate(context.contradictions, 1):
                prompt += f"{i}. {contradiction.explanation}\n"
            prompt += "\n"
        
        prompt += (
            "Please provide a clinical recommendation that:\n"
            "1. Directly answers the clinical question\n"
            "2. Is based on the available evidence\n"
            "3. Acknowledges any contradictions or limitations\n"
            "4. Includes specific citations to the evidence\n"
            "5. Provides actionable clinical guidance\n\n"
            "Format your response as a clear, professional clinical recommendation."
        )
        
        return prompt
    
    def _generate_fallback_recommendation_text(self, context: RecommendationContext) -> str:
        """Generate recommendation text without LLM."""
        query_text = context.query_analysis.original_query.query_text
        
        recommendation = f"Based on analysis of your query: '{query_text}'\n\n"
        
        if not context.evidence_list:
            recommendation += (
                "No specific evidence was found in the current knowledge base for this query. "
                "I recommend consulting current clinical guidelines and recent literature, "
                "or seeking consultation with relevant specialists."
            )
        else:
            # Summarize evidence
            top_evidence = context.evidence_list[0]
            recommendation += (
                f"Based on {len(context.evidence_list)} evidence sources, "
                f"with the highest quality evidence being {top_evidence.evidence_level.value.replace('_', ' ')} "
                f"(relevance: {top_evidence.relevance_score:.2f}):\n\n"
            )
            
            # Add evidence-based guidance
            if top_evidence.evidence_level in [EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.META_ANALYSIS]:
                recommendation += "Strong evidence supports the following approach:\n"
            elif top_evidence.evidence_level == EvidenceLevel.RCT:
                recommendation += "Good quality evidence suggests:\n"
            else:
                recommendation += "Available evidence indicates:\n"
            
            recommendation += f"• {top_evidence.excerpt[:300]}...\n\n"
            
            # Handle contradictions
            if context.contradictions:
                recommendation += (
                    f"Note: {len(context.contradictions)} contradiction(s) were identified in the evidence. "
                    "Consider individual patient factors and seek additional consultation if needed.\n\n"
                )
        
        # Add reasoning summary
        reasoning_summary = self._create_reasoning_summary(
            reasoning_steps=None,
            query_analysis=context.query_analysis, 
            evidence_list=context.evidence_list, 
            contradictions=context.contradictions
        )
        recommendation += f"**Reasoning Process:**\n{reasoning_summary}"
        
        return recommendation
    
    def _create_reasoning_summary(
        self, 
        reasoning_steps: List[ReasoningStep] = None,
        query_analysis: QueryAnalysis = None,
        evidence_list: List[Evidence] = None,
        contradictions: List[Contradiction] = None
    ) -> str:
        """Create a summary of the reasoning process."""
        if reasoning_steps:
            summary = ""
            for step in reasoning_steps:
                summary += f"{step.step_number}. {step.description}: {step.reasoning}\n"
            return summary
        
        # Fallback summary creation
        summary = ""
        if query_analysis:
            entities_count = 0
            if hasattr(query_analysis, 'medical_entities') and query_analysis.medical_entities:
                entities_count = len(query_analysis.medical_entities)
            complexity = "unknown"
            if hasattr(query_analysis, 'complexity'):
                complexity = query_analysis.complexity.value if hasattr(query_analysis.complexity, 'value') else str(query_analysis.complexity)
            summary += f"1. Query Analysis: Identified {entities_count} medical entities, complexity: {complexity}\n"
        
        if evidence_list:
            summary += f"2. Evidence Retrieval: Found {len(evidence_list)} relevant sources\n"
            summary += f"3. Evidence Evaluation: Ranked by medical hierarchy, top evidence: {evidence_list[0].evidence_level.value}\n"
        
        if contradictions:
            summary += f"4. Contradiction Analysis: Identified {len(contradictions)} conflicting findings\n"
        
        summary += "5. Recommendation Synthesis: Combined evidence using medical best practices\n"
        
        return summary
    
    def _get_previous_recommendation(self, query_id: str) -> Optional[ClinicalRecommendation]:
        """Get the most recent recommendation for a query."""
        if query_id in self.recommendation_history:
            history = self.recommendation_history[query_id]
            return history[-1] if history else None
        return None
    
    def _determine_change_reason(self, context: RecommendationContext) -> Optional[str]:
        """Determine why a recommendation changed from the previous version."""
        if not context.previous_recommendation:
            return None
        
        # Compare evidence
        prev_evidence_ids = {e.document_id for e in context.previous_recommendation.supporting_evidence}
        current_evidence_ids = {e.document_id for e in context.evidence_list}
        
        new_evidence = current_evidence_ids - prev_evidence_ids
        removed_evidence = prev_evidence_ids - current_evidence_ids
        
        reasons = []
        
        if new_evidence:
            reasons.append(f"New evidence added: {len(new_evidence)} documents")
        
        if removed_evidence:
            reasons.append(f"Evidence removed: {len(removed_evidence)} documents")
        
        # Compare contradictions
        prev_contradiction_count = len(context.previous_recommendation.contradictions)
        current_contradiction_count = len(context.contradictions)
        
        if current_contradiction_count > prev_contradiction_count:
            reasons.append("New contradictory evidence identified")
        elif current_contradiction_count < prev_contradiction_count:
            reasons.append("Contradictions resolved")
        
        # Compare confidence
        confidence_change = abs(
            context.previous_recommendation.confidence_score - 
            self.confidence_calculator.calculate_confidence(
                context.evidence_list, context.contradictions, 
                context.query_analysis.complexity.value
            )
        )
        
        if confidence_change > 0.1:
            reasons.append("Significant confidence change")
        
        return "; ".join(reasons) if reasons else "Evidence base updated"
    
    def _store_recommendation_history(self, recommendation: ClinicalRecommendation) -> None:
        """Store recommendation in history for change tracking."""
        query_id = recommendation.query_id
        
        if query_id not in self.recommendation_history:
            self.recommendation_history[query_id] = []
        
        self.recommendation_history[query_id].append(recommendation)
        
        # Keep only last 10 recommendations per query to manage memory
        if len(self.recommendation_history[query_id]) > 10:
            self.recommendation_history[query_id] = self.recommendation_history[query_id][-10:]
    
    def get_recommendation_history(self, query_id: str) -> List[ClinicalRecommendation]:
        """Get the history of recommendations for a query."""
        return self.recommendation_history.get(query_id, [])


# Export main classes
__all__ = [
    'RecommendationGenerator',
    'ConfidenceCalculator',
    'ReasoningExplainer',
    'CitationGenerator',
    'RecommendationContext',
    'ReasoningStep'
]