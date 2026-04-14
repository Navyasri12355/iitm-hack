"""
Clinical service layer for the FastAPI application.

Handles business logic and coordinates between API endpoints
and the underlying clinical reasoning components.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import asyncio

from ..models.core import (
    ClinicalQuery, ClinicalRecommendation, ParsedDocument,
    Evidence, DocumentType
)
from ..reasoning.recommendation_generator import RecommendationGenerator
from ..reasoning.evidence_retrieval import MedicalContextualRetriever, SearchContext
from ..ingestion.vector_store import PathwayVectorStore
from ..ingestion.parser import UnstructuredParser
from ..ingestion.validation import DocumentValidator
from ..config import get_settings
from .models import (
    DocumentUploadRequest, CitationInfo, ReasoningStep as APIReasoningStep
)
from .websocket import websocket_manager

logger = logging.getLogger(__name__)


class ClinicalService:
    """
    Main service class that coordinates clinical reasoning components.
    
    Provides high-level methods for processing queries, managing documents,
    and tracking recommendation changes.
    """
    
    def __init__(self):
        """Initialize the clinical service."""
        self.settings = get_settings()
        
        # Core components (initialized in initialize())
        self.vector_store: Optional[PathwayVectorStore] = None
        self.evidence_retriever: Optional[MedicalContextualRetriever] = None
        self.recommendation_generator: Optional[RecommendationGenerator] = None
        self.document_parser: Optional[UnstructuredParser] = None
        self.document_validator: Optional[DocumentValidator] = None
        
        # In-memory storage for demo (replace with proper database in production)
        self.documents: Dict[str, ParsedDocument] = {}
        self.recommendations: Dict[str, List[ClinicalRecommendation]] = {}
        
        logger.info("ClinicalService initialized")
    
    async def initialize(self):
        """Initialize all service components."""
        try:
            # Initialize vector store
            self.vector_store = PathwayVectorStore()
            await self._run_in_thread(self.vector_store.initialize)
            
            # Initialize evidence retriever
            self.evidence_retriever = MedicalContextualRetriever(self.vector_store)
            
            # Initialize recommendation generator
            self.recommendation_generator = RecommendationGenerator(self.evidence_retriever)
            
            # Initialize document processing components
            self.document_parser = UnstructuredParser()
            self.document_validator = DocumentValidator()
            
            logger.info("All clinical service components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clinical service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup service resources."""
        try:
            if self.vector_store:
                await self._run_in_thread(self.vector_store.cleanup)
            logger.info("Clinical service cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def process_query(self, query: ClinicalQuery) -> ClinicalRecommendation:
        """
        Process a clinical query and generate a recommendation.
        
        Args:
            query: The clinical query to process
            
        Returns:
            Clinical recommendation with evidence and reasoning
        """
        try:
            logger.info(f"Processing query {query.id}: {query.query_text[:100]}...")
            
            # Create search context based on query
            search_context = self._create_search_context(query)
            
            # Generate recommendation using the recommendation generator
            recommendation = await self._run_in_thread(
                self.recommendation_generator.generate_recommendation,
                query,
                search_context,
                self._get_document_metadata()
            )
            
            # Store recommendation for history tracking
            if query.id not in self.recommendations:
                self.recommendations[query.id] = []
            self.recommendations[query.id].append(recommendation)
            
            logger.info(f"Successfully processed query {query.id}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error processing query {query.id}: {e}")
            raise
    
    async def list_documents(
        self, 
        limit: int = 100, 
        offset: int = 0, 
        document_type: Optional[str] = None
    ) -> List[ParsedDocument]:
        """List documents in the knowledge base."""
        try:
            documents = list(self.documents.values())
            
            # Filter by document type if specified
            if document_type:
                documents = [
                    doc for doc in documents 
                    if doc.document_type.value == document_type
                ]
            
            # Apply pagination
            start = offset
            end = offset + limit
            
            return documents[start:end]
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise
    
    async def upload_document(self, request: DocumentUploadRequest) -> ParsedDocument:
        """
        Upload and process a new document.
        
        Args:
            request: Document upload request
            
        Returns:
            Processed ParsedDocument
        """
        try:
            logger.info(f"Uploading document: {request.title}")
            
            # Create document ID
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.documents) + 1:03d}"
            
            # Validate document content
            is_valid = await self._run_in_thread(
                self.document_validator.validate_content,
                request.content
            )
            
            if not is_valid:
                raise ValueError("Document content validation failed")
            
            # Calculate credibility score (simplified for demo)
            credibility_score = self._calculate_credibility_score(request)
            
            # Create ParsedDocument
            document = ParsedDocument(
                id=doc_id,
                title=request.title,
                authors=request.authors,
                publication_date=request.publication_date,
                source=request.source,
                document_type=request.document_type,
                content=request.content,
                metadata=request.metadata,
                credibility_score=credibility_score
            )
            
            # Store document
            self.documents[doc_id] = document
            
            logger.info(f"Successfully uploaded document {doc_id}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            raise
    
    async def index_document(self, document: ParsedDocument):
        """Index a document in the vector store (background task)."""
        try:
            logger.info(f"Indexing document {document.id}")
            
            # Add to vector store
            await self._run_in_thread(
                self.vector_store.add_document,
                document
            )
            
            # Update metadata to mark as indexed
            document.metadata['indexed_at'] = datetime.now()
            
            # Check if this document affects existing queries/recommendations
            affected_queries = await self._find_affected_queries(document)
            
            if affected_queries:
                # Notify via WebSocket about new evidence
                await websocket_manager.notify_new_evidence(
                    document_title=document.title,
                    affected_queries=affected_queries,
                    evidence_level=self._infer_evidence_level_from_document(document)
                )
            
            logger.info(f"Successfully indexed document {document.id}")
            
        except Exception as e:
            logger.error(f"Error indexing document {document.id}: {e}")
    
    async def _find_affected_queries(self, document: ParsedDocument) -> List[str]:
        """Find queries that might be affected by a new document."""
        # In a full implementation, this would use semantic similarity
        # to find queries that might be affected by the new document
        # For now, return a simple heuristic based on stored recommendations
        
        affected_queries = []
        document_keywords = self._extract_keywords(document.content)
        
        for query_id, recommendations in self.recommendations.items():
            if recommendations:  # Check if there are any recommendations for this query
                # Simple keyword matching (in production, use semantic similarity)
                latest_rec = recommendations[-1]
                rec_text = latest_rec.recommendation_text.lower()
                
                # Check if any document keywords appear in the recommendation
                for keyword in document_keywords:
                    if keyword.lower() in rec_text:
                        affected_queries.append(query_id)
                        break
        
        return affected_queries
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from document text (simplified)."""
        # In production, use proper NLP for keyword extraction
        import re
        
        # Simple keyword extraction - get words longer than 4 characters
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        
        # Medical terms that are commonly important
        medical_terms = [
            'hypertension', 'diabetes', 'treatment', 'therapy', 'medication',
            'clinical', 'patient', 'study', 'trial', 'efficacy', 'safety'
        ]
        
        keywords = []
        for word in words[:20]:  # Limit to first 20 words
            if word in medical_terms or len(word) > 6:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _infer_evidence_level_from_document(self, document: ParsedDocument) -> str:
        """Infer evidence level from document type."""
        type_mapping = {
            DocumentType.SYSTEMATIC_REVIEW: "systematic_review",
            DocumentType.META_ANALYSIS: "meta_analysis", 
            DocumentType.CLINICAL_TRIAL: "randomized_controlled_trial",
            DocumentType.RESEARCH_PAPER: "observational",
            DocumentType.GUIDELINE: "systematic_review",
            DocumentType.CASE_STUDY: "case_study"
        }
        return type_mapping.get(document.document_type, "observational")
    
    async def get_document(self, document_id: str) -> Optional[ParsedDocument]:
        """Get a specific document by ID."""
        return self.documents.get(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base."""
        try:
            if document_id not in self.documents:
                return False
            
            # Remove from vector store
            await self._run_in_thread(
                self.vector_store.remove_document,
                document_id
            )
            
            # Remove from local storage
            del self.documents[document_id]
            
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def cleanup_document_references(self, document_id: str):
        """Clean up references to a deleted document (background task)."""
        try:
            # Remove document references from recommendations
            for query_id, recommendations in self.recommendations.items():
                for recommendation in recommendations:
                    # Remove evidence that references the deleted document
                    recommendation.supporting_evidence = [
                        evidence for evidence in recommendation.supporting_evidence
                        if evidence.document_id != document_id
                    ]
            
            logger.info(f"Cleaned up references to document {document_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up document references: {e}")
    
    async def get_recommendation_history(self, query_id: str) -> List[ClinicalRecommendation]:
        """Get the history of recommendations for a query."""
        return self.recommendations.get(query_id, [])
    
    async def get_recent_recommendations(
        self, 
        limit: int = 50, 
        clinician_id: Optional[str] = None
    ) -> List[ClinicalRecommendation]:
        """Get recent recommendations, optionally filtered by clinician."""
        try:
            all_recommendations = []
            
            # Collect all recommendations
            for recommendations in self.recommendations.values():
                all_recommendations.extend(recommendations)
            
            # Filter by clinician if specified
            if clinician_id:
                # Note: We'd need to store clinician_id in recommendations for this to work
                # For now, we'll return all recommendations
                pass
            
            # Sort by timestamp (most recent first)
            all_recommendations.sort(key=lambda r: r.last_updated, reverse=True)
            
            return all_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent recommendations: {e}")
            return []
    
    async def notify_recommendation_change(self, recommendation: ClinicalRecommendation):
        """Send notifications for recommendation changes (background task)."""
        try:
            if not recommendation.change_reason:
                return
            
            logger.info(f"Notifying recommendation change for {recommendation.query_id}: {recommendation.change_reason}")
            
            # In a real implementation, this would:
            # 1. Find clinicians who previously queried similar topics
            # 2. Send notifications via WebSocket, email, or push notifications
            # 3. Store notifications in a database
            
            # For now, just log the notification
            logger.info(f"Notification sent for recommendation change: {recommendation.id}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def get_citations_for_recommendation(self, recommendation: ClinicalRecommendation) -> List[CitationInfo]:
        """Generate citation information for a recommendation."""
        try:
            citations = []
            
            for i, evidence in enumerate(recommendation.supporting_evidence, 1):
                # Get document metadata
                document = self.documents.get(evidence.document_id)
                
                if document:
                    citation = CitationInfo(
                        citation_number=i,
                        title=document.title,
                        authors=document.authors,
                        journal=document.source,
                        publication_date=document.publication_date.strftime('%Y'),
                        doi=document.metadata.get('doi'),
                        evidence_level=evidence.evidence_level.value.replace('_', ' ').title(),
                        relevance_score=evidence.relevance_score
                    )
                    citations.append(citation)
                else:
                    # Fallback citation for missing documents
                    citation = CitationInfo(
                        citation_number=i,
                        title=f"Document {evidence.document_id}",
                        authors=["Unknown Author"],
                        journal="Unknown Source",
                        publication_date="Unknown",
                        doi=None,
                        evidence_level=evidence.evidence_level.value.replace('_', ' ').title(),
                        relevance_score=evidence.relevance_score
                    )
                    citations.append(citation)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            return []
    
    def get_reasoning_steps(self, recommendation: ClinicalRecommendation) -> List[APIReasoningStep]:
        """Get reasoning steps for a recommendation."""
        try:
            # In a full implementation, reasoning steps would be stored with the recommendation
            # For now, generate basic reasoning steps
            
            steps = [
                APIReasoningStep(
                    step_number=1,
                    step_type="evidence_retrieval",
                    description=f"Retrieved {len(recommendation.supporting_evidence)} evidence sources",
                    reasoning=f"Found evidence from {len(set(e.evidence_level for e in recommendation.supporting_evidence))} different evidence levels",
                    confidence=0.8
                ),
                APIReasoningStep(
                    step_number=2,
                    step_type="evidence_evaluation",
                    description="Evaluated evidence quality and relevance",
                    reasoning=f"Highest evidence level: {recommendation.supporting_evidence[0].evidence_level.value if recommendation.supporting_evidence else 'none'}",
                    confidence=0.85
                ),
                APIReasoningStep(
                    step_number=3,
                    step_type="recommendation_synthesis",
                    description="Synthesized evidence into clinical recommendation",
                    reasoning=f"Generated recommendation with {recommendation.confidence_score:.2f} confidence",
                    confidence=recommendation.confidence_score
                )
            ]
            
            if recommendation.contradictions:
                steps.insert(2, APIReasoningStep(
                    step_number=3,
                    step_type="contradiction_analysis",
                    description=f"Analyzed {len(recommendation.contradictions)} contradictions",
                    reasoning="Identified conflicting evidence and provided resolution guidance",
                    confidence=0.7
                ))
                # Update subsequent step numbers
                for i, step in enumerate(steps[3:], 4):
                    step.step_number = i
            
            return steps
            
        except Exception as e:
            logger.error(f"Error generating reasoning steps: {e}")
            return []
    
    def _create_search_context(self, query: ClinicalQuery) -> SearchContext:
        """Create search context from clinical query."""
        context = SearchContext()
        
        # Set urgency-based preferences
        if query.urgency_level.value == "emergency":
            context.evidence_recency_preference = 2  # Very recent evidence
            context.minimum_evidence_level = None  # Accept any evidence level
        elif query.urgency_level.value == "urgent":
            context.evidence_recency_preference = 5
        else:
            context.evidence_recency_preference = 10
        
        # Set patient demographics if available
        if query.patient_context:
            context.patient_demographics = {
                "age_range": query.patient_context.age_range,
                "gender": query.patient_context.gender,
                "conditions": query.patient_context.conditions,
                "medications": query.patient_context.medications,
                "allergies": query.patient_context.allergies
            }
        
        return context
    
    def _calculate_credibility_score(self, request: DocumentUploadRequest) -> float:
        """Calculate credibility score for a document (simplified)."""
        score = 0.5  # Base score
        
        # Boost score based on document type
        type_scores = {
            DocumentType.SYSTEMATIC_REVIEW: 0.4,
            DocumentType.META_ANALYSIS: 0.35,
            DocumentType.CLINICAL_TRIAL: 0.3,
            DocumentType.RESEARCH_PAPER: 0.2,
            DocumentType.GUIDELINE: 0.35,
            DocumentType.CASE_STUDY: 0.1
        }
        score += type_scores.get(request.document_type, 0.1)
        
        # Boost score for recent publications
        years_old = (datetime.now() - request.publication_date).days / 365.25
        if years_old < 2:
            score += 0.1
        elif years_old < 5:
            score += 0.05
        
        # Boost score for multiple authors
        if len(request.authors) > 1:
            score += 0.05
        
        return min(1.0, score)
    
    def _get_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all documents."""
        metadata = {}
        for doc_id, document in self.documents.items():
            metadata[doc_id] = {
                'title': document.title,
                'authors': document.authors,
                'journal': document.source,
                'publication_date': document.publication_date,
                'doi': document.metadata.get('doi'),
                'document_type': document.document_type.value
            }
        return metadata
    
    async def _run_in_thread(self, func, *args, **kwargs):
        """Run a synchronous function in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)


# Export the service class
__all__ = ['ClinicalService']