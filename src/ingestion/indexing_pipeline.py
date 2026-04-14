"""
Integrated document indexing pipeline for clinical evidence copilot.

This module combines document validation, embedding generation, and vector storage
into a unified pipeline for real-time document processing.

Validates Requirements 2.4, 2.5:
- Implement document authenticity validation
- Set up vector embedding generation using OpenAI embeddings
- Create Pathway vector store for real-time indexing
- Maintain system availability during ingestion
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False

from ..config import get_settings
from ..models.core import ParsedDocument
from .validation import DocumentValidator, create_document_validator
from .vector_store import PathwayVectorStore, create_vector_store

logger = logging.getLogger(__name__)


class DocumentIndexingPipeline:
    """
    Integrated pipeline for document validation, embedding generation, and indexing.
    
    Combines validation and vector storage components to provide a complete
    document processing pipeline with real-time capabilities.
    """
    
    def __init__(
        self, 
        validator: Optional[DocumentValidator] = None,
        vector_store: Optional[PathwayVectorStore] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the indexing pipeline.
        
        Args:
            validator: DocumentValidator instance (creates one if not provided)
            vector_store: PathwayVectorStore instance (creates one if not provided)
            openai_api_key: OpenAI API key for embeddings
        """
        if not PATHWAY_AVAILABLE:
            raise ImportError("Pathway library not available. Install with: pip install pathway")
        
        self.settings = get_settings()
        self.validator = validator or create_document_validator()
        self.vector_store = vector_store or create_vector_store(api_key=openai_api_key)
        
        # Pipeline statistics
        self.stats = {
            'documents_processed': 0,
            'documents_validated': 0,
            'documents_indexed': 0,
            'validation_failures': 0,
            'indexing_failures': 0,
            'last_processed': None
        }
        
        logger.info("Initialized DocumentIndexingPipeline")
    
    def create_indexing_pipeline(self, documents_table: pw.Table) -> pw.Table:
        """
        Create complete Pathway indexing pipeline.
        
        Args:
            documents_table: Pathway table with processed documents
            
        Returns:
            Pathway table with indexed documents
        """
        # Step 1: Validate documents
        validated_docs = self._create_validation_step(documents_table)
        
        # Step 2: Generate embeddings for validated documents
        indexed_docs = self._create_embedding_step(validated_docs)
        
        # Step 3: Set up monitoring and statistics
        self._setup_monitoring(indexed_docs)
        
        return indexed_docs
    
    def _create_validation_step(self, documents_table: pw.Table) -> pw.Table:
        """Create document validation step in the pipeline."""
        
        def validate_document(row):
            """Validate a single document."""
            try:
                parsed_doc = row.result['parsed_document']
                if not parsed_doc:
                    return {
                        'document_id': row.result.get('document_id'),
                        'parsed_document': None,
                        'validation_result': None,
                        'status': 'no_document',
                        'error': 'No parsed document available'
                    }
                
                # Perform validation
                is_valid, validation_details = self.validator.validate_document(parsed_doc)
                
                # Update statistics
                self.stats['documents_processed'] += 1
                if is_valid:
                    self.stats['documents_validated'] += 1
                else:
                    self.stats['validation_failures'] += 1
                
                logger.info(f"Document {parsed_doc.id} validation: {'PASSED' if is_valid else 'FAILED'}")
                
                return {
                    'document_id': parsed_doc.id,
                    'parsed_document': parsed_doc if is_valid else None,
                    'validation_result': validation_details,
                    'status': 'validated' if is_valid else 'validation_failed',
                    'credibility_score': validation_details.get('credibility_score', 0.0)
                }
                
            except Exception as e:
                logger.error(f"Error validating document: {e}")
                self.stats['validation_failures'] += 1
                return {
                    'document_id': row.result.get('document_id'),
                    'parsed_document': None,
                    'validation_result': None,
                    'status': 'validation_error',
                    'error': str(e)
                }
        
        # Apply validation to documents
        validated_table = documents_table.select(
            validation_result=pw.apply(validate_document, pw.this)
        )
        
        # Filter successful validations
        successful_validations = validated_table.filter(
            lambda row: row.validation_result['status'] == 'validated'
        )
        
        # Log validation failures
        validation_failures = validated_table.filter(
            lambda row: row.validation_result['status'] in ['validation_failed', 'validation_error', 'no_document']
        )
        validation_failures.debug("validation_failures")
        
        return successful_validations
    
    def _create_embedding_step(self, validated_docs: pw.Table) -> pw.Table:
        """Create embedding generation step in the pipeline."""
        
        def generate_embeddings_with_validation(row):
            """Generate embeddings for validated documents."""
            try:
                validation_result = row.validation_result
                parsed_doc = validation_result['parsed_document']
                
                if not parsed_doc:
                    return {
                        'document_id': validation_result['document_id'],
                        'status': 'no_document_for_embedding',
                        'error': 'No validated document available for embedding'
                    }
                
                # Generate embeddings using the vector store
                embedding_data = self.vector_store.embedding_generator.generate_embeddings(parsed_doc)
                
                # Store embeddings
                self.vector_store._store_embeddings(embedding_data)
                
                # Update statistics
                self.stats['documents_indexed'] += 1
                self.stats['last_processed'] = datetime.now()
                
                logger.info(f"Successfully indexed document {parsed_doc.id}")
                
                return {
                    'document_id': parsed_doc.id,
                    'embedding_data': embedding_data,
                    'validation_result': validation_result,
                    'status': 'indexed',
                    'indexed_at': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                self.stats['indexing_failures'] += 1
                return {
                    'document_id': row.validation_result.get('document_id'),
                    'status': 'indexing_error',
                    'error': str(e)
                }
        
        # Apply embedding generation
        indexed_table = validated_docs.select(
            indexing_result=pw.apply(generate_embeddings_with_validation, pw.this)
        )
        
        # Filter successful indexing
        successful_indexing = indexed_table.filter(
            lambda row: row.indexing_result['status'] == 'indexed'
        )
        
        # Log indexing failures
        indexing_failures = indexed_table.filter(
            lambda row: row.indexing_result['status'] in ['indexing_error', 'no_document_for_embedding']
        )
        indexing_failures.debug("indexing_failures")
        
        return successful_indexing
    
    def _setup_monitoring(self, indexed_docs: pw.Table) -> None:
        """Set up monitoring and statistics collection."""
        
        def log_successful_indexing(row):
            """Log successful document indexing."""
            result = row.indexing_result
            logger.info(f"Document successfully indexed: {result['document_id']}")
            return result
        
        # Set up monitoring output
        monitoring_table = indexed_docs.select(
            monitoring_result=pw.apply(log_successful_indexing, pw.this)
        )
        
        # Debug output for monitoring
        monitoring_table.debug("successful_indexing")
    
    def search_similar_documents(
        self, 
        query_text: str, 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the indexed embeddings.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            filters: Optional filters to apply
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query_text=query_text,
                top_k=top_k,
                filters=filters
            )
            
            # Format results
            formatted_results = []
            for document_id, similarity_score, metadata in results:
                formatted_results.append({
                    'document_id': document_id,
                    'similarity_score': similarity_score,
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', []),
                    'document_type': metadata.get('document_type', 'unknown'),
                    'publication_date': metadata.get('publication_date'),
                    'credibility_score': metadata.get('credibility_score', 0.0)
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        vector_stats = self.vector_store.get_store_stats()
        
        return {
            'pipeline_stats': self.stats.copy(),
            'vector_store_stats': vector_stats,
            'validation_success_rate': (
                self.stats['documents_validated'] / max(1, self.stats['documents_processed'])
            ) * 100,
            'indexing_success_rate': (
                self.stats['documents_indexed'] / max(1, self.stats['documents_validated'])
            ) * 100
        }
    
    def validate_single_document(self, document: ParsedDocument) -> tuple[bool, Dict[str, Any]]:
        """
        Validate a single document outside the pipeline.
        
        Args:
            document: ParsedDocument to validate
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        return self.validator.validate_document(document)
    
    def index_single_document(self, document: ParsedDocument) -> Optional[Dict[str, Any]]:
        """
        Index a single document outside the pipeline.
        
        Args:
            document: ParsedDocument to index
            
        Returns:
            Embedding data if successful, None otherwise
        """
        try:
            # Validate first
            is_valid, validation_details = self.validator.validate_document(document)
            
            if not is_valid:
                logger.warning(f"Document {document.id} failed validation, not indexing")
                return None
            
            # Generate and store embeddings
            embedding_data = self.vector_store.embedding_generator.generate_embeddings(document)
            self.vector_store._store_embeddings(embedding_data)
            
            logger.info(f"Successfully indexed single document {document.id}")
            return embedding_data
            
        except Exception as e:
            logger.error(f"Error indexing single document {document.id}: {e}")
            return None


def create_indexing_pipeline(openai_api_key: Optional[str] = None) -> DocumentIndexingPipeline:
    """
    Factory function to create a DocumentIndexingPipeline instance.
    
    Args:
        openai_api_key: OpenAI API key for embeddings
        
    Returns:
        Configured DocumentIndexingPipeline instance
    """
    return DocumentIndexingPipeline(openai_api_key=openai_api_key)


# Export main classes
__all__ = ['DocumentIndexingPipeline', 'create_indexing_pipeline']