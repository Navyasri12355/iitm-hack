"""
Integration test for document validation and indexing pipeline.

Tests the complete pipeline from document validation through embedding generation
using mocked components to avoid external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os

from src.models.core import ParsedDocument, DocumentType
from src.ingestion.validation import DocumentValidator
from src.ingestion.vector_store import EmbeddingGenerator, PathwayVectorStore
from src.ingestion.indexing_pipeline import DocumentIndexingPipeline


class TestIndexingIntegration:
    """Integration tests for the complete indexing pipeline."""
    
    def setup_method(self):
        """Set up test environment with mocked components."""
        # Create validator (real component)
        self.validator = DocumentValidator()
        
        # Mock embedding generator
        self.mock_embedding_generator = Mock()
        self.mock_embedding_data = {
            'document_id': 'doc_001',
            'document_embedding': [0.1] * 1536,
            'chunk_embeddings': [[0.1] * 1536],
            'chunk_texts': ['test content'],
            'embedding_model': 'test-model',
            'embedding_dimension': 1536,
            'chunk_count': 1,
            'generated_at': datetime.now().isoformat(),
            'document_metadata': {
                'title': 'Test Document',
                'authors': ['Author'],
                'document_type': 'research_paper',
                'publication_date': datetime.now().isoformat(),
                'credibility_score': 0.8
            }
        }
        self.mock_embedding_generator.generate_embeddings.return_value = self.mock_embedding_data
        self.mock_embedding_generator._generate_single_embedding.return_value = [0.1] * 1536
        self.mock_embedding_generator.model = "test-model"
        
        # Mock vector store
        self.mock_vector_store = Mock()
        self.mock_vector_store.embedding_generator = self.mock_embedding_generator
        self.mock_vector_store._store_embeddings = Mock()
        self.mock_vector_store.similarity_search.return_value = [
            ('doc_001', 0.95, {'title': 'Test Document', 'credibility_score': 0.8})
        ]
        
        # Create pipeline with mocked components
        self.pipeline = DocumentIndexingPipeline(
            validator=self.validator,
            vector_store=self.mock_vector_store
        )
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly."""
        assert self.pipeline.validator is not None
        assert self.pipeline.vector_store is not None
        assert self.pipeline.stats['documents_processed'] == 0
    
    def test_single_document_validation(self):
        """Test validation of a single document."""
        # High-quality document
        good_document = ParsedDocument(
            id="doc_good",
            title="Clinical Trial of Hypertension Treatment",
            authors=["Dr. Smith", "Dr. Jones"],
            publication_date=datetime(2023, 6, 1),
            source="Medical Journal",
            document_type=DocumentType.CLINICAL_TRIAL,
            content="""
            This randomized controlled trial examined the effectiveness of ACE inhibitors
            in treating hypertension. The study included 2,500 patients (n=2500) with
            essential hypertension. Statistical analysis showed p<0.05 with 95% confidence
            interval. The methodology followed double-blind protocols with peer review.
            Results demonstrated significant improvement in blood pressure control.
            """ * 5,  # Make it substantial
            metadata={
                'journal': 'Hypertension Research',
                'doi': '10.1234/hr.2023.67890',
                'study_methodology': 'randomized controlled trial',
                'sample_size': 2500
            },
            credibility_score=0.8
        )
        
        is_valid, validation_details = self.pipeline.validate_single_document(good_document)
        
        assert is_valid is True
        assert validation_details['is_authentic'] is True
        assert validation_details['credibility_score'] > 0.7
        assert len(validation_details['suspicious_content']) == 0
    
    def test_single_document_indexing(self):
        """Test indexing of a single document."""
        # Create a valid document
        document = ParsedDocument(
            id="doc_index",
            title="Medical Research Paper",
            authors=["Dr. Author"],
            publication_date=datetime(2023, 6, 1),
            source="Medical Journal",
            document_type=DocumentType.RESEARCH_PAPER,
            content="""
            This is a comprehensive medical research paper about cardiovascular disease.
            The study methodology included randomized controlled trial design with
            statistical analysis showing p<0.05. Sample size was n=1000 patients.
            Results showed significant improvement in patient outcomes with confidence
            intervals reported. The study was peer-reviewed and published in a
            reputable medical journal with proper citations and references.
            """ * 10,  # Make it substantial
            metadata={
                'journal': 'Cardiology Today',
                'doi': '10.1234/ct.2023.12345',
                'medical_term_counts': {'methodology': 3, 'statistics': 2}
            },
            credibility_score=0.8
        )
        
        embedding_data = self.pipeline.index_single_document(document)
        
        assert embedding_data is not None
        assert embedding_data['document_id'] == 'doc_index'
        self.mock_vector_store._store_embeddings.assert_called_once()
    
    def test_low_quality_document_rejection(self):
        """Test that low-quality documents are rejected."""
        # Low-quality document
        bad_document = ParsedDocument(
            id="doc_bad",
            title="Miracle Cure",
            authors=[],
            publication_date=datetime(2010, 1, 1),
            source="Unknown Blog",
            document_type=DocumentType.RESEARCH_PAPER,
            content="""
            This miracle cure will solve all your health problems! Doctors hate this
            one weird trick that big pharma doesn't want you to know. 100% effective
            guaranteed results with no side effects. This is a conspiracy by the
            medical establishment to keep you sick.
            """,
            metadata={},
            credibility_score=0.2
        )
        
        # Should fail validation
        is_valid, validation_details = self.pipeline.validate_single_document(bad_document)
        assert is_valid is False
        
        # Should not be indexed
        embedding_data = self.pipeline.index_single_document(bad_document)
        assert embedding_data is None
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        results = self.pipeline.search_similar_documents(
            query_text="hypertension treatment",
            top_k=5
        )
        
        assert len(results) > 0
        assert 'document_id' in results[0]
        assert 'similarity_score' in results[0]
        assert 'title' in results[0]
        self.mock_vector_store.similarity_search.assert_called_once()
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics collection."""
        # Initial stats
        stats = self.pipeline.get_pipeline_stats()
        assert 'pipeline_stats' in stats
        assert 'vector_store_stats' in stats
        assert 'validation_success_rate' in stats
        assert 'indexing_success_rate' in stats
        
        # Stats should be properly calculated
        assert stats['validation_success_rate'] >= 0
        assert stats['indexing_success_rate'] >= 0
    
    def test_document_content_validation_edge_cases(self):
        """Test validation with edge cases."""
        # Very short document
        short_document = ParsedDocument(
            id="doc_short",
            title="Short",
            authors=["Author"],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.RESEARCH_PAPER,
            content="Too short",  # Very short content
            metadata={},
            credibility_score=0.5
        )
        
        is_valid, _ = self.pipeline.validate_single_document(short_document)
        assert is_valid is False  # Should be rejected for being too short
        
        # Document with medical terminology but suspicious content
        mixed_document = ParsedDocument(
            id="doc_mixed",
            title="Mixed Quality Document",
            authors=["Dr. Author"],
            publication_date=datetime.now(),
            source="test",
            document_type=DocumentType.RESEARCH_PAPER,
            content="""
            This clinical trial examined patient outcomes with n=500 subjects.
            Statistical analysis showed p<0.05 with confidence intervals.
            However, this miracle cure is guaranteed to work 100% of the time!
            The methodology was peer-reviewed but doctors hate this one trick.
            """ * 10,  # Make it substantial
            metadata={'medical_term_counts': {'methodology': 2}},
            credibility_score=0.6
        )
        
        is_valid, validation_details = self.pipeline.validate_single_document(mixed_document)
        # Should pass validation despite suspicious content due to medical terminology
        assert len(validation_details['suspicious_content']) > 0
        # Decision depends on overall scoring


def test_integration_with_real_sample_document():
    """Test integration with a realistic medical document."""
    # Create a realistic medical document based on the sample files
    realistic_document = ParsedDocument(
        id="doc_realistic",
        title="Randomized Controlled Trial: ACE Inhibitors vs ARBs in Hypertension",
        authors=["Wilson, A.", "Davis, R.", "Miller, S."],
        publication_date=datetime(2023, 8, 20),
        source="Hypertension Research",
        document_type=DocumentType.CLINICAL_TRIAL,
        content="""
        Randomized Controlled Trial: ACE Inhibitors vs ARBs in Hypertension

        Study Design: Double-blind, randomized controlled trial
        Sample Size: 2,500 patients (n=2500)
        Duration: 24 months

        Objective:
        To compare the effectiveness of ACE inhibitors versus ARBs in treating 
        essential hypertension in adults aged 40-70.

        Methodology:
        This randomized controlled trial used double-blind methodology with
        statistical analysis including p-values and confidence intervals.
        The study was peer-reviewed and published in a medical journal.

        Results:
        - ACE inhibitors: 78% achieved target BP (<140/90)
        - ARBs: 82% achieved target BP (<140/90)
        - Side effects: ACE inhibitors 15%, ARBs 8%
        - Statistical significance: p<0.05 with 95% confidence interval

        Conclusion:
        Both drug classes are effective, but ARBs show slightly better tolerability 
        and efficacy in this population. The methodology followed established
        clinical trial protocols with proper peer review.
        """,
        metadata={
            'journal': 'Hypertension Research',
            'doi': '10.1234/hr.2023.67890',
            'study_methodology': 'randomized controlled trial',
            'sample_size': 2500,
            'medical_term_counts': {
                'methodology': 3,
                'statistics': 2,
                'sample_size': 1,
                'peer_review': 2
            }
        },
        credibility_score=0.85
    )
    
    # Create pipeline with mocked vector store
    validator = DocumentValidator()
    mock_vector_store = Mock()
    mock_vector_store.embedding_generator = Mock()
    mock_vector_store.embedding_generator.generate_embeddings.return_value = {
        'document_id': 'doc_realistic',
        'document_embedding': [0.1] * 1536,
        'chunk_count': 1
    }
    mock_vector_store._store_embeddings = Mock()
    
    pipeline = DocumentIndexingPipeline(
        validator=validator,
        vector_store=mock_vector_store
    )
    
    # Test validation
    is_valid, validation_details = pipeline.validate_single_document(realistic_document)
    
    assert is_valid is True
    assert validation_details['is_authentic'] is True
    assert validation_details['credibility_score'] > 0.8
    assert validation_details['quality_indicators']['methodology'] is True
    assert validation_details['quality_indicators']['statistics'] is True
    assert validation_details['quality_indicators']['peer_review'] is True
    assert len(validation_details['suspicious_content']) == 0
    
    # Test indexing
    embedding_data = pipeline.index_single_document(realistic_document)
    assert embedding_data is not None
    mock_vector_store._store_embeddings.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])