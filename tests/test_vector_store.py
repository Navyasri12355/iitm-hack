"""
Tests for vector store functionality.

Tests the embedding generation and vector storage capabilities
without requiring actual OpenAI API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np

from src.ingestion.vector_store import EmbeddingGenerator, PathwayVectorStore, create_vector_store
from src.models.core import ParsedDocument, DocumentType


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator (mocked)."""
    
    def setup_method(self):
        """Set up test environment with mocked OpenAI."""
        # Mock the OpenAI client
        self.mock_client = Mock()
        self.mock_response = Mock()
        self.mock_response.data = [Mock()]
        self.mock_response.data[0].embedding = [0.1] * 1536  # Mock embedding vector
        self.mock_client.embeddings.create.return_value = self.mock_response
        
        # Create generator with mocked client
        with patch('src.ingestion.vector_store.openai.OpenAI') as mock_openai:
            mock_openai.return_value = self.mock_client
            self.generator = EmbeddingGenerator(api_key="test_key")
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        assert self.generator.api_key == "test_key"
        assert self.generator.model == "text-embedding-ada-002"  # Default model
        assert self.generator.max_chunk_size == 8000
    
    def test_text_chunking(self):
        """Test text splitting into chunks."""
        # Short text (no chunking needed)
        short_text = "This is a short text."
        chunks = self.generator._split_into_chunks(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Long text (chunking needed)
        long_text = "This is a sentence. " * 500  # Create long text
        chunks = self.generator._split_into_chunks(long_text)
        assert len(chunks) > 1
        
        # Check overlap
        if len(chunks) > 1:
            # There should be some overlap between chunks
            assert len(chunks[0]) <= self.generator.max_chunk_size
    
    def test_single_embedding_generation(self):
        """Test generation of single embedding."""
        text = "This is test medical content about hypertension treatment."
        
        embedding = self.generator._generate_single_embedding(text)
        
        assert len(embedding) == 1536  # Expected embedding dimension
        assert all(isinstance(x, float) for x in embedding)
        self.mock_client.embeddings.create.assert_called_once()
    
    def test_document_embedding_generation(self):
        """Test generation of embeddings for a complete document."""
        document = ParsedDocument(
            id="doc_001",
            title="Test Medical Document",
            authors=["Dr. Smith"],
            publication_date=datetime(2023, 6, 1),
            source="test_source",
            document_type=DocumentType.RESEARCH_PAPER,
            content="This is a medical document about hypertension treatment. " * 100,  # Make it substantial
            metadata={'study_type': 'clinical_trial'},
            credibility_score=0.8
        )
        
        embedding_data = self.generator.generate_embeddings(document)
        
        assert embedding_data['document_id'] == "doc_001"
        assert 'document_embedding' in embedding_data
        assert 'chunk_embeddings' in embedding_data
        assert 'chunk_texts' in embedding_data
        assert len(embedding_data['document_embedding']) == 1536
        assert embedding_data['chunk_count'] > 0
        assert 'document_metadata' in embedding_data


class TestPathwayVectorStore:
    """Test suite for PathwayVectorStore (mocked)."""
    
    def setup_method(self):
        """Set up test environment with mocked components."""
        # Mock embedding generator
        self.mock_generator = Mock()
        self.mock_generator.model = "test-model"
        self.mock_generator._generate_single_embedding.return_value = [0.1] * 1536
        
        # Mock embedding data
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
        
        self.mock_generator.generate_embeddings.return_value = self.mock_embedding_data
        
        # Create vector store with mocked generator
        self.vector_store = PathwayVectorStore(embedding_generator=self.mock_generator)
    
    def test_vector_store_initialization(self):
        """Test that vector store initializes correctly."""
        assert self.vector_store.embedding_generator == self.mock_generator
        assert self.vector_store.vector_dimension == 1536
        assert len(self.vector_store._embeddings_store) == 0
    
    def test_embedding_storage(self):
        """Test storing embeddings in the vector store."""
        self.vector_store._store_embeddings(self.mock_embedding_data)
        
        assert len(self.vector_store._embeddings_store) == 1
        assert 'doc_001' in self.vector_store._document_index
        
        # Test retrieval
        stored_data = self.vector_store.get_document_embedding('doc_001')
        assert stored_data is not None
        assert stored_data['document_id'] == 'doc_001'
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]
        
        # Orthogonal vectors should have similarity 0
        similarity_orthogonal = self.vector_store._cosine_similarity(vec1, vec2)
        assert abs(similarity_orthogonal) < 1e-10
        
        # Identical vectors should have similarity 1
        similarity_identical = self.vector_store._cosine_similarity(vec1, vec3)
        assert abs(similarity_identical - 1.0) < 1e-10
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Store some test embeddings
        self.vector_store._store_embeddings(self.mock_embedding_data)
        
        # Store another document with different embedding
        different_embedding_data = self.mock_embedding_data.copy()
        different_embedding_data['document_id'] = 'doc_002'
        different_embedding_data['document_embedding'] = [0.2] * 1536
        different_embedding_data['document_metadata']['title'] = 'Different Document'
        self.vector_store._store_embeddings(different_embedding_data)
        
        # Perform search
        results = self.vector_store.similarity_search("test query", top_k=5)
        
        assert len(results) <= 2  # Should find both documents
        assert all(len(result) == 3 for result in results)  # (doc_id, similarity, metadata)
        assert all(isinstance(result[1], float) for result in results)  # Similarity scores
    
    def test_search_with_filters(self):
        """Test similarity search with filters."""
        # Store test embedding
        self.vector_store._store_embeddings(self.mock_embedding_data)
        
        # Search with document type filter
        results = self.vector_store.similarity_search(
            "test query", 
            top_k=5, 
            filters={'document_type': 'research_paper'}
        )
        
        assert len(results) <= 1
        
        # Search with non-matching filter
        results_filtered_out = self.vector_store.similarity_search(
            "test query", 
            top_k=5, 
            filters={'document_type': 'clinical_trial'}
        )
        
        assert len(results_filtered_out) == 0
    
    def test_document_removal(self):
        """Test removing document embeddings."""
        # Store embedding
        self.vector_store._store_embeddings(self.mock_embedding_data)
        assert len(self.vector_store._embeddings_store) == 1
        
        # Remove embedding
        removed = self.vector_store.remove_document_embedding('doc_001')
        assert removed is True
        assert len(self.vector_store._embeddings_store) == 0
        assert 'doc_001' not in self.vector_store._document_index
        
        # Try to remove non-existent document
        removed_again = self.vector_store.remove_document_embedding('doc_001')
        assert removed_again is False
    
    def test_store_statistics(self):
        """Test vector store statistics."""
        # Empty store
        stats = self.vector_store.get_store_stats()
        assert stats['total_embeddings'] == 0
        assert stats['total_documents'] == 0
        
        # Store embedding
        self.vector_store._store_embeddings(self.mock_embedding_data)
        
        # Updated stats
        updated_stats = self.vector_store.get_store_stats()
        assert updated_stats['total_embeddings'] == 1
        assert updated_stats['total_documents'] == 1
        assert updated_stats['vector_dimension'] == 1536


def test_create_vector_store():
    """Test factory function for creating vector store."""
    with patch('src.ingestion.vector_store.EmbeddingGenerator') as mock_generator_class:
        mock_generator_class.return_value = Mock()
        vector_store = create_vector_store(api_key="test_key")
        assert isinstance(vector_store, PathwayVectorStore)


if __name__ == "__main__":
    pytest.main([__file__])