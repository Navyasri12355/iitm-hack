"""
Vector store implementation for clinical evidence copilot.

This module implements vector embedding generation and storage using OpenAI embeddings
with Pathway for real-time indexing and similarity search.

Validates Requirements 2.4, 2.5:
- Set up vector embedding generation using OpenAI embeddings
- Create Pathway vector store for real-time indexing
- Maintain system availability during ingestion
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False

from ..config import get_settings
from ..models.core import ParsedDocument, Evidence

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates vector embeddings for medical documents using OpenAI's embedding API.
    
    Handles text chunking, embedding generation, and error recovery for
    robust document processing.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
            model: Embedding model to use (uses config if not provided)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.settings = get_settings()
        self.api_key = api_key or self.settings.openai_api_key
        self.model = model or self.settings.embedding_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Configuration
        self.max_chunk_size = 8000  # OpenAI embedding limit
        self.overlap_size = 200     # Overlap between chunks
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")
    
    def generate_embeddings(self, document: ParsedDocument) -> Dict[str, Any]:
        """
        Generate embeddings for a document.
        
        Args:
            document: ParsedDocument to generate embeddings for
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        try:
            # Split document into chunks
            chunks = self._split_into_chunks(document.content)
            
            # Generate embeddings for each chunk
            chunk_embeddings = []
            chunk_texts = []
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self._generate_single_embedding(chunk)
                    chunk_embeddings.append(embedding)
                    chunk_texts.append(chunk)
                    
                    logger.debug(f"Generated embedding for chunk {i+1}/{len(chunks)} of document {document.id}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {i+1} of document {document.id}: {e}")
                    continue
            
            if not chunk_embeddings:
                raise ValueError(f"Failed to generate any embeddings for document {document.id}")
            
            # Create document-level embedding (average of chunk embeddings)
            document_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            
            # Prepare embedding data
            embedding_data = {
                'document_id': document.id,
                'document_embedding': document_embedding,
                'chunk_embeddings': chunk_embeddings,
                'chunk_texts': chunk_texts,
                'embedding_model': self.model,
                'embedding_dimension': len(document_embedding),
                'chunk_count': len(chunk_embeddings),
                'generated_at': datetime.now().isoformat(),
                'document_metadata': {
                    'title': document.title,
                    'authors': document.authors,
                    'document_type': document.document_type.value,
                    'publication_date': document.publication_date.isoformat(),
                    'credibility_score': document.credibility_score
                }
            }
            
            logger.info(f"Generated embeddings for document {document.id}: {len(chunk_embeddings)} chunks")
            return embedding_data
            
        except Exception as e:
            logger.error(f"Error generating embeddings for document {document.id}: {e}")
            raise
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for embedding generation.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the overlap region
                search_start = max(start, end - self.overlap_size)
                sentence_end = text.rfind('.', search_start, end)
                
                if sentence_end > search_start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap_size if end < len(text) else end
        
        return chunks
    
    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.replace('\n', ' ')  # Clean newlines
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}")
            raise


class PathwayVectorStore:
    """
    Pathway-based vector store for real-time document indexing and similarity search.
    
    Provides real-time vector storage, similarity search, and index management
    for medical document embeddings.
    """
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the Pathway vector store.
        
        Args:
            embedding_generator: EmbeddingGenerator instance (creates one if not provided)
        """
        if not PATHWAY_AVAILABLE:
            raise ImportError("Pathway library not available. Install with: pip install pathway")
        
        self.settings = get_settings()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Vector store configuration
        self.vector_dimension = self.settings.vector_dimension
        self.similarity_threshold = self.settings.similarity_threshold
        
        # In-memory storage for embeddings (in production, this would be persistent)
        self._embeddings_store: Dict[str, Dict[str, Any]] = {}
        self._document_index: Dict[str, str] = {}  # doc_id -> embedding_id mapping
        
        logger.info(f"Initialized PathwayVectorStore with dimension: {self.vector_dimension}")
    
    def create_embedding_table(self, documents_table: pw.Table) -> pw.Table:
        """
        Create Pathway table for embedding generation and storage.
        
        Args:
            documents_table: Pathway table with processed documents
            
        Returns:
            Pathway table with embeddings
        """
        def generate_document_embeddings(row):
            """Generate embeddings for a document row."""
            try:
                parsed_doc = row.result['parsed_document']
                if not parsed_doc:
                    return {
                        'document_id': row.result.get('document_id'),
                        'embedding_data': None,
                        'status': 'no_document',
                        'error': 'No parsed document available'
                    }
                
                # Generate embeddings
                embedding_data = self.embedding_generator.generate_embeddings(parsed_doc)
                
                # Store in memory (in production, this would be persistent storage)
                self._store_embeddings(embedding_data)
                
                logger.info(f"Generated and stored embeddings for document {parsed_doc.id}")
                
                return {
                    'document_id': parsed_doc.id,
                    'embedding_data': embedding_data,
                    'status': 'embedded',
                    'embedding_dimension': len(embedding_data['document_embedding']),
                    'chunk_count': embedding_data['chunk_count']
                }
                
            except Exception as e:
                logger.error(f"Error generating embeddings for document: {e}")
                return {
                    'document_id': row.result.get('document_id'),
                    'embedding_data': None,
                    'status': 'embedding_error',
                    'error': str(e)
                }
        
        # Apply embedding generation to documents
        embeddings_table = documents_table.select(
            embedding_result=pw.apply(generate_document_embeddings, pw.this)
        )
        
        # Filter successful embeddings
        successful_embeddings = embeddings_table.filter(
            lambda row: row.embedding_result['status'] == 'embedded'
        )
        
        # Log embedding errors
        embedding_errors = embeddings_table.filter(
            lambda row: row.embedding_result['status'] in ['embedding_error', 'no_document']
        )
        embedding_errors.debug("embedding_errors")
        
        return successful_embeddings
    
    def _store_embeddings(self, embedding_data: Dict[str, Any]) -> None:
        """Store embeddings in the vector store."""
        document_id = embedding_data['document_id']
        embedding_id = f"emb_{document_id}_{int(datetime.now().timestamp())}"
        
        self._embeddings_store[embedding_id] = embedding_data
        self._document_index[document_id] = embedding_id
        
        logger.debug(f"Stored embeddings for document {document_id} with ID {embedding_id}")
    
    def similarity_search(
        self, 
        query_text: str, 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform similarity search against stored embeddings.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            filters: Optional filters to apply
            
        Returns:
            List of (document_id, similarity_score, metadata) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator._generate_single_embedding(query_text)
            
            # Calculate similarities
            similarities = []
            
            for embedding_id, embedding_data in self._embeddings_store.items():
                doc_embedding = embedding_data['document_embedding']
                
                # Apply filters if provided
                if filters and not self._apply_filters(embedding_data, filters):
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= self.similarity_threshold:
                    similarities.append((
                        embedding_data['document_id'],
                        similarity,
                        embedding_data['document_metadata']
                    ))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def _apply_filters(self, embedding_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to embedding data."""
        metadata = embedding_data.get('document_metadata', {})
        
        for key, value in filters.items():
            if key == 'document_type' and metadata.get('document_type') != value:
                return False
            elif key == 'min_credibility' and metadata.get('credibility_score', 0) < value:
                return False
            elif key == 'max_age_days':
                pub_date_str = metadata.get('publication_date')
                if pub_date_str:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    age_days = (datetime.now() - pub_date).days
                    if age_days > value:
                        return False
        
        return True
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_document_embedding(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding data for a specific document."""
        embedding_id = self._document_index.get(document_id)
        if embedding_id:
            return self._embeddings_store.get(embedding_id)
        return None
    
    def remove_document_embedding(self, document_id: str) -> bool:
        """Remove embedding for a document."""
        embedding_id = self._document_index.get(document_id)
        if embedding_id:
            del self._embeddings_store[embedding_id]
            del self._document_index[document_id]
            logger.info(f"Removed embeddings for document {document_id}")
            return True
        return False
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_embeddings': len(self._embeddings_store),
            'total_documents': len(self._document_index),
            'vector_dimension': self.vector_dimension,
            'similarity_threshold': self.similarity_threshold,
            'embedding_model': self.embedding_generator.model
        }


def create_vector_store(api_key: Optional[str] = None) -> PathwayVectorStore:
    """
    Factory function to create a PathwayVectorStore instance.
    
    Args:
        api_key: OpenAI API key (uses config if not provided)
        
    Returns:
        Configured PathwayVectorStore instance
    """
    embedding_generator = EmbeddingGenerator(api_key=api_key)
    return PathwayVectorStore(embedding_generator=embedding_generator)


# Export main classes
__all__ = ['EmbeddingGenerator', 'PathwayVectorStore', 'create_vector_store']