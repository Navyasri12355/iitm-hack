"""
Pathway file system connector for real-time document ingestion.

This module implements the Pathway streaming connector that monitors
medical document folders and processes changes in real-time.

Validates Requirements 2.1, 2.2, 2.3:
- Index new documents within 60 seconds
- Re-index modified portions immediately  
- Remove deleted documents immediately
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
import pathway as pw
from datetime import datetime

from ..config import get_settings
from ..models.core import ParsedDocument, DocumentType
from .parser import create_document_parser, DocumentParsingError

logger = logging.getLogger(__name__)


class PathwayDocumentConnector:
    """
    Pathway-based file system connector for medical document monitoring.
    
    Provides real-time document detection, parsing, and change tracking
    for medical literature folders.
    """
    
    def __init__(self, 
                 documents_path: str = None,
                 supported_formats: List[str] = None):
        """
        Initialize the Pathway document connector.
        
        Args:
            documents_path: Path to monitor for documents
            supported_formats: List of supported file extensions
        """
        self.settings = get_settings()
        self.documents_path = documents_path or self.settings.documents_path
        self.supported_formats = supported_formats or self.settings.supported_formats
        self.parser = create_document_parser()
        
        # Ensure documents directory exists
        Path(self.documents_path).mkdir(parents=True, exist_ok=True)
        
        # Track processed documents for change detection
        self._processed_documents: Dict[str, datetime] = {}
        self._document_hashes: Dict[str, str] = {}
        
        logger.info(f"Initialized PathwayDocumentConnector monitoring: {self.documents_path}")
        logger.info(f"Supported formats: {self.supported_formats}")
    
    def _is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported for processing."""
        extension = Path(file_path).suffix.lower().lstrip('.')
        return extension in self.supported_formats
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content to detect changes."""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return ""
    
    def _detect_document_type(self, file_path: str, content: str) -> DocumentType:
        """
        Detect document type based on file path and content.
        
        This is a simplified heuristic - in production, this would use
        more sophisticated classification methods.
        """
        file_name = Path(file_path).name.lower()
        content_lower = content.lower()
        
        # Simple heuristics for document type detection
        if any(term in content_lower for term in ['systematic review', 'meta-analysis']):
            return DocumentType.SYSTEMATIC_REVIEW
        elif any(term in content_lower for term in ['randomized controlled trial', 'rct']):
            return DocumentType.CLINICAL_TRIAL
        elif any(term in content_lower for term in ['guideline', 'recommendation', 'practice guideline']):
            return DocumentType.GUIDELINE
        elif any(term in file_name for term in ['meta', 'systematic']):
            return DocumentType.META_ANALYSIS
        elif any(term in content_lower for term in ['case study', 'case report']):
            return DocumentType.CASE_STUDY
        else:
            return DocumentType.RESEARCH_PAPER
    
    def _calculate_credibility_score(self, parsed_doc: Dict) -> float:
        """
        Calculate credibility score based on document metadata.
        
        This is a simplified scoring system - production would use
        more sophisticated credibility assessment.
        """
        score = 0.5  # Base score
        
        # Boost for peer-reviewed journals
        if parsed_doc.get('journal'):
            score += 0.2
        
        # Boost for DOI presence
        if parsed_doc.get('doi'):
            score += 0.1
        
        # Boost for recent publication
        pub_date = parsed_doc.get('publication_date')
        if pub_date and isinstance(pub_date, datetime):
            years_old = (datetime.now() - pub_date).days / 365
            if years_old < 5:
                score += 0.2
            elif years_old < 10:
                score += 0.1
        
        # Boost for multiple authors
        authors = parsed_doc.get('authors', [])
        if len(authors) > 1:
            score += 0.1
        
        return min(1.0, score)
    
    def create_pathway_table(self) -> pw.Table:
        """
        Create Pathway table for monitoring document folder.
        
        Returns:
            Pathway table configured for file system monitoring
        """
        try:
            logger.info(f"Creating Pathway table for path: {self.documents_path}")
            
            # Ensure the directory exists and is accessible
            if not os.path.exists(self.documents_path):
                logger.warning(f"Documents path does not exist, creating: {self.documents_path}")
                Path(self.documents_path).mkdir(parents=True, exist_ok=True)
            
            # Create file system connector with streaming mode for real-time monitoring
            files_table = pw.io.fs.read(
                path=self.documents_path,
                format="binary",
                mode="streaming",
                with_metadata=True,
                autocommit_duration_ms=1000,  # Commit changes every second for real-time processing
            )
            
            # Filter for supported formats
            supported_files = files_table.filter(
                lambda row: self._is_supported_format(row.path)
            )
            
            logger.info(f"Created Pathway table monitoring {self.documents_path}")
            logger.info(f"Supported formats: {self.supported_formats}")
            return supported_files
            
        except Exception as e:
            logger.error(f"Failed to create Pathway table: {e}")
            raise
    
    def process_document_changes(self, files_table: pw.Table) -> pw.Table:
        """
        Process document changes (new, modified, deleted) in real-time.
        
        Args:
            files_table: Pathway table with file system data
            
        Returns:
            Pathway table with processed document data
        """
        
        def parse_and_validate_document(row):
            """Parse document and create ParsedDocument instance."""
            try:
                file_path = row.path
                file_content = row.data
                
                # Check if file was modified using metadata
                current_time = datetime.now()
                
                logger.info(f"Processing document: {file_path}")
                
                # Parse document content
                parsed_content = self.parser.parse_from_bytes(file_content, file_path)
                
                # Extract metadata
                metadata = {
                    'file_path': file_path,
                    'file_size': len(file_content),
                    'last_modified': current_time,
                    'processing_timestamp': current_time.isoformat()
                }
                
                # Add any additional metadata from parser
                if hasattr(parsed_content, 'metadata') and parsed_content.metadata:
                    metadata.update(parsed_content.metadata)
                
                # Detect document type
                doc_type = self._detect_document_type(file_path, parsed_content.content)
                
                # Generate document ID based on file path and content hash
                import hashlib
                content_hash = hashlib.md5(file_content).hexdigest()
                doc_id = f"doc_{Path(file_path).stem}_{content_hash[:8]}"
                
                # Create ParsedDocument
                parsed_doc = ParsedDocument(
                    id=doc_id,
                    title=parsed_content.title or Path(file_path).stem,
                    authors=parsed_content.authors or [],
                    publication_date=parsed_content.publication_date or current_time,
                    source=file_path,
                    document_type=doc_type,
                    content=parsed_content.content,
                    metadata=metadata,
                    credibility_score=self._calculate_credibility_score(metadata)
                )
                
                # Track processing
                self._processed_documents[file_path] = current_time
                
                logger.info(f"Successfully processed document: {file_path} (ID: {doc_id})")
                return {
                    'document_id': doc_id,
                    'file_path': file_path,
                    'parsed_document': parsed_doc,
                    'processing_time': current_time,
                    'status': 'processed'
                }
                
            except DocumentParsingError as e:
                logger.error(f"Failed to parse document {row.path}: {e}")
                return {
                    'document_id': None,
                    'file_path': row.path,
                    'parsed_document': None,
                    'processing_time': datetime.now(),
                    'status': 'parse_error',
                    'error': str(e)
                }
            except Exception as e:
                logger.error(f"Unexpected error processing document {row.path}: {e}")
                return {
                    'document_id': None,
                    'file_path': row.path,
                    'parsed_document': None,
                    'processing_time': datetime.now(),
                    'status': 'error',
                    'error': str(e)
                }
        
        # Apply document processing with error handling
        processed_docs = files_table.select(
            result=pw.apply(parse_and_validate_document, pw.this)
        )
        
        # Filter out failed processing attempts for the main pipeline
        # but keep them for error tracking
        successful_docs = processed_docs.filter(
            lambda row: row.result['status'] == 'processed'
        )
        
        # Log processing errors separately
        error_docs = processed_docs.filter(
            lambda row: row.result['status'] in ['parse_error', 'error']
        )
        
        # Set up error logging
        error_docs.debug("processing_errors")
        
        return successful_docs
    
    def handle_document_deletions(self, current_files: Set[str]) -> List[str]:
        """
        Detect and handle document deletions.
        
        Args:
            current_files: Set of currently existing file paths
            
        Returns:
            List of deleted file paths
        """
        deleted_files = []
        
        # Find files that were previously processed but no longer exist
        for file_path in list(self._processed_documents.keys()):
            if file_path not in current_files:
                deleted_files.append(file_path)
                
                # Clean up tracking
                del self._processed_documents[file_path]
                if file_path in self._document_hashes:
                    del self._document_hashes[file_path]
                
                logger.info(f"Detected deletion: {file_path}")
        
        return deleted_files
    
    def start_monitoring(self) -> pw.Table:
        """
        Start real-time monitoring of the documents folder.
        
        This method sets up the complete Pathway pipeline for document monitoring,
        processing, and change detection.
        
        Returns:
            Pathway table with processed documents
        """
        logger.info(f"Starting real-time document monitoring for {self.documents_path}")
        logger.info(f"Monitoring formats: {self.supported_formats}")
        
        # Verify directory exists and is readable
        if not os.path.exists(self.documents_path):
            logger.error(f"Documents path does not exist: {self.documents_path}")
            raise FileNotFoundError(f"Documents path not found: {self.documents_path}")
        
        if not os.access(self.documents_path, os.R_OK):
            logger.error(f"Documents path is not readable: {self.documents_path}")
            raise PermissionError(f"Cannot read documents path: {self.documents_path}")
        
        # Create the file system monitoring table
        files_table = self.create_pathway_table()
        
        # Process document changes in real-time
        processed_docs = self.process_document_changes(files_table)
        
        # Set up output handling for processed documents
        processed_docs.debug("processed_document")
        
        logger.info("Document monitoring pipeline started successfully")
        logger.info("Pathway connector is now monitoring for document changes...")
        return processed_docs
    
    def stop_monitoring(self):
        """Stop the document monitoring process."""
        logger.info("Stopping document monitoring")
        # In a real implementation, this would clean up Pathway resources
        # For now, we'll just log the stop action
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about document processing."""
        return {
            'total_processed': len(self._processed_documents),
            'monitored_path': self.documents_path,
            'supported_formats': self.supported_formats,
            'last_activity': max(self._processed_documents.values()) if self._processed_documents else None
        }


def create_pathway_connector(documents_path: str = None, 
                           supported_formats: List[str] = None) -> PathwayDocumentConnector:
    """
    Factory function to create a PathwayDocumentConnector instance.
    
    Args:
        documents_path: Path to monitor for documents
        supported_formats: List of supported file extensions
        
    Returns:
        Configured PathwayDocumentConnector instance
    """
    return PathwayDocumentConnector(
        documents_path=documents_path,
        supported_formats=supported_formats
    )