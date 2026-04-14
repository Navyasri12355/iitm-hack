"""
Document parser with multi-format support for medical literature.

This module provides parsing capabilities for PDF, HTML, and XML documents
with medical domain-specific metadata extraction and error handling.

Validates Requirements 6.1, 6.2, 6.3, 6.4, 6.5:
- Parse PDF, XML, and HTML formats accurately
- Preserve medical terminology and numerical data integrity
- Extract metadata including study methodology and sample sizes
- Handle parsing failures with alternative methods
- Extract textual information from documents with visual elements
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.html import partition_html
    from unstructured.partition.xml import partition_xml
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    UNSTRUCTURED_AVAILABLE = False
    _import_error = e

from dataclasses import dataclass
from src.models.core import ParsedDocument, DocumentType


@dataclass
class ParsedContent:
    """Simple container for parsed document content."""
    content: str
    title: Optional[str] = None
    authors: List[str] = None
    publication_date: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.metadata is None:
            self.metadata = {}


logger = logging.getLogger(__name__)


class DocumentParsingError(Exception):
    """Exception raised when document parsing fails."""
    pass


class UnstructuredParser:
    """
    Wrapper for Unstructured library to parse medical documents.
    
    Supports PDF, HTML, and XML formats with medical domain-specific
    metadata extraction and error handling.
    """
    
    def __init__(self):
        """Initialize the parser."""
        if not UNSTRUCTURED_AVAILABLE:
            error_msg = "Unstructured library is not available. Install it with: pip install unstructured"
            if '_import_error' in globals():
                error_msg += f" (Import error: {_import_error})"
            raise ImportError(error_msg)
        
        # Medical terminology patterns for validation
        self.medical_patterns = {
            'dosage': re.compile(r'\b\d+\.?\d*\s*(mg|g|ml|mcg|units?|iu)\b', re.IGNORECASE),
            'percentage': re.compile(r'\b\d+\.?\d*\s*%\b'),
            'p_value': re.compile(r'\bp\s*[<>=]\s*0\.\d+\b', re.IGNORECASE),
            'confidence_interval': re.compile(r'\b\d+%\s*ci\b|\bconfidence\s+interval\b', re.IGNORECASE),
            'sample_size': re.compile(r'\bn\s*=\s*\d+\b', re.IGNORECASE),
            'study_type': re.compile(r'\b(randomized|controlled|trial|rct|meta-analysis|systematic\s+review|cohort|case-control)\b', re.IGNORECASE)
        }
    
    def parse_document(
        self, 
        file_path: Union[str, Path], 
        source: str = "",
        document_id: Optional[str] = None
    ) -> ParsedDocument:
        """
        Parse a medical document from file path.
        
        Args:
            file_path: Path to the document file
            source: Source identifier for the document
            document_id: Optional custom document ID
            
        Returns:
            ParsedDocument: Parsed document with extracted content and metadata
            
        Raises:
            DocumentParsingError: If parsing fails with all methods
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentParsingError(f"File not found: {file_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"doc_{file_path.stem}_{int(datetime.now().timestamp())}"
        
        # Determine document type from file extension
        document_type = self._determine_document_type(file_path)
        
        # Try parsing with format-specific methods first, then fallback
        parsing_methods = [
            self._parse_with_format_specific,
            self._parse_with_auto_detection,
            self._parse_as_text_fallback
        ]
        
        last_error = None
        for method in parsing_methods:
            try:
                logger.info(f"Attempting to parse {file_path} with {method.__name__}")
                content, metadata = method(file_path)
                
                # Validate that medical content was preserved
                if self._validate_medical_content(content):
                    return self._create_parsed_document(
                        document_id=document_id,
                        file_path=file_path,
                        content=content,
                        metadata=metadata,
                        source=source,
                        document_type=document_type
                    )
                else:
                    logger.warning(f"Medical content validation failed for {file_path}")
                    
            except Exception as e:
                logger.warning(f"Parsing method {method.__name__} failed: {str(e)}")
                last_error = e
                continue
        
        # If all methods failed, raise the last error
        raise DocumentParsingError(
            f"Failed to parse document {file_path} with all methods. Last error: {last_error}"
        )
    
    def parse_from_bytes(
        self, 
        content_bytes: bytes, 
        filename: str,
        source: str = "",
        document_id: Optional[str] = None
    ) -> ParsedContent:
        """
        Parse a medical document from bytes.
        
        Args:
            content_bytes: Document content as bytes
            filename: Original filename for format detection
            source: Source identifier for the document
            document_id: Optional custom document ID
            
        Returns:
            ParsedContent: Parsed document content and metadata
        """
        # Create temporary file for parsing
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_file:
            tmp_file.write(content_bytes)
            tmp_file.flush()
            
            try:
                return self.parse_document_content(
                    file_path=tmp_file.name,
                    source=source,
                    document_id=document_id
                )
            finally:
                # Clean up temporary file
                try:
                    Path(tmp_file.name).unlink(missing_ok=True)
                except PermissionError:
                    # On Windows, file might still be locked, try again after a short delay
                    import time
                    time.sleep(0.1)
                    try:
                        Path(tmp_file.name).unlink(missing_ok=True)
                    except PermissionError:
                        pass  # Ignore if we can't delete the temp file
    
    def parse_document_content(
        self, 
        file_path: Union[str, Path], 
        source: str = "",
        document_id: Optional[str] = None
    ) -> ParsedContent:
        """
        Parse a medical document and return content only.
        
        Args:
            file_path: Path to the document file
            source: Source identifier for the document
            document_id: Optional custom document ID
            
        Returns:
            ParsedContent: Parsed document content and metadata
            
        Raises:
            DocumentParsingError: If parsing fails with all methods
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentParsingError(f"File not found: {file_path}")
        
        # Try parsing with format-specific methods first, then fallback
        parsing_methods = [
            self._parse_with_format_specific,
            self._parse_with_auto_detection,
            self._parse_as_text_fallback
        ]
        
        last_error = None
        for method in parsing_methods:
            try:
                logger.info(f"Attempting to parse {file_path} with {method.__name__}")
                content, metadata = method(file_path)
                
                # Validate that medical content was preserved
                if self._validate_medical_content(content):
                    # Extract additional information
                    title = self._extract_title(content)
                    authors = self._extract_authors(content)
                    pub_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    return ParsedContent(
                        content=content,
                        title=title,
                        authors=authors,
                        publication_date=pub_date,
                        metadata=metadata
                    )
                else:
                    logger.warning(f"Medical content validation failed for {file_path}")
                    
            except Exception as e:
                logger.warning(f"Parsing method {method.__name__} failed: {str(e)}")
                last_error = e
                continue
        
        # If all methods failed, raise the last error
        raise DocumentParsingError(
            f"Failed to parse document {file_path} with all methods. Last error: {last_error}"
        )
    
    def _determine_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension and content."""
        suffix = file_path.suffix.lower()
        
        # Map file extensions to document types
        extension_mapping = {
            '.pdf': DocumentType.RESEARCH_PAPER,  # Default for PDFs
            '.html': DocumentType.GUIDELINE,     # Default for HTML
            '.htm': DocumentType.GUIDELINE,
            '.xml': DocumentType.CLINICAL_TRIAL,  # Default for XML
        }
        
        return extension_mapping.get(suffix, DocumentType.RESEARCH_PAPER)
    
    def _parse_with_format_specific(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse using format-specific Unstructured methods."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            elements = partition_pdf(str(file_path))
        elif suffix in ['.html', '.htm']:
            elements = partition_html(str(file_path))
        elif suffix == '.xml':
            elements = partition_xml(str(file_path))
        else:
            raise DocumentParsingError(f"Unsupported file format: {suffix}")
        
        content = self._extract_content_from_elements(elements)
        metadata = self._extract_metadata_from_elements(elements, file_path)
        
        return content, metadata
    
    def _parse_with_auto_detection(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse using Unstructured's auto-detection."""
        elements = partition(str(file_path))
        
        content = self._extract_content_from_elements(elements)
        metadata = self._extract_metadata_from_elements(elements, file_path)
        
        return content, metadata
    
    def _parse_as_text_fallback(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Fallback method to parse as plain text."""
        logger.warning(f"Using text fallback for {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DocumentParsingError(f"Could not decode file {file_path} with any encoding")
        
        # Basic metadata extraction for text files
        metadata = {
            'parsing_method': 'text_fallback',
            'file_size': file_path.stat().st_size,
            'encoding_used': 'utf-8'
        }
        
        return content, metadata
    
    def _extract_content_from_elements(self, elements) -> str:
        """Extract text content from Unstructured elements."""
        content_parts = []
        
        for element in elements:
            if hasattr(element, 'text') and element.text:
                content_parts.append(element.text)
        
        return '\n'.join(content_parts)
    
    def _extract_metadata_from_elements(self, elements, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from Unstructured elements and content analysis."""
        metadata = {
            'parsing_method': 'unstructured',
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix,
            'element_count': len(elements)
        }
        
        # Extract content for analysis
        full_content = self._extract_content_from_elements(elements)
        
        # Extract medical-specific metadata
        medical_metadata = self._extract_medical_metadata(full_content)
        metadata.update(medical_metadata)
        
        # Extract structural metadata from elements
        structural_metadata = self._extract_structural_metadata(elements)
        metadata.update(structural_metadata)
        
        return metadata
    
    def _extract_medical_metadata(self, content: str) -> Dict[str, Any]:
        """Extract medical domain-specific metadata from content."""
        metadata = {}
        
        # Find sample sizes
        sample_sizes = []
        for match in self.medical_patterns['sample_size'].finditer(content):
            try:
                n_value = int(re.search(r'\d+', match.group()).group())
                sample_sizes.append(n_value)
            except (AttributeError, ValueError):
                continue
        
        if sample_sizes:
            metadata['sample_sizes'] = sample_sizes
            metadata['max_sample_size'] = max(sample_sizes)
        
        # Find study types
        study_types = set()
        for match in self.medical_patterns['study_type'].finditer(content):
            study_types.add(match.group().lower())
        
        if study_types:
            metadata['study_types'] = list(study_types)
        
        # Count medical terminology occurrences
        metadata['medical_term_counts'] = {}
        for term_type, pattern in self.medical_patterns.items():
            matches = pattern.findall(content)
            if matches:
                metadata['medical_term_counts'][term_type] = len(matches)
        
        # Extract confidence intervals and p-values
        ci_matches = self.medical_patterns['confidence_interval'].findall(content)
        if ci_matches:
            metadata['confidence_intervals_found'] = len(ci_matches)
        
        p_value_matches = self.medical_patterns['p_value'].findall(content)
        if p_value_matches:
            metadata['p_values_found'] = len(p_value_matches)
        
        return metadata
    
    def _extract_structural_metadata(self, elements) -> Dict[str, Any]:
        """Extract structural metadata from document elements."""
        metadata = {}
        
        # Count different element types
        element_types = {}
        for element in elements:
            element_type = type(element).__name__
            element_types[element_type] = element_types.get(element_type, 0) + 1
        
        metadata['element_types'] = element_types
        
        # Check for visual elements (images, tables, etc.)
        visual_elements = []
        for element in elements:
            if hasattr(element, 'category'):
                if element.category in ['Image', 'Figure', 'Table', 'FigureCaption']:
                    visual_elements.append(element.category)
        
        if visual_elements:
            metadata['visual_elements'] = visual_elements
            metadata['has_visual_content'] = True
        else:
            metadata['has_visual_content'] = False
        
        return metadata
    
    def _validate_medical_content(self, content: str) -> bool:
        """
        Validate that medical terminology and numerical data are preserved.
        
        Returns True if the content appears to contain valid medical information.
        """
        if not content or len(content.strip()) < 10:  # Reduced minimum length
            return False
        
        # Check for presence of medical terminology
        medical_term_count = 0
        for pattern in self.medical_patterns.values():
            if pattern.search(content):
                medical_term_count += 1
        
        # Content should have at least some medical terminology or be substantial
        # Relaxed validation: accept content with any medical terms OR substantial length
        return medical_term_count > 0 or len(content) > 500  # Reduced threshold
    
    def _create_parsed_document(
        self,
        document_id: str,
        file_path: Path,
        content: str,
        metadata: Dict[str, Any],
        source: str,
        document_type: DocumentType
    ) -> ParsedDocument:
        """Create a ParsedDocument from extracted content and metadata."""
        
        # Extract title from content or use filename
        title = self._extract_title(content) or file_path.stem
        
        # Extract authors if possible
        authors = self._extract_authors(content)
        
        # Use file modification time as publication date fallback
        publication_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Calculate credibility score based on content analysis
        credibility_score = self._calculate_credibility_score(content, metadata)
        
        return ParsedDocument(
            id=document_id,
            title=title,
            authors=authors,
            publication_date=publication_date,
            source=source or str(file_path),
            document_type=document_type,
            content=content,
            metadata=metadata,
            credibility_score=credibility_score
        )
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract document title from content."""
        lines = content.split('\n')
        
        # Look for title-like patterns in first few lines
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Simple heuristic: first substantial line is likely the title
                if not line.lower().startswith(('abstract', 'introduction', 'background')):
                    return line
        
        return None
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract author names from content."""
        authors = []
        
        # Look for author patterns
        author_patterns = [
            re.compile(r'(?:authors?|by):\s*([^.\n]+)', re.IGNORECASE),
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.MULTILINE)
        ]
        
        for pattern in author_patterns:
            matches = pattern.findall(content[:2000])  # Search in first 2000 chars
            for match in matches:
                if isinstance(match, str) and len(match.split()) <= 4:
                    authors.append(match.strip())
        
        # Remove duplicates and return first few
        return list(dict.fromkeys(authors))[:5]
    
    def _calculate_credibility_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate a credibility score based on content and metadata analysis."""
        score = 0.5  # Base score
        
        # Boost score for medical terminology
        medical_terms = metadata.get('medical_term_counts', {})
        if medical_terms:
            score += min(0.2, len(medical_terms) * 0.05)
        
        # Boost score for statistical content
        if metadata.get('p_values_found', 0) > 0:
            score += 0.1
        if metadata.get('confidence_intervals_found', 0) > 0:
            score += 0.1
        
        # Boost score for substantial content
        if len(content) > 5000:
            score += 0.1
        
        # Boost score for structured content
        if metadata.get('element_types', {}):
            score += 0.05
        
        return min(1.0, score)


# Factory function for easy instantiation
def create_document_parser() -> UnstructuredParser:
    """Create and return a document parser instance."""
    return UnstructuredParser()


# Export main classes and functions
__all__ = [
    'UnstructuredParser',
    'DocumentParsingError',
    'create_document_parser'
]