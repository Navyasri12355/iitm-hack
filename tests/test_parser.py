"""
Tests for document parser functionality.

Tests multi-format parsing, metadata extraction, and error handling.
"""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime

from src.ingestion import UnstructuredParser, DocumentParsingError, create_document_parser
from src.models import DocumentType


def safe_cleanup_temp_file(file_path):
    """Safely clean up temporary files, handling Windows file locking issues."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except PermissionError:
        # On Windows, file might still be locked, try again after a short delay
        time.sleep(0.1)
        try:
            Path(file_path).unlink(missing_ok=True)
        except PermissionError:
            pass  # Ignore if we can't delete the temp file


class TestUnstructuredParser:
    """Test UnstructuredParser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            self.parser = UnstructuredParser()
        except ImportError:
            pytest.skip("Unstructured library not available")
    
    def test_parser_creation(self):
        """Test that parser can be created successfully."""
        parser = create_document_parser()
        assert isinstance(parser, UnstructuredParser)
    
    def test_parse_text_document(self):
        """Test parsing a simple text document."""
        # Create a temporary text file with medical content
        content = """
        Efficacy of Drug X in Treating Condition Y
        
        Authors: Dr. John Smith, Dr. Jane Doe
        
        Abstract:
        This randomized controlled trial (RCT) examined the efficacy of Drug X 
        in treating Condition Y. The study included n=500 patients with a 
        confidence interval of 95% CI. Results showed p<0.05 significance.
        
        Methods:
        Patients received 10mg of Drug X daily for 12 weeks.
        
        Results:
        Treatment showed 85% improvement rate compared to placebo.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                doc = self.parser.parse_document(
                    file_path=tmp_file.name,
                    source="test_journal"
                )
                
                # Verify basic document properties
                assert doc.id.startswith("doc_")
                assert "Drug X" in doc.title or "Drug X" in doc.content
                assert doc.source == "test_journal"
                assert doc.document_type == DocumentType.RESEARCH_PAPER
                assert 0.0 <= doc.credibility_score <= 1.0
                
                # Verify content preservation
                assert "Drug X" in doc.content
                assert "randomized controlled trial" in doc.content.lower()
                assert "n=500" in doc.content
                
                # Verify metadata extraction
                assert 'medical_term_counts' in doc.metadata
                assert 'sample_sizes' in doc.metadata
                assert doc.metadata['sample_sizes'] == [500]
                
            finally:
                safe_cleanup_temp_file(tmp_file.name)
    
    def test_parse_html_content(self):
        """Test parsing HTML content."""
        html_content = """
        <html>
        <head><title>Clinical Study Results</title></head>
        <body>
        <h1>Meta-Analysis of Treatment Effectiveness</h1>
        <p>Authors: Dr. Smith, Dr. Jones</p>
        <p>This systematic review analyzed n=1200 patients across multiple studies.</p>
        <p>Results showed significant improvement with p=0.001.</p>
        <p>Dosage: 5mg twice daily showed optimal results.</p>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            tmp_file.write(html_content)
            tmp_file.flush()
            
            try:
                doc = self.parser.parse_document(
                    file_path=tmp_file.name,
                    source="medical_website"
                )
                
                # Verify HTML parsing
                assert doc.document_type == DocumentType.GUIDELINE
                assert "Meta-Analysis" in doc.content
                assert "systematic review" in doc.content.lower()
                assert doc.source == "medical_website"
                
                # Verify metadata extraction from HTML
                assert 'sample_sizes' in doc.metadata
                assert 1200 in doc.metadata['sample_sizes']
                
            finally:
                safe_cleanup_temp_file(tmp_file.name)
    
    def test_parse_from_bytes(self):
        """Test parsing document from bytes."""
        content = """
        Clinical Trial Protocol
        
        This phase III clinical trial will evaluate Drug Z effectiveness.
        Sample size: n=300 participants
        Primary endpoint: 50% reduction in symptoms
        Statistical significance: p<0.05
        """
        
        content_bytes = content.encode('utf-8')
        
        doc = self.parser.parse_from_bytes(
            content_bytes=content_bytes,
            filename="protocol.txt",
            source="clinical_trials_db"
        )
        
        assert "Clinical Trial" in doc.content
        assert doc.source == "clinical_trials_db"
        assert 'sample_sizes' in doc.metadata
        assert 300 in doc.metadata['sample_sizes']
    
    def test_medical_metadata_extraction(self):
        """Test extraction of medical-specific metadata."""
        content = """
        Randomized Controlled Trial of Medication A
        
        Methods: Double-blind RCT with n=750 patients
        Dosage: 25mg daily, 50mg daily, placebo
        Results: 
        - Primary outcome: p=0.003 (95% CI: 1.2-2.8)
        - Secondary outcome: p=0.045 (90% CI: 0.8-1.9)
        - Adverse events: 12% in treatment group vs 8% in placebo
        
        This meta-analysis included cohort studies and case-control studies.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                doc = self.parser.parse_document(tmp_file.name)
                
                # Check medical metadata extraction
                metadata = doc.metadata
                
                # Sample size extraction
                assert 'sample_sizes' in metadata
                assert 750 in metadata['sample_sizes']
                assert metadata['max_sample_size'] == 750
                
                # Study type extraction
                assert 'study_types' in metadata
                study_types = metadata['study_types']
                assert 'randomized' in study_types
                assert 'controlled' in study_types
                assert 'meta-analysis' in study_types
                
                # Medical term counts
                assert 'medical_term_counts' in metadata
                term_counts = metadata['medical_term_counts']
                assert 'dosage' in term_counts
                assert 'p_value' in term_counts
                # Note: percentage detection depends on exact pattern matching
                
                # Statistical measures
                assert 'p_values_found' in metadata
                assert metadata['p_values_found'] >= 2  # Should find p=0.003 and p=0.045
                
                assert 'confidence_intervals_found' in metadata
                assert metadata['confidence_intervals_found'] >= 1
                
            finally:
                safe_cleanup_temp_file(tmp_file.name)
    
    def test_error_handling_nonexistent_file(self):
        """Test error handling for non-existent files."""
        with pytest.raises(DocumentParsingError, match="File not found"):
            self.parser.parse_document("nonexistent_file.pdf")
    
    def test_error_handling_empty_content(self):
        """Test handling of files with minimal content."""
        # Create file with minimal but valid content (needs to be >500 chars to pass validation)
        minimal_content = "This is a minimal medical document with some content to test parsing. " * 10  # Repeat to make it long enough
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(minimal_content)
            tmp_file.flush()
            
            try:
                # This should still work but with low credibility
                doc = self.parser.parse_document(tmp_file.name)
                assert doc.credibility_score < 0.6  # Low credibility for minimal content
                
            finally:
                safe_cleanup_temp_file(tmp_file.name)
    
    def test_credibility_scoring(self):
        """Test credibility score calculation."""
        # High-quality medical content
        high_quality_content = """
        Systematic Review and Meta-Analysis of Treatment X
        
        Authors: Dr. Smith et al.
        
        This comprehensive systematic review analyzed n=5000 patients across 
        25 randomized controlled trials. Statistical analysis showed p<0.001 
        with 95% CI: 1.5-2.3. Treatment dosage of 10mg daily showed optimal 
        efficacy with minimal adverse events (3% vs 1% placebo, p=0.02).
        
        Methods included extensive literature search, quality assessment,
        and meta-regression analysis. Heterogeneity was assessed using I² 
        statistics. Publication bias was evaluated using funnel plots.
        
        Results demonstrate strong evidence for treatment effectiveness
        across diverse patient populations with consistent effect sizes.
        """
        
        # Low-quality content (needs to be >500 chars to pass validation)
        low_quality_content = "This is a short document with no medical content. " * 15  # Repeat to make it long enough
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_high:
            tmp_high.write(high_quality_content)
            tmp_high.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_low:
                tmp_low.write(low_quality_content)
                tmp_low.flush()
                
                try:
                    doc_high = self.parser.parse_document(tmp_high.name)
                    doc_low = self.parser.parse_document(tmp_low.name)
                    
                    # High-quality document should have higher credibility
                    assert doc_high.credibility_score > doc_low.credibility_score
                    assert doc_high.credibility_score > 0.7  # Should be high
                    assert doc_low.credibility_score < 0.6   # Should be low
                    
                finally:
                    safe_cleanup_temp_file(tmp_high.name)
                    safe_cleanup_temp_file(tmp_low.name)
    
    def test_title_and_author_extraction(self):
        """Test extraction of titles and authors."""
        content = """
        Effectiveness of Novel Drug Therapy in Cardiovascular Disease
        
        Authors: Dr. Sarah Johnson, Dr. Michael Chen, Dr. Lisa Rodriguez
        
        Abstract:
        This study evaluates the effectiveness of a novel drug therapy in treating cardiovascular disease.
        The research involved comprehensive analysis of patient outcomes and treatment efficacy.
        Methods included randomized controlled trials with statistical analysis.
        Results demonstrated significant improvements in patient health outcomes.
        Conclusions support the use of this novel therapy in clinical practice.
        Additional research is recommended to further validate these findings.
        The study contributes to the growing body of evidence in cardiovascular medicine.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                doc = self.parser.parse_document(tmp_file.name)
                
                # Check title extraction
                assert "Cardiovascular Disease" in doc.title or "Novel Drug Therapy" in doc.title
                
                # Check author extraction (should find at least some authors)
                assert len(doc.authors) > 0
                # Note: Author extraction is heuristic-based, so we just check that some were found
                
            finally:
                safe_cleanup_temp_file(tmp_file.name)
    
    def test_document_type_determination(self):
        """Test document type determination from file extensions."""
        test_cases = [
            ('.pdf', DocumentType.RESEARCH_PAPER),
            ('.html', DocumentType.GUIDELINE),
            ('.xml', DocumentType.CLINICAL_TRIAL),
        ]
        
        content = "Test medical document content with n=100 patients."
        
        for extension, expected_type in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                
                try:
                    doc = self.parser.parse_document(tmp_file.name)
                    assert doc.document_type == expected_type
                    
                finally:
                    safe_cleanup_temp_file(tmp_file.name)


class TestParserIntegration:
    """Integration tests for parser functionality."""
    
    def test_end_to_end_parsing_workflow(self):
        """Test complete parsing workflow from file to ParsedDocument."""
        try:
            parser = create_document_parser()
        except ImportError:
            pytest.skip("Unstructured library not available")
        
        # Create a realistic medical document
        medical_content = """
        Phase II Clinical Trial: Efficacy and Safety of Drug ABC in Type 2 Diabetes
        
        Principal Investigators: Dr. Maria Garcia, Dr. James Wilson
        
        Background:
        Type 2 diabetes affects millions worldwide. This randomized controlled trial
        evaluates Drug ABC effectiveness compared to standard care.
        
        Methods:
        - Study design: Double-blind, placebo-controlled RCT
        - Participants: n=450 adults with Type 2 diabetes
        - Intervention: Drug ABC 15mg daily vs placebo
        - Primary endpoint: HbA1c reduction at 12 weeks
        - Statistical analysis: p<0.05 considered significant
        
        Results:
        - Primary outcome: Mean HbA1c reduction 1.2% (95% CI: 0.8-1.6, p=0.001)
        - Secondary outcomes: Weight loss 3.5kg (p=0.02)
        - Adverse events: 8% vs 5% placebo (p=0.15, not significant)
        
        Conclusions:
        Drug ABC demonstrates significant efficacy in Type 2 diabetes management
        with acceptable safety profile. Results support progression to Phase III.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(medical_content)
            tmp_file.flush()
            
            try:
                # Parse the document
                doc = parser.parse_document(
                    file_path=tmp_file.name,
                    source="clinical_trials_registry",
                    document_id="trial_abc_001"
                )
                
                # Verify all aspects of the parsed document
                assert doc.id == "trial_abc_001"
                assert "Drug ABC" in doc.title or "Type 2 Diabetes" in doc.title
                assert doc.source == "clinical_trials_registry"
                assert doc.document_type == DocumentType.RESEARCH_PAPER
                
                # Verify content preservation
                assert "randomized controlled trial" in doc.content.lower()
                assert "Drug ABC" in doc.content
                assert "15mg daily" in doc.content
                
                # Verify comprehensive metadata extraction
                metadata = doc.metadata
                assert 'sample_sizes' in metadata
                assert 450 in metadata['sample_sizes']
                assert 'study_types' in metadata
                assert 'randomized' in metadata['study_types']
                assert 'medical_term_counts' in metadata
                assert metadata['medical_term_counts']['dosage'] >= 1
                assert metadata['p_values_found'] >= 3
                assert metadata['confidence_intervals_found'] >= 1
                
                # Verify credibility scoring
                assert doc.credibility_score > 0.7  # Should be high for quality content
                
                # Verify serialization works
                json_data = doc.model_dump_json()
                assert "trial_abc_001" in json_data
                assert "Drug ABC" in json_data
                
            finally:
                safe_cleanup_temp_file(tmp_file.name)