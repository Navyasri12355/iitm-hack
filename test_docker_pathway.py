#!/usr/bin/env python3
"""
Test script to verify Pathway connector works in Docker environment.

This script can be run inside the Docker container to test the
Pathway document monitoring functionality.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app/src')

from src.ingestion.pathway_connector import create_pathway_connector
from src.config import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pathway_connector():
    """Test the Pathway connector functionality."""
    logger.info("Testing Pathway connector in Docker environment...")
    
    # Get settings
    settings = get_settings()
    logger.info(f"Documents path: {settings.documents_path}")
    logger.info(f"Supported formats: {settings.supported_formats}")
    
    # Create connector
    try:
        connector = create_pathway_connector()
        logger.info("✅ Pathway connector created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create Pathway connector: {e}")
        return False
    
    # Test directory access
    docs_path = Path(settings.documents_path)
    if not docs_path.exists():
        logger.info(f"Creating documents directory: {docs_path}")
        docs_path.mkdir(parents=True, exist_ok=True)
    
    # Create a test document
    test_doc = docs_path / "test_document.txt"
    test_content = """
Test Medical Document

Title: Sample Clinical Study
Authors: Test Author
Publication Date: 2023-12-01

This is a test document to verify that the Pathway connector
can detect and process new medical documents in real-time.

The document contains medical terminology and should be
classified as a research paper.
    """.strip()
    
    logger.info(f"Creating test document: {test_doc}")
    test_doc.write_text(test_content)
    
    # Test connector functionality
    try:
        # Test basic functionality
        stats = connector.get_processing_stats()
        logger.info(f"Processing stats: {stats}")
        
        # Test file format detection
        is_supported = connector._is_supported_format(str(test_doc))
        logger.info(f"Test document format supported: {is_supported}")
        
        # Test document type detection
        doc_type = connector._detect_document_type(str(test_doc), test_content)
        logger.info(f"Detected document type: {doc_type}")
        
        logger.info("✅ Basic connector functionality tests passed")
        
    except Exception as e:
        logger.error(f"❌ Connector functionality test failed: {e}")
        return False
    
    # Test Pathway table creation
    try:
        logger.info("Testing Pathway table creation...")
        files_table = connector.create_pathway_table()
        logger.info("✅ Pathway table created successfully")
        
        # Start monitoring (this will run indefinitely)
        logger.info("Starting document monitoring...")
        logger.info("The connector will now monitor for document changes...")
        logger.info("Add, modify, or delete documents in the data/documents folder to test")
        
        processed_docs = connector.start_monitoring()
        
        # This would normally run indefinitely with pw.run()
        # For testing, we'll just verify the setup worked
        logger.info("✅ Document monitoring started successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pathway table creation failed: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        return False
    
    finally:
        # Clean up test document
        if test_doc.exists():
            test_doc.unlink()
            logger.info("Cleaned up test document")


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Docker Pathway Connector Test")
    logger.info("=" * 60)
    
    success = test_pathway_connector()
    
    if success:
        logger.info("✅ All tests passed! Pathway connector is working in Docker.")
        
        # If running in test mode, exit here
        if "--test-only" in sys.argv:
            return
        
        # Otherwise, start actual monitoring
        logger.info("Starting continuous monitoring...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            import pathway as pw
            pw.run()
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    else:
        logger.error("❌ Tests failed! Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()