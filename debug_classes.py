#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

print("Debugging class definitions...")

try:
    # Read the file content
    with open('src/reasoning/evidence_retrieval.py', 'r') as f:
        content = f.read()
    
    # Check if class definitions exist in the content
    if 'class EvidenceHierarchyRanker:' in content:
        print("✓ EvidenceHierarchyRanker class definition found in file")
    else:
        print("✗ EvidenceHierarchyRanker class definition NOT found in file")
    
    if 'class MedicalContextualRetriever:' in content:
        print("✓ MedicalContextualRetriever class definition found in file")
    else:
        print("✗ MedicalContextualRetriever class definition NOT found in file")
    
    # Try to execute with globals tracking
    print("\nExecuting with globals tracking...")
    global_vars = {}
    local_vars = {}
    
    # Import dependencies first
    exec("""
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from src.models.core import (
    Evidence, EvidenceLevel, ClinicalQuery, ParsedDocument, 
    Contradiction, DocumentType
)

# Import PathwayVectorStore conditionally to avoid import errors
try:
    from src.ingestion.vector_store import PathwayVectorStore
except ImportError:
    # Create a mock class if PathwayVectorStore is not available
    class PathwayVectorStore:
        def __init__(self, *args, **kwargs):
            pass
        
        def similarity_search(self, query_text: str, top_k: int = 10, filters=None):
            return []

logger = logging.getLogger(__name__)
""", global_vars, local_vars)
    
    print(f"After imports - globals keys: {list(global_vars.keys())}")
    print(f"After imports - locals keys: {list(local_vars.keys())}")
    
    # Now execute the rest of the file
    exec(content, global_vars, local_vars)
    
    print(f"After execution - globals keys: {list(global_vars.keys())}")
    print(f"After execution - locals keys: {list(local_vars.keys())}")
    
    if 'EvidenceHierarchyRanker' in local_vars:
        print("✓ EvidenceHierarchyRanker found in locals after execution")
    if 'MedicalContextualRetriever' in local_vars:
        print("✓ MedicalContextualRetriever found in locals after execution")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Debug completed.")