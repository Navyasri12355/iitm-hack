#!/usr/bin/env python3
"""
Docker development helper for Clinical Evidence Copilot.

This script helps manage Docker containers for Pathway integration
and provides utilities for testing the document monitoring system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def build_containers():
    """Build Docker containers."""
    print("Building Docker containers...")
    run_command("docker-compose build")
    print("✅ Containers built successfully")


def start_services():
    """Start all services."""
    print("Starting services...")
    run_command("docker-compose up -d")
    print("✅ Services started")
    
    # Wait for services to be ready
    print("Waiting for services to be ready...")
    time.sleep(10)
    
    # Check if services are running
    result = run_command("docker-compose ps", check=False)
    print(result.stdout)


def stop_services():
    """Stop all services."""
    print("Stopping services...")
    run_command("docker-compose down")
    print("✅ Services stopped")


def view_logs(service=None):
    """View logs for services."""
    if service:
        run_command(f"docker-compose logs -f {service}")
    else:
        run_command("docker-compose logs -f")


def run_tests():
    """Run tests inside Docker container."""
    print("Running tests in Docker container...")
    run_command("docker-compose exec clinical-evidence-copilot python -m pytest tests/test_pathway_connector.py -v")


def create_sample_documents():
    """Create sample documents for testing."""
    print("Creating sample documents...")
    
    # Ensure documents directory exists
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample medical documents
    samples = [
        {
            "filename": "systematic_review_diabetes.txt",
            "content": """
Systematic Review: Diabetes Management in Primary Care

Authors: Smith, J., Johnson, M., Brown, K.
Publication Date: 2023-06-15
Journal: Journal of Clinical Medicine
DOI: 10.1234/jcm.2023.12345

Abstract:
This systematic review examines the effectiveness of various diabetes management 
strategies in primary care settings. We analyzed 45 randomized controlled trials 
involving 12,000 patients with Type 2 diabetes.

Key Findings:
- Metformin remains first-line therapy with 85% efficacy
- Lifestyle interventions show 70% improvement in HbA1c levels
- Patient education programs reduce complications by 40%

Conclusion:
Comprehensive diabetes management combining medication, lifestyle changes, and 
patient education provides optimal outcomes in primary care settings.
            """
        },
        {
            "filename": "clinical_trial_hypertension.txt",
            "content": """
Randomized Controlled Trial: ACE Inhibitors vs ARBs in Hypertension

Authors: Wilson, A., Davis, R., Miller, S.
Publication Date: 2023-08-20
Journal: Hypertension Research
DOI: 10.1234/hr.2023.67890

Study Design: Double-blind, randomized controlled trial
Sample Size: 2,500 patients
Duration: 24 months

Objective:
To compare the effectiveness of ACE inhibitors versus ARBs in treating 
essential hypertension in adults aged 40-70.

Results:
- ACE inhibitors: 78% achieved target BP (<140/90)
- ARBs: 82% achieved target BP (<140/90)
- Side effects: ACE inhibitors 15%, ARBs 8%

Conclusion:
Both drug classes are effective, but ARBs show slightly better tolerability 
and efficacy in this population.
            """
        },
        {
            "filename": "practice_guideline_cardiology.txt",
            "content": """
Clinical Practice Guideline: Management of Acute Coronary Syndrome

Organization: American College of Cardiology
Publication Date: 2023-09-10
Version: 2023.1

Recommendations:

Class I (Strong Recommendation):
1. Dual antiplatelet therapy for all ACS patients
2. Early invasive strategy within 24 hours for high-risk patients
3. Statin therapy initiated within 24 hours

Class IIa (Moderate Recommendation):
1. Beta-blocker therapy within 24 hours if no contraindications
2. ACE inhibitor therapy for patients with reduced ejection fraction

Class III (Not Recommended):
1. Routine use of glycoprotein IIb/IIIa inhibitors
2. Prophylactic antiarrhythmic therapy

Evidence Level: Based on multiple randomized controlled trials and 
meta-analyses involving over 50,000 patients.
            """
        }
    ]
    
    for sample in samples:
        file_path = docs_dir / sample["filename"]
        file_path.write_text(sample["content"].strip())
        print(f"Created: {file_path}")
    
    print("✅ Sample documents created")


def monitor_documents():
    """Monitor document processing in real-time."""
    print("Monitoring document processing...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        run_command("docker-compose logs -f pathway-worker")
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")


def shell():
    """Open shell in the main container."""
    print("Opening shell in clinical-evidence-copilot container...")
    run_command("docker-compose exec clinical-evidence-copilot /bin/bash")


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("""
Docker Development Helper for Clinical Evidence Copilot

Usage: python docker-dev.py <command>

Commands:
  build       - Build Docker containers
  start       - Start all services
  stop        - Stop all services
  logs        - View logs for all services
  logs <svc>  - View logs for specific service
  test        - Run tests in container
  samples     - Create sample documents
  monitor     - Monitor document processing
  shell       - Open shell in container
  
Examples:
  python docker-dev.py build
  python docker-dev.py start
  python docker-dev.py samples
  python docker-dev.py monitor
        """)
        return
    
    command = sys.argv[1]
    
    if command == "build":
        build_containers()
    elif command == "start":
        start_services()
    elif command == "stop":
        stop_services()
    elif command == "logs":
        service = sys.argv[2] if len(sys.argv) > 2 else None
        view_logs(service)
    elif command == "test":
        run_tests()
    elif command == "samples":
        create_sample_documents()
    elif command == "monitor":
        monitor_documents()
    elif command == "shell":
        shell()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()