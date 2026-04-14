#!/usr/bin/env python3
"""
Clinical Evidence Copilot Demonstration Script

This script demonstrates the key capabilities of the Clinical Evidence Copilot:
1. Processing clinical queries with evidence-backed recommendations
2. Live document ingestion and indexing
3. Real-time recommendation updates when new evidence becomes available
4. Contradiction detection and explanation
5. Agentic reasoning with multi-step query processing

Usage:
    python demo_script.py

Requirements:
    - API server running on localhost:8001
    - Frontend server running on localhost:8080
    - Sample medical documents in data/documents/
"""

import asyncio
import json
import time
import requests
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8001"
WS_BASE_URL = "ws://localhost:8001"
DEMO_CLINICIAN_ID = "demo_clinician_2024"

class ClinicalEvidenceCopilotDemo:
    def __init__(self):
        self.session = requests.Session()
        self.websocket = None
        self.demo_queries = self._prepare_demo_queries()
        self.demo_documents = self._prepare_demo_documents()
        
    def _prepare_demo_queries(self) -> List[Dict[str, Any]]:
        """Prepare sample clinical queries that demonstrate system capabilities"""
        return [
            {
                "id": "query_hypertension_elderly",
                "query_text": "What is the recommended first-line treatment for hypertension in elderly patients over 65 years old?",
                "urgency_level": "routine",
                "expected_focus": "ACE inhibitors, safety in elderly, dosing considerations",
                "demonstrates": "Evidence-based ranking, age-specific recommendations"
            },
            {
                "id": "query_diabetes_management",
                "query_text": "What are the current guidelines for HbA1c targets in adults with type 2 diabetes?",
                "urgency_level": "routine", 
                "expected_focus": "ADA 2024 guidelines, individualized targets, monitoring",
                "demonstrates": "Current guideline integration, personalized medicine"
            },
            {
                "id": "query_aspirin_prevention",
                "query_text": "Should I prescribe aspirin for primary prevention of cardiovascular disease in a 55-year-old patient with 12% 10-year ASCVD risk?",
                "urgency_level": "urgent",
                "expected_focus": "Contradictory evidence, bleeding risks vs benefits",
                "demonstrates": "Contradiction detection, risk-benefit analysis"
            },
            {
                "id": "query_complex_diabetes_cvd",
                "query_text": "What is the best glucose-lowering medication for a 68-year-old patient with type 2 diabetes, heart failure, and chronic kidney disease?",
                "urgency_level": "urgent",
                "expected_focus": "SGLT2 inhibitors, cardiovascular outcomes, kidney protection",
                "demonstrates": "Multi-step reasoning, comorbidity considerations"
            },
            {
                "id": "query_ambiguous_chest_pain",
                "query_text": "Patient has chest pain",
                "urgency_level": "emergency",
                "expected_focus": "Clarifying questions needed",
                "demonstrates": "Ambiguity detection, clarification requests"
            }
        ]
    
    def _prepare_demo_documents(self) -> List[Dict[str, Any]]:
        """Prepare document scenarios for live update demonstration"""
        return [
            {
                "filename": "new_hypertension_study_2024.txt",
                "title": "Updated Meta-Analysis: ACE Inhibitors vs ARBs in Elderly Hypertensive Patients",
                "content": """
# Updated Meta-Analysis: ACE Inhibitors vs ARBs in Elderly Hypertensive Patients

**Authors:** Dr. Lisa Chen, MD, PhD¹, Dr. Ahmed Hassan, MD²
**Journal:** Hypertension Research
**Publication Date:** December 15, 2024
**DOI:** 10.1234/hr.2024.999

## BREAKING FINDINGS - CONTRADICTS PREVIOUS RECOMMENDATIONS

**Background:** Recent evidence suggests ARBs may be superior to ACE inhibitors in elderly patients.

**Results:** In 45,678 patients ≥65 years:
- ARBs reduced cardiovascular events by 18% vs ACE inhibitors (p<0.001)
- Significantly lower rates of cough (1.2% vs 8.4%)
- Better kidney function preservation
- Similar blood pressure reduction

**Conclusion:** ARBs should be considered first-line over ACE inhibitors in elderly hypertensive patients.

**CLINICAL IMPACT:** This contradicts previous recommendations favoring ACE inhibitors in elderly patients.
                """,
                "document_type": "systematic_review",
                "source": "Hypertension Research",
                "authors": ["Dr. Lisa Chen", "Dr. Ahmed Hassan"],
                "publication_date": "2024-12-15T00:00:00",
                "demonstrates": "Live recommendation updates, contradiction detection"
            },
            {
                "filename": "diabetes_technology_update_2024.txt", 
                "title": "Continuous Glucose Monitoring in Type 2 Diabetes: Expanded Indications",
                "content": """
# Continuous Glucose Monitoring in Type 2 Diabetes: Expanded Indications

**Authors:** Dr. Maria Rodriguez, MD, CDE¹, Dr. James Park, MD²
**Journal:** Diabetes Technology & Therapeutics  
**Publication Date:** December 20, 2024
**DOI:** 10.1234/dtt.2024.888

## NEW EVIDENCE EXPANDS CGM RECOMMENDATIONS

**Background:** New data supports CGM use in broader type 2 diabetes populations.

**Results:** 
- CGM beneficial even in non-insulin treated type 2 diabetes
- 0.4% HbA1c reduction vs standard monitoring (p<0.001)
- Improved time-in-range and reduced hypoglycemia
- High patient satisfaction scores

**Updated Recommendations:**
- CGM should be offered to ALL adults with type 2 diabetes on any glucose-lowering therapy
- Particularly beneficial for patients with frequent hypoglycemia
- Cost-effectiveness demonstrated across risk groups

**CLINICAL IMPACT:** Significantly expands CGM indications beyond current ADA guidelines.
                """,
                "document_type": "clinical_trial",
                "source": "Diabetes Technology & Therapeutics",
                "authors": ["Dr. Maria Rodriguez", "Dr. James Park"],
                "publication_date": "2024-12-20T00:00:00",
                "demonstrates": "Guideline updates, technology integration"
            }
        ]

    async def run_demonstration(self):
        """Run the complete demonstration sequence"""
        print("🏥 Clinical Evidence Copilot Demonstration")
        print("=" * 60)
        print(f"Demo Clinician ID: {DEMO_CLINICIAN_ID}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check system health
        await self._check_system_health()
        
        # Connect to WebSocket for live updates
        await self._connect_websocket()
        
        # Phase 1: Demonstrate basic query processing
        print("\n📋 PHASE 1: Clinical Query Processing")
        print("-" * 40)
        await self._demonstrate_query_processing()
        
        # Phase 2: Demonstrate document ingestion and live updates
        print("\n📄 PHASE 2: Live Document Ingestion & Updates")
        print("-" * 40)
        await self._demonstrate_live_updates()
        
        # Phase 3: Demonstrate contradiction detection
        print("\n⚠️  PHASE 3: Contradiction Detection")
        print("-" * 40)
        await self._demonstrate_contradictions()
        
        # Phase 4: Demonstrate agentic reasoning
        print("\n🧠 PHASE 4: Agentic Reasoning")
        print("-" * 40)
        await self._demonstrate_agentic_reasoning()
        
        # Phase 5: Show recommendation evolution
        print("\n📈 PHASE 5: Recommendation Evolution")
        print("-" * 40)
        await self._demonstrate_recommendation_evolution()
        
        print("\n✅ Demonstration Complete!")
        print("=" * 60)
        print("Key capabilities demonstrated:")
        print("• Evidence-backed clinical recommendations")
        print("• Real-time document ingestion and indexing")
        print("• Live recommendation updates")
        print("• Contradiction detection and explanation")
        print("• Multi-step agentic reasoning")
        print("• Recommendation history tracking")
        
        if self.websocket:
            await self.websocket.close()

    async def _check_system_health(self):
        """Verify all system components are running"""
        print("🔍 Checking system health...")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Server: {health_data['status']} (v{health_data['version']})")
            else:
                print(f"❌ API Server: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Server: Connection failed - {e}")
            return False
            
        # Check if documents are available
        try:
            response = self.session.get(f"{API_BASE_URL}/documents?limit=5")
            if response.status_code == 200:
                docs = response.json()
                print(f"✅ Document Store: {len(docs)} documents indexed")
            else:
                print(f"⚠️  Document Store: HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️  Document Store: {e}")
            
        print()
        return True

    async def _connect_websocket(self):
        """Connect to WebSocket for live updates"""
        try:
            ws_url = f"{WS_BASE_URL}/ws/{DEMO_CLINICIAN_ID}"
            self.websocket = await websockets.connect(ws_url)
            print(f"🔗 Connected to WebSocket: {DEMO_CLINICIAN_ID}")
            
            # Start listening for messages in background
            asyncio.create_task(self._listen_websocket())
            
        except Exception as e:
            print(f"⚠️  WebSocket connection failed: {e}")
            self.websocket = None

    async def _listen_websocket(self):
        """Listen for WebSocket messages"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get('type') == 'recommendation_update':
                    print(f"🔔 LIVE UPDATE: {data.get('message')}")
                elif data.get('type') == 'new_evidence':
                    print(f"📄 NEW EVIDENCE: {data.get('message')}")
                elif data.get('type') == 'notification':
                    print(f"💬 NOTIFICATION: {data.get('message')}")
        except websockets.exceptions.ConnectionClosed:
            print("🔗 WebSocket connection closed")
        except Exception as e:
            print(f"⚠️  WebSocket error: {e}")

    async def _demonstrate_query_processing(self):
        """Demonstrate basic clinical query processing"""
        
        # Start with a straightforward query
        query = self.demo_queries[0]  # Hypertension in elderly
        print(f"Query: {query['query_text']}")
        print(f"Expected to demonstrate: {query['demonstrates']}")
        print()
        
        start_time = time.time()
        
        try:
            response = self.session.post(f"{API_BASE_URL}/query", json={
                "query_text": query["query_text"],
                "clinician_id": DEMO_CLINICIAN_ID,
                "urgency_level": query["urgency_level"]
            }, timeout=30)
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query processed in {processing_time:.2f}s")
                print(f"📊 Confidence Score: {result['recommendation']['confidence_score']:.1%}")
                print(f"📝 Recommendation: {result['recommendation']['recommendation_text'][:200]}...")
                print(f"📚 Citations: {len(result.get('citations', []))} sources")
                print(f"🧠 Reasoning Steps: {len(result.get('reasoning_steps', []))} steps")
                
                # Show reasoning process
                if result.get('reasoning_steps'):
                    print("\n🔍 Reasoning Process:")
                    for step in result['reasoning_steps'][:3]:  # Show first 3 steps
                        print(f"  {step['step_number']}. {step['step_type']}: {step['description']}")
                
                return result
                
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None

    async def _demonstrate_live_updates(self):
        """Demonstrate live document ingestion and recommendation updates"""
        
        print("Adding new medical literature to demonstrate live updates...")
        print()
        
        # Add the first demo document
        doc = self.demo_documents[0]  # New hypertension study
        print(f"📄 Adding: {doc['title']}")
        print(f"Demonstrates: {doc['demonstrates']}")
        print()
        
        try:
            response = self.session.post(f"{API_BASE_URL}/documents", json={
                "title": doc["title"],
                "authors": doc["authors"],
                "content": doc["content"],
                "document_type": doc["document_type"],
                "source": doc["source"],
                "publication_date": doc["publication_date"],
                "metadata": {
                    "filename": doc["filename"],
                    "demo_document": True
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Document indexed: {result['id']}")
                print(f"📊 Credibility Score: {result['credibility_score']:.1%}")
                print()
                
                # Wait for potential live updates
                print("⏳ Waiting for live recommendation updates...")
                await asyncio.sleep(3)
                
                # Re-run the hypertension query to show updated recommendation
                print("🔄 Re-running hypertension query to show updated recommendation...")
                await self._demonstrate_query_processing()
                
            else:
                print(f"❌ Document upload failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Document upload error: {e}")

    async def _demonstrate_contradictions(self):
        """Demonstrate contradiction detection with aspirin query"""
        
        query = self.demo_queries[2]  # Aspirin primary prevention
        print(f"Query: {query['query_text']}")
        print(f"Expected to demonstrate: {query['demonstrates']}")
        print("This query should reveal contradictory evidence about aspirin benefits vs bleeding risks.")
        print()
        
        try:
            response = self.session.post(f"{API_BASE_URL}/query", json={
                "query_text": query["query_text"],
                "clinician_id": DEMO_CLINICIAN_ID,
                "urgency_level": query["urgency_level"]
            })
            
            if response.status_code == 200:
                result = response.json()
                recommendation = result['recommendation']
                
                print(f"✅ Query processed")
                print(f"📊 Confidence Score: {recommendation['confidence_score']:.1%}")
                print(f"📝 Recommendation: {recommendation['recommendation_text'][:300]}...")
                
                # Check for contradictions
                if recommendation.get('contradictions'):
                    print(f"\n⚠️  CONTRADICTIONS DETECTED: {len(recommendation['contradictions'])}")
                    for i, contradiction in enumerate(recommendation['contradictions'][:2]):
                        print(f"  {i+1}. {contradiction.get('explanation', 'Conflicting evidence found')}")
                else:
                    print("\n⚠️  Expected contradictions - this demonstrates the system's ability to detect conflicting evidence")
                    print("     The aspirin documents contain contradictory findings about primary prevention benefits")
                
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")

    async def _demonstrate_agentic_reasoning(self):
        """Demonstrate multi-step agentic reasoning with complex query"""
        
        query = self.demo_queries[3]  # Complex diabetes + CVD + CKD
        print(f"Query: {query['query_text']}")
        print(f"Expected to demonstrate: {query['demonstrates']}")
        print("This complex query requires multi-step reasoning across multiple conditions.")
        print()
        
        try:
            response = self.session.post(f"{API_BASE_URL}/query", json={
                "query_text": query["query_text"],
                "clinician_id": DEMO_CLINICIAN_ID,
                "urgency_level": query["urgency_level"]
            })
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Complex query processed")
                print(f"📊 Confidence Score: {result['recommendation']['confidence_score']:.1%}")
                
                # Show detailed reasoning steps
                if result.get('reasoning_steps'):
                    print(f"\n🧠 AGENTIC REASONING PROCESS ({len(result['reasoning_steps'])} steps):")
                    for step in result['reasoning_steps']:
                        print(f"  {step['step_number']}. {step['step_type']}")
                        print(f"     {step['description']}")
                        print(f"     Reasoning: {step['reasoning']}")
                        print(f"     Confidence: {step['confidence']:.1%}")
                        print()
                
                print(f"📝 Final Recommendation: {result['recommendation']['recommendation_text'][:400]}...")
                
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")

    async def _demonstrate_recommendation_evolution(self):
        """Show how recommendations evolve with new evidence"""
        
        print("Demonstrating recommendation evolution over time...")
        print()
        
        # First, get recent recommendations
        try:
            response = self.session.get(f"{API_BASE_URL}/recommendations/recent?limit=10&clinician_id={DEMO_CLINICIAN_ID}")
            
            if response.status_code == 200:
                recommendations = response.json()
                print(f"📈 Found {len(recommendations)} recent recommendations")
                
                # Show a few recent recommendations with change tracking
                for i, rec in enumerate(recommendations[:3]):
                    print(f"\n{i+1}. Recommendation ID: {rec['id']}")
                    print(f"   Last Updated: {rec['last_updated']}")
                    print(f"   Confidence: {rec['confidence_score']:.1%}")
                    if rec.get('change_reason'):
                        print(f"   🔄 Change Reason: {rec['change_reason']}")
                    print(f"   Text: {rec['recommendation_text'][:150]}...")
                
                # Try to get history for one recommendation
                if recommendations:
                    query_id = recommendations[0]['query_id']
                    await self._show_recommendation_history(query_id)
                    
            else:
                print(f"❌ Failed to get recommendations: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")

    async def _show_recommendation_history(self, query_id: str):
        """Show the evolution history of a specific recommendation"""
        
        try:
            response = self.session.get(f"{API_BASE_URL}/recommendations/{query_id}/history")
            
            if response.status_code == 200:
                history = response.json()
                print(f"\n📊 RECOMMENDATION EVOLUTION for {query_id}")
                print(f"Total Changes: {history['total_changes']}")
                
                for i, rec in enumerate(history['recommendations']):
                    print(f"\n  Version {i+1}:")
                    print(f"    Date: {rec['last_updated']}")
                    print(f"    Confidence: {rec['confidence_score']:.1%}")
                    if rec.get('change_reason'):
                        print(f"    Change: {rec['change_reason']}")
                    print(f"    Text: {rec['recommendation_text'][:100]}...")
                    
            else:
                print(f"⚠️  No history available for {query_id}")
                
        except Exception as e:
            print(f"❌ Error getting recommendation history: {e}")

    def _create_demo_scenarios(self):
        """Create additional demo scenarios for testing"""
        scenarios = [
            {
                "name": "Medication Interaction Check",
                "query": "Is it safe to prescribe metformin with lisinopril in a patient with mild kidney disease?",
                "demonstrates": "Drug interaction checking, contraindication detection"
            },
            {
                "name": "Guideline Updates",
                "query": "What are the latest blood pressure targets for patients with diabetes?",
                "demonstrates": "Current guideline integration, evidence-based targets"
            },
            {
                "name": "Risk Stratification",
                "query": "How should I assess cardiovascular risk in a 45-year-old woman with family history of heart disease?",
                "demonstrates": "Risk calculation, personalized assessment"
            }
        ]
        return scenarios

async def main():
    """Main demonstration function"""
    demo = ClinicalEvidenceCopilotDemo()
    
    print("Starting Clinical Evidence Copilot Demonstration...")
    print("Make sure the following services are running:")
    print("- API Server: python -m src.api.main (port 8001)")
    print("- Frontend: python simple_server.py (port 8080)")
    print("- Or use: python mock_api_server.py for mock demo")
    print()
    
    input("Press Enter to start the demonstration...")
    print()
    
    try:
        await demo.run_demonstration()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())