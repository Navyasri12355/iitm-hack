#!/usr/bin/env python3
"""
Clinical Evidence Copilot Demo Runner

This script provides multiple demonstration modes for the Clinical Evidence Copilot:
- Interactive demo with user prompts
- Automated full demo
- Specific scenario testing
- Performance benchmarking

Usage:
    python run_demo.py --mode interactive
    python run_demo.py --mode automated
    python run_demo.py --mode scenario --scenario contradiction_detection
    python run_demo.py --mode benchmark
"""

import argparse
import asyncio
import json
import time
import sys
from pathlib import Path
from demo_script import ClinicalEvidenceCopilotDemo

class DemoRunner:
    def __init__(self, mode="interactive"):
        self.mode = mode
        self.demo = ClinicalEvidenceCopilotDemo()
        self.scenarios = self._load_scenarios()
        
    def _load_scenarios(self):
        """Load demo scenarios from JSON file"""
        try:
            with open('demo_scenarios.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️  demo_scenarios.json not found, using basic scenarios")
            return {"clinical_queries": [], "document_scenarios": []}
    
    async def run(self, scenario_name=None):
        """Run the demonstration based on mode"""
        print("🏥 Clinical Evidence Copilot Demo Runner")
        print("=" * 60)
        
        if self.mode == "interactive":
            await self._run_interactive_demo()
        elif self.mode == "automated":
            await self._run_automated_demo()
        elif self.mode == "scenario":
            await self._run_scenario_demo(scenario_name)
        elif self.mode == "benchmark":
            await self._run_benchmark_demo()
        else:
            print(f"❌ Unknown mode: {self.mode}")
            return False
            
        return True
    
    async def _run_interactive_demo(self):
        """Run interactive demo with user prompts"""
        print("🎯 Interactive Demo Mode")
        print("This demo will walk you through each capability with explanations.")
        print()
        
        # Check system health first
        if not await self.demo._check_system_health():
            print("❌ System health check failed. Please ensure all services are running.")
            return
        
        # Connect to WebSocket
        await self.demo._connect_websocket()
        
        # Phase 1: Basic Query Processing
        print("\n" + "="*50)
        print("PHASE 1: Basic Clinical Query Processing")
        print("="*50)
        print("We'll start with a straightforward clinical question to show")
        print("how the system provides evidence-backed recommendations.")
        input("\nPress Enter to continue...")
        
        await self.demo._demonstrate_query_processing()
        
        # Phase 2: Live Document Updates
        print("\n" + "="*50)
        print("PHASE 2: Live Document Ingestion & Updates")
        print("="*50)
        print("Now we'll add new medical literature and show how the system")
        print("automatically updates recommendations in real-time.")
        input("\nPress Enter to continue...")
        
        await self.demo._demonstrate_live_updates()
        
        # Phase 3: Contradiction Detection
        print("\n" + "="*50)
        print("PHASE 3: Contradiction Detection")
        print("="*50)
        print("This phase demonstrates how the system identifies and explains")
        print("conflicting evidence, such as the aspirin primary prevention controversy.")
        input("\nPress Enter to continue...")
        
        await self.demo._demonstrate_contradictions()
        
        # Phase 4: Agentic Reasoning
        print("\n" + "="*50)
        print("PHASE 4: Multi-step Agentic Reasoning")
        print("="*50)
        print("We'll show how the system handles complex queries requiring")
        print("multi-step reasoning across multiple medical conditions.")
        input("\nPress Enter to continue...")
        
        await self.demo._demonstrate_agentic_reasoning()
        
        # Phase 5: Recommendation Evolution
        print("\n" + "="*50)
        print("PHASE 5: Recommendation Evolution")
        print("="*50)
        print("Finally, we'll show how recommendations evolve over time")
        print("as new evidence becomes available.")
        input("\nPress Enter to continue...")
        
        await self.demo._demonstrate_recommendation_evolution()
        
        print("\n" + "="*60)
        print("✅ Interactive Demo Complete!")
        print("="*60)
        
        if self.demo.websocket:
            await self.demo.websocket.close()
    
    async def _run_automated_demo(self):
        """Run fully automated demo"""
        print("🤖 Automated Demo Mode")
        print("Running complete demonstration automatically...")
        print()
        
        await self.demo.run_demonstration()
    
    async def _run_scenario_demo(self, scenario_name):
        """Run specific scenario demonstration"""
        print(f"🎯 Scenario Demo Mode: {scenario_name}")
        print()
        
        if not scenario_name:
            print("Available scenarios:")
            for phase in self.scenarios.get('demonstration_flow', []):
                print(f"  - {phase['phase']}: {phase['title']}")
            return
        
        # Find and run specific scenario
        scenario = None
        for phase in self.scenarios.get('demonstration_flow', []):
            if phase['phase'] == scenario_name or phase['title'].lower().replace(' ', '_') == scenario_name:
                scenario = phase
                break
        
        if not scenario:
            print(f"❌ Scenario '{scenario_name}' not found")
            return
        
        print(f"Running scenario: {scenario['title']}")
        print(f"Description: {scenario['description']}")
        print(f"Expected duration: {scenario['expected_duration']}")
        print()
        
        # Check system health
        if not await self.demo._check_system_health():
            return
        
        await self.demo._connect_websocket()
        
        # Run scenario-specific demonstration
        if scenario_name in ['1_basic_queries', 'basic_queries']:
            await self.demo._demonstrate_query_processing()
        elif scenario_name in ['2_live_ingestion', 'live_ingestion']:
            await self.demo._demonstrate_live_updates()
        elif scenario_name in ['4_contradiction_detection', 'contradiction_detection']:
            await self.demo._demonstrate_contradictions()
        elif scenario_name in ['5_agentic_reasoning', 'agentic_reasoning']:
            await self.demo._demonstrate_agentic_reasoning()
        elif scenario_name in ['7_recommendation_evolution', 'recommendation_evolution']:
            await self.demo._demonstrate_recommendation_evolution()
        else:
            print(f"⚠️  Scenario implementation not found for {scenario_name}")
        
        if self.demo.websocket:
            await self.demo.websocket.close()
    
    async def _run_benchmark_demo(self):
        """Run performance benchmark demonstration"""
        print("📊 Benchmark Demo Mode")
        print("Testing system performance across key metrics...")
        print()
        
        # Check system health
        if not await self.demo._check_system_health():
            return
        
        await self.demo._connect_websocket()
        
        benchmarks = {
            "query_response_times": [],
            "document_indexing_times": [],
            "websocket_latency": [],
            "recommendation_accuracy": []
        }
        
        # Test query response times
        print("🔍 Testing query response times...")
        test_queries = [
            "What is the first-line treatment for hypertension?",
            "What are the HbA1c targets for diabetes?",
            "Should I prescribe aspirin for primary prevention?",
            "What is the best diabetes medication for heart failure?",
            "How do I manage chest pain in the emergency department?"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"  Query {i+1}/5: {query[:50]}...")
            start_time = time.time()
            
            try:
                response = self.demo.session.post(f"{self.demo.API_BASE_URL}/query", json={
                    "query_text": query,
                    "clinician_id": "benchmark_test",
                    "urgency_level": "routine"
                }, timeout=60)
                
                response_time = time.time() - start_time
                benchmarks["query_response_times"].append(response_time)
                
                if response.status_code == 200:
                    print(f"    ✅ {response_time:.2f}s")
                else:
                    print(f"    ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                benchmarks["query_response_times"].append(60)  # Timeout
        
        # Test document indexing
        print("\n📄 Testing document indexing times...")
        test_doc = {
            "title": "Benchmark Test Document",
            "authors": ["Dr. Test"],
            "content": "This is a test document for benchmarking indexing performance. " * 100,
            "document_type": "research_paper",
            "source": "Benchmark Journal",
            "publication_date": "2024-12-28T00:00:00"
        }
        
        start_time = time.time()
        try:
            response = self.demo.session.post(f"{self.demo.API_BASE_URL}/documents", json=test_doc)
            indexing_time = time.time() - start_time
            benchmarks["document_indexing_times"].append(indexing_time)
            
            if response.status_code == 200:
                print(f"  ✅ Document indexed in {indexing_time:.2f}s")
            else:
                print(f"  ❌ Indexing failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Indexing error: {e}")
        
        # Calculate and display results
        print("\n📊 BENCHMARK RESULTS")
        print("-" * 40)
        
        if benchmarks["query_response_times"]:
            avg_query_time = sum(benchmarks["query_response_times"]) / len(benchmarks["query_response_times"])
            max_query_time = max(benchmarks["query_response_times"])
            min_query_time = min(benchmarks["query_response_times"])
            
            print(f"Query Response Times:")
            print(f"  Average: {avg_query_time:.2f}s")
            print(f"  Min: {min_query_time:.2f}s")
            print(f"  Max: {max_query_time:.2f}s")
            print(f"  Target: <30s {'✅' if avg_query_time < 30 else '❌'}")
        
        if benchmarks["document_indexing_times"]:
            avg_indexing_time = sum(benchmarks["document_indexing_times"]) / len(benchmarks["document_indexing_times"])
            print(f"\nDocument Indexing:")
            print(f"  Average: {avg_indexing_time:.2f}s")
            print(f"  Target: <60s {'✅' if avg_indexing_time < 60 else '❌'}")
        
        # Success rate
        successful_queries = sum(1 for t in benchmarks["query_response_times"] if t < 60)
        success_rate = (successful_queries / len(benchmarks["query_response_times"])) * 100 if benchmarks["query_response_times"] else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        print(f"Target: >95% {'✅' if success_rate > 95 else '❌'}")
        
        if self.demo.websocket:
            await self.demo.websocket.close()

def main():
    parser = argparse.ArgumentParser(description="Clinical Evidence Copilot Demo Runner")
    parser.add_argument("--mode", choices=["interactive", "automated", "scenario", "benchmark"], 
                       default="interactive", help="Demo mode to run")
    parser.add_argument("--scenario", help="Specific scenario to run (for scenario mode)")
    
    args = parser.parse_args()
    
    # Pre-flight checks
    print("🔍 Pre-flight checks...")
    
    # Check if required files exist
    required_files = ["demo_script.py", "demo_scenarios.json"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file missing: {file}")
            sys.exit(1)
    
    print("✅ All required files present")
    
    # Check if services are likely running
    import requests
    try:
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            print("✅ API server appears to be running")
        else:
            print("⚠️  API server may not be running properly")
    except:
        print("⚠️  API server not detected - make sure it's running on port 8001")
        print("   Start with: python mock_api_server.py")
    
    print()
    
    # Run the demo
    runner = DemoRunner(args.mode)
    
    try:
        asyncio.run(runner.run(args.scenario))
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()