#!/usr/bin/env python3
"""
Clinical Evidence Copilot Demo Setup Script

This script helps set up and run the demonstration by:
1. Checking system requirements
2. Starting necessary services
3. Providing clear instructions
4. Running the demonstration

Usage:
    python setup_demo.py
"""

import subprocess
import sys
import time
import requests
from pathlib import Path
import json

class DemoSetup:
    def __init__(self):
        self.api_port = 8001
        self.frontend_port = 8080
        self.services = {}
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("🔍 Checking demo requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check required files
        required_files = [
            "demo_script.py",
            "demo_scenarios.json", 
            "run_demo.py",
            "mock_api_server.py",
            "simple_server.py",
            "static/index.html"
        ]
        
        for file in required_files:
            if Path(file).exists():
                print(f"✅ {file}")
            else:
                print(f"❌ Missing: {file}")
                return False
        
        # Check required packages
        required_packages = ["requests", "websockets", "fastapi", "uvicorn"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ Missing package: {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\nInstall missing packages with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def start_services(self):
        """Start the required services"""
        print("\n🚀 Starting services...")
        
        # Start mock API server
        try:
            print("Starting mock API server on port 8001...")
            api_process = subprocess.Popen([
                sys.executable, "mock_api_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if API is responding
            try:
                response = requests.get(f"http://localhost:{self.api_port}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Mock API server running")
                    self.services['api'] = api_process
                else:
                    print(f"❌ API server not responding properly: {response.status_code}")
                    api_process.terminate()
                    return False
            except requests.exceptions.RequestException as e:
                print(f"❌ API server connection failed: {e}")
                api_process.terminate()
                return False
                
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False
        
        # Start frontend server
        try:
            print("Starting frontend server on port 8080...")
            frontend_process = subprocess.Popen([
                sys.executable, "simple_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if frontend is responding
            try:
                response = requests.get(f"http://localhost:{self.frontend_port}", timeout=5)
                if response.status_code == 200:
                    print("✅ Frontend server running")
                    self.services['frontend'] = frontend_process
                else:
                    print(f"❌ Frontend server not responding: {response.status_code}")
                    frontend_process.terminate()
                    return False
            except requests.exceptions.RequestException as e:
                print(f"❌ Frontend server connection failed: {e}")
                frontend_process.terminate()
                return False
                
        except Exception as e:
            print(f"❌ Failed to start frontend server: {e}")
            return False
        
        return True
    
    def show_demo_options(self):
        """Show available demonstration options"""
        print("\n🎯 Demo Options Available:")
        print("=" * 50)
        
        options = [
            {
                "name": "Interactive Demo",
                "command": "python run_demo.py --mode interactive",
                "description": "Guided walkthrough with explanations at each step",
                "duration": "15-20 minutes",
                "best_for": "First-time users, presentations"
            },
            {
                "name": "Automated Demo", 
                "command": "python run_demo.py --mode automated",
                "description": "Complete demonstration running automatically",
                "duration": "10-15 minutes",
                "best_for": "Quick overview, unattended running"
            },
            {
                "name": "Scenario Testing",
                "command": "python run_demo.py --mode scenario --scenario contradiction_detection",
                "description": "Test specific capabilities (contradiction_detection, agentic_reasoning, etc.)",
                "duration": "3-5 minutes per scenario",
                "best_for": "Focused testing, development"
            },
            {
                "name": "Performance Benchmark",
                "command": "python run_demo.py --mode benchmark", 
                "description": "Test system performance and response times",
                "duration": "5-10 minutes",
                "best_for": "Performance validation, system testing"
            },
            {
                "name": "Web Interface",
                "command": "Open http://localhost:8080 in browser",
                "description": "Interactive web interface for manual testing",
                "duration": "As needed",
                "best_for": "Manual exploration, UI testing"
            }
        ]
        
        for i, option in enumerate(options, 1):
            print(f"{i}. {option['name']}")
            print(f"   Command: {option['command']}")
            print(f"   Description: {option['description']}")
            print(f"   Duration: {option['duration']}")
            print(f"   Best for: {option['best_for']}")
            print()
    
    def run_interactive_selection(self):
        """Let user select and run a demo option"""
        while True:
            try:
                choice = input("Select demo option (1-5) or 'q' to quit: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == '1':
                    print("\n🎯 Starting Interactive Demo...")
                    subprocess.run([sys.executable, "run_demo.py", "--mode", "interactive"])
                elif choice == '2':
                    print("\n🤖 Starting Automated Demo...")
                    subprocess.run([sys.executable, "run_demo.py", "--mode", "automated"])
                elif choice == '3':
                    scenario = input("Enter scenario name (or press Enter for list): ").strip()
                    if not scenario:
                        print("Available scenarios: contradiction_detection, agentic_reasoning, basic_queries, live_ingestion")
                        continue
                    print(f"\n🎯 Starting Scenario: {scenario}")
                    subprocess.run([sys.executable, "run_demo.py", "--mode", "scenario", "--scenario", scenario])
                elif choice == '4':
                    print("\n📊 Starting Performance Benchmark...")
                    subprocess.run([sys.executable, "run_demo.py", "--mode", "benchmark"])
                elif choice == '5':
                    print(f"\n🌐 Web interface available at: http://localhost:{self.frontend_port}")
                    print("Open this URL in your browser to use the interactive interface.")
                    input("Press Enter when done with web interface...")
                else:
                    print("Invalid choice. Please select 1-5 or 'q'.")
                    
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"Error running demo: {e}")
    
    def cleanup(self):
        """Clean up running services"""
        print("\n🧹 Cleaning up services...")
        
        for service_name, process in self.services.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {service_name} service")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  Force killed {service_name} service")
            except Exception as e:
                print(f"❌ Error stopping {service_name}: {e}")
    
    def run(self):
        """Run the complete demo setup process"""
        print("🏥 Clinical Evidence Copilot Demo Setup")
        print("=" * 60)
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Requirements check failed. Please fix the issues above.")
            return False
        
        # Start services
        if not self.start_services():
            print("\n❌ Failed to start services. Check the errors above.")
            self.cleanup()
            return False
        
        print("\n✅ All services started successfully!")
        
        # Show demo options
        self.show_demo_options()
        
        # Let user select demo
        try:
            self.run_interactive_selection()
        finally:
            self.cleanup()
        
        print("\n👋 Demo setup complete. Thank you!")
        return True

def main():
    setup = DemoSetup()
    
    try:
        setup.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        setup.cleanup()
    except Exception as e:
        print(f"\n\n❌ Setup error: {e}")
        setup.cleanup()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()