#!/usr/bin/env python3
"""Verify that the Clinical Evidence Copilot project is properly set up."""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10+")
        return False

def check_virtual_environment():
    """Check if virtual environment is active."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment - Active")
        return True
    else:
        print("❌ Virtual environment - Not active")
        return False

def check_dependencies():
    """Check if core dependencies are installed."""
    dependencies = [
        ('fastapi', 'fastapi'), 
        ('uvicorn', 'uvicorn'), 
        ('pydantic', 'pydantic'), 
        ('pydantic-settings', 'pydantic_settings'), 
        ('python-dotenv', 'dotenv'), 
        ('openai', 'openai'), 
        ('unstructured', 'unstructured'), 
        ('pytest', 'pytest'), 
        ('hypothesis', 'hypothesis')
    ]
    
    missing = []
    for display_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {display_name} - Installed")
        except ImportError:
            print(f"❌ {display_name} - Missing")
            missing.append(display_name)
    
    return len(missing) == 0

def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = [
        'src', 'src/api', 'src/models', 'src/ingestion', 
        'src/reasoning', 'src/services', 'data', 'data/documents', 
        'data/samples', 'tests'
    ]
    
    required_files = [
        'src/__init__.py', 'src/config.py', 'requirements.txt', 
        '.env.example', 'README.md'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ - Exists")
        else:
            print(f"❌ {dir_path}/ - Missing")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - Exists")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False
    
    return all_good

def check_configuration():
    """Check if configuration can be loaded."""
    try:
        from src.config import get_settings
        settings = get_settings()
        print("✅ Configuration - Loads successfully")
        print(f"   - Documents path: {settings.documents_path}")
        print(f"   - Server: {settings.host}:{settings.port}")
        print(f"   - Timeout: {settings.response_timeout_seconds}s")
        return True
    except Exception as e:
        print(f"❌ Configuration - Error: {e}")
        return False

def check_pathway_compatibility():
    """Check Pathway framework compatibility."""
    import platform
    system = platform.system()
    
    if system in ['Linux', 'Darwin']:  # Darwin is macOS
        print(f"✅ Platform ({system}) - Pathway compatible")
        return True
    elif system == 'Windows':
        print(f"⚠️  Platform ({system}) - Pathway requires WSL/Docker")
        print("   Consider using Windows Subsystem for Linux (WSL)")
        return False
    else:
        print(f"❓ Platform ({system}) - Unknown compatibility")
        return False

def main():
    """Run all setup verification checks."""
    print("🔍 Clinical Evidence Copilot - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Configuration", check_configuration),
        ("Platform Compatibility", check_pathway_compatibility),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        result = check_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All checks passed! Project is ready for development.")
    elif passed >= total - 1:  # Allow one failure (likely Pathway on Windows)
        print("✨ Setup mostly complete! Check warnings above.")
    else:
        print("⚠️  Some issues found. Please address them before continuing.")
    
    print(f"📊 Summary: {passed}/{total} checks passed")
    
    if not check_pathway_compatibility():
        print("\n💡 Next steps for Windows users:")
        print("   1. Install WSL: https://docs.microsoft.com/en-us/windows/wsl/install")
        print("   2. Or use Docker for Pathway components")
        print("   3. Other components can be developed on Windows")

if __name__ == "__main__":
    main()