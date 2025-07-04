#!/usr/bin/env python3
"""
ZKPAS System Setup Script
========================

Simple setup script to prepare the ZKPAS system for running.
This script checks dependencies and sets up the environment.

Usage:
    python setup_zkpas.py              # Check and install dependencies
    python setup_zkpas.py --check      # Check only, don't install
    python setup_zkpas.py --minimal    # Install minimal dependencies only
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

def print_header():
    """Print setup header."""
    print("🚀 ZKPAS System Setup")
    print("=" * 40)
    print("Setting up Zero Knowledge Proof Authentication System")
    print()

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 3.7+ required, found {version.major}.{version.minor}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip - OK")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip not found")
        return False

def install_package(package, check_import=None):
    """Install a package if not already installed."""
    if check_import is None:
        check_import = package
    
    try:
        __import__(check_import)
        print(f"✅ {package} - already installed")
        return True
    except ImportError:
        print(f"📥 Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package} - installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False

def setup_minimal_dependencies():
    """Install minimal dependencies."""
    print("\n📚 Setting up minimal dependencies...")
    
    minimal_deps = [
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("cryptography", "cryptography"),
        ("loguru", "loguru")
    ]
    
    success = True
    for package, import_name in minimal_deps:
        if not install_package(package, import_name):
            success = False
    
    return success

def setup_full_dependencies():
    """Install full dependencies."""
    print("\n📚 Setting up full dependencies...")
    
    # Read requirements from file if it exists
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print(f"📋 Installing from {requirements_file}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                          check=True)
            print("✅ All dependencies installed from requirements.txt")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install from requirements.txt: {e}")
            print("🔄 Falling back to minimal installation...")
            return setup_minimal_dependencies()
    else:
        print("⚠️ requirements.txt not found, installing minimal dependencies...")
        return setup_minimal_dependencies()

def check_optional_dependencies():
    """Check optional dependencies."""
    print("\n🔍 Checking optional dependencies...")
    
    optional_deps = [
        ("tensorflow", "TensorFlow (for advanced LSTM models)"),
        ("torch", "PyTorch (for alternative deep learning)"),
        ("pandas", "Pandas (for data processing)"),
        ("matplotlib", "Matplotlib (for visualization)"),
        ("mlflow", "MLflow (for experiment tracking)")
    ]
    
    available = []
    missing = []
    
    for dep, description in optional_deps:
        try:
            __import__(dep)
            available.append((dep, description))
            print(f"✅ {dep} - available")
        except ImportError:
            missing.append((dep, description))
            print(f"⚪ {dep} - not available (optional)")
    
    if available:
        print("\n🎯 Available optional features:")
        for dep, desc in available:
            print(f"   • {desc}")
    
    if missing:
        print("\n💡 Optional features that could be added:")
        for dep, desc in missing:
            print(f"   • {desc} (pip install {dep})")
    
    return len(available), len(missing)

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "logs",
        "models",
        "results"
    ]
    
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}/")
        else:
            print(f"✅ {directory}/ - exists")

def run_quick_test():
    """Run a quick system test."""
    print("\n🧪 Running quick system test...")
    
    try:
        # Test basic imports
        import numpy as np
        import hashlib
        import time
        
        # Test basic operations
        test_array = np.array([1, 2, 3, 4, 5])
        test_hash = hashlib.sha256(b"test").hexdigest()
        test_time = time.time()
        
        print("✅ Basic operations - OK")
        
        # Test cryptography if available
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec
            
            # Generate a test key pair
            private_key = ec.generate_private_key(ec.SECP256R1())
            public_key = private_key.public_key()
            
            print("✅ Cryptography - OK")
        except ImportError:
            print("⚠️ Cryptography - not available")
        
        # Test sklearn if available
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Create test model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            scaler = StandardScaler()
            
            print("✅ Scikit-learn - OK")
        except ImportError:
            print("⚠️ Scikit-learn - not available")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n🎯 ZKPAS System Ready!")
    print("=" * 40)
    print("To run the ZKPAS system:")
    print()
    print("   python run_zkpas.py                    # Interactive menu")
    print("   python run_zkpas.py --demo basic       # Basic demo")
    print("   python run_zkpas.py --demo lstm        # LSTM demo")
    print("   python run_zkpas.py --demo all         # All demos")
    print("   python run_zkpas.py --health           # Check system health")
    print()
    print("For advanced users:")
    print("   python demos/demo_zkpas_basic.py       # Basic authentication")
    print("   python demos/demo_lstm_system.py       # LSTM prediction")
    print("   python app/lightweight_predictor.py    # Lightweight predictor")
    print()

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="ZKPAS System Setup")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    parser.add_argument("--minimal", action="store_true", help="Install minimal dependencies")
    parser.add_argument("--test", action="store_true", help="Run system test only")
    args = parser.parse_args()
    
    print_header()
    
    # Run test only
    if args.test:
        if run_quick_test():
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
        return
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        print("Please install pip first")
        sys.exit(1)
    
    # Check or install dependencies
    if args.check:
        print("\n🔍 Checking dependencies...")
        check_optional_dependencies()
    else:
        if args.minimal:
            success = setup_minimal_dependencies()
        else:
            success = setup_full_dependencies()
        
        if not success:
            print("\n❌ Failed to install some dependencies")
            print("You can try running with --minimal for basic functionality")
            sys.exit(1)
        
        # Check optional dependencies
        available, missing = check_optional_dependencies()
        
        # Create directories
        create_directories()
        
        # Run quick test
        if run_quick_test():
            print_usage_instructions()
        else:
            print("\n⚠️ Setup completed but some tests failed")
            print("The system may still work with reduced functionality")

if __name__ == "__main__":
    main()