#!/usr/bin/env python3
"""
Quick Setup Script for Real-Time Crime Detection System
Handles first-time installation and basic configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🚀 Real-Time Crime Detection System - Quick Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Ensure Python 3.8+ is available"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("   Virtual environment detected ✅")
        else:
            print("   Installing to system Python")
            
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up project directories...")

    directories = [
        "models",
        "models/synthetic_data",
        "logs",
        "saved_clips"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")

    print("✅ Directory structure ready")
    return True   # <-- FIXED

def test_system():
    """Test core system components"""
    print("\n🧪 Testing system components...")
    
    try:
        # Test imports
        sys.path.append(os.getcwd())
        from src.crime_detector import SimpleCrimeDetector
        print("   Core modules: ✅")
        
        # Test basic model creation
        model = SimpleCrimeDetector()
        print("   AI model: ✅")
        
        # Test OpenCV
        import cv2
        print("   OpenCV: ✅")
        
        print("✅ All components working")
        return True
        
    except Exception as e:
        print(f"❌ Error testing system: {e}")
        return False

def show_usage_instructions():
    """Display usage instructions"""
    print("\n" + "=" * 60)
    print("🎯 Setup Complete! Here's how to use your system:")
    print("=" * 60)
    
    print("\n1️⃣  Quick Demo (5 minutes):")
    print("   python scripts/minimal_training_demo.py")
    print("   python src/main_gui.py")
    
    print("\n2️⃣  Professional Training:")
    print("   python scripts/train_model.py")
    print("   python src/main_gui.py")
    
    print("\n3️⃣  Generate Test Data:")
    print("   python scripts/generate_synthetic_data.py")
    
    print("\n📋 Next Steps:")
    print("   • Read the README.md for detailed instructions")
    print("   • Run the quick demo to see it working")
    print("   • Train with real data for production use")
    
    print("\n🆘 Need Help?")
    print("   • Check README.md for full documentation")
    print("   • Create GitHub issues for problems")
    
    print("\n" + "=" * 60)

def main():
    """Main setup routine"""
    print_header()
    
    # System checks
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up directories", setup_directories), 
        ("Testing system", test_system)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    # Success!
    show_usage_instructions()

if __name__ == "__main__":
    main()