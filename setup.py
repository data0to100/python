#!/usr/bin/env python3
"""
Setup script for PDF to Video Creator
Automates the installation and configuration process.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    try:
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("📝 Please edit .env file and add your RenderForest API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_streamlit():
    """Check if Streamlit is available"""
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def run_demo():
    """Run the demo script"""
    print("🎬 Running demo...")
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎬 PDF to Video Creator - Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create environment file
    if not create_env_file():
        print("❌ Setup failed during environment setup")
        sys.exit(1)
    
    # Check Streamlit
    if not check_streamlit():
        print("❌ Streamlit not available after installation")
        sys.exit(1)
    
    # Run demo
    print("\n🚀 Running demo to test installation...")
    run_demo()
    
    print("\n✅ Setup completed successfully!")
    print("\n🎉 Next steps:")
    print("1. Edit .env file and add your RenderForest API key")
    print("2. Run: streamlit run app.py")
    print("3. Open: http://localhost:8501")
    print("4. Start creating videos from your PDF webinars!")

if __name__ == "__main__":
    main()