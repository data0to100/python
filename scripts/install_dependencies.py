#!/usr/bin/env python3
"""
Installation script for PDF Script to Speech/Video Converter.

This script checks for and helps install system dependencies required for the application.
"""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd if shell else cmd.split(),
            capture_output=True,
            text=True,
            shell=shell
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip...")
    success, _, _ = run_command("pip --version")
    
    if success:
        print("✅ pip is available")
        return True
    else:
        print("❌ pip is not available")
        return False

def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("📦 Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    success, stdout, stderr = run_command(f"pip install -r {requirements_file}")
    
    if success:
        print("✅ Python dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install Python dependencies")
        print(f"Error: {stderr}")
        return False

def check_system_dependencies():
    """Check for system-level dependencies."""
    print("🔧 Checking system dependencies...")
    
    system = platform.system().lower()
    missing_deps = []
    
    # Check for FFmpeg (required for moviepy)
    success, _, _ = run_command("ffmpeg -version")
    if not success:
        missing_deps.append("ffmpeg")
    else:
        print("✅ FFmpeg is available")
    
    # Check for espeak (required for pyttsx3 on Linux)
    if system == "linux":
        success, _, _ = run_command("espeak --version")
        if not success:
            missing_deps.append("espeak")
        else:
            print("✅ espeak is available")
    
    if missing_deps:
        print(f"❌ Missing system dependencies: {', '.join(missing_deps)}")
        print_installation_instructions(missing_deps, system)
        return False
    
    print("✅ All system dependencies are available")
    return True

def print_installation_instructions(missing_deps, system):
    """Print installation instructions for missing dependencies."""
    print("\n📋 Installation instructions:")
    
    if "ffmpeg" in missing_deps:
        print("\nFFmpeg installation:")
        if system == "windows":
            print("  1. Download from: https://ffmpeg.org/download.html")
            print("  2. Extract and add to PATH")
            print("  3. Or use chocolatey: choco install ffmpeg")
        elif system == "darwin":  # macOS
            print("  brew install ffmpeg")
        elif system == "linux":
            print("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
            print("  CentOS/RHEL: sudo yum install ffmpeg")
            print("  Arch: sudo pacman -S ffmpeg")
    
    if "espeak" in missing_deps:
        print("\nespeak installation (Linux):")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install espeak")
        print("  CentOS/RHEL: sudo yum install espeak")
        print("  Arch: sudo pacman -S espeak")

def test_imports():
    """Test if all major modules can be imported."""
    print("🧪 Testing module imports...")
    
    modules_to_test = [
        "fitz",  # PyMuPDF
        "pdfplumber",
        "pyttsx3",
        "gtts",
        "moviepy.editor",
        "streamlit",
        "PIL",  # Pillow
        "requests"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Try reinstalling dependencies: pip install -r requirements.txt")
        return False
    
    print("✅ All modules imported successfully")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "pdf_scripts",
        "audio_output", 
        "video_output",
        "temp"
    ]
    
    base_path = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def main():
    """Main installation process."""
    print("🚀 PDF Script to Speech/Video Converter - Installation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("pip", check_pip),
        ("Python Dependencies", install_python_dependencies),
        ("System Dependencies", check_system_dependencies),
        ("Module Imports", test_imports),
        ("Directories", create_directories)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            failed_checks.append(check_name)
    
    print("\n" + "=" * 60)
    
    if failed_checks:
        print("❌ Installation incomplete. Failed checks:")
        for check in failed_checks:
            print(f"  - {check}")
        print("\nPlease resolve the issues above and run the script again.")
        return 1
    else:
        print("✅ Installation completed successfully!")
        print("\nYou can now use the PDF Script Converter:")
        print("  Web interface: python main.py --interface web")
        print("  CLI interface: python main.py --interface cli --help")
        print("  Sample script: python examples/sample_script.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())