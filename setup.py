#!/usr/bin/env python3
"""
Setup script for PDF Script to Speech/Video Converter.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="pdf-script-converter",
    version="1.0.0",
    description="Convert PDF scripts to speech and video with subtitles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PDF Script Converter Team",
    author_email="contact@example.com",
    url="https://github.com/yourusername/pdf-script-converter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "reportlab>=4.0.0"
        ],
        "audio": [
            "pydub>=0.25.0",
            "soundfile>=0.12.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "pdf-convert=interfaces.cli:main",
            "pdf-converter-web=interfaces.streamlit_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.8",
    keywords="pdf, text-to-speech, video, subtitles, conversion, accessibility",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pdf-script-converter/issues",
        "Source": "https://github.com/yourusername/pdf-script-converter",
        "Documentation": "https://github.com/yourusername/pdf-script-converter/wiki",
    },
)