# Makefile for PDF Script to Speech/Video Converter

.PHONY: help install install-dev clean test lint format run-web run-cli setup sample-pdf

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Run installation setup script"
	@echo "  clean        - Clean up temporary files"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  run-web      - Run Streamlit web interface"
	@echo "  run-cli      - Show CLI help"
	@echo "  sample-pdf   - Create sample PDF for testing"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

setup:
	python scripts/install_dependencies.py

# Development targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf temp/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ examples/ scripts/ --line-length=100

# Run targets
run-web:
	python main.py --interface web

run-cli:
	python main.py --interface cli --help

# Utility targets
sample-pdf:
	python scripts/create_sample_pdf.py

# Package building
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Docker targets (if using Docker)
docker-build:
	docker build -t pdf-script-converter .

docker-run:
	docker run -p 8501:8501 pdf-script-converter