"""
PDF text extraction module for PDF Script to Speech/Video Converter.

This module provides robust PDF text extraction using multiple libraries
to ensure maximum compatibility and accuracy.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF text extraction using multiple extraction methods.
    
    Uses PyMuPDF as primary method and pdfplumber as fallback
    to ensure robust text extraction from various PDF formats.
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_pymupdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and text
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean up the text
                text = self._clean_text(text)
                
                if text.strip():  # Only add pages with content
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'extraction_method': 'pymupdf'
                    })
            
            doc.close()
            return pages
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {str(e)}")
            return []
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        Extract text from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and text
        """
        try:
            pages = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text:
                        # Clean up the text
                        text = self._clean_text(text)
                        
                        if text.strip():  # Only add pages with content
                            pages.append({
                                'page_number': page_num + 1,
                                'text': text,
                                'extraction_method': 'pdfplumber'
                            })
            
            return pages
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text suitable for TTS
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic heuristics)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) < 3:
                continue
            # Skip lines that are just numbers (likely page numbers)
            if line.isdigit():
                continue
            cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        
        # Normalize punctuation for better TTS
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
        text = re.sub(r'\s*-\s*', ' - ', text)  # Normalize dashes
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def extract_text(self, pdf_path: Path, method: str = "auto") -> List[Dict[str, str]]:
        """
        Extract text from PDF using specified or automatic method selection.
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ("auto", "pymupdf", "pdfplumber")
            
        Returns:
            List of dictionaries containing page data
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF format is not supported
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {pdf_path.suffix}")
        
        logger.info(f"Extracting text from {pdf_path} using method: {method}")
        
        pages = []
        
        if method == "pymupdf":
            pages = self.extract_text_pymupdf(pdf_path)
        elif method == "pdfplumber":
            pages = self.extract_text_pdfplumber(pdf_path)
        elif method == "auto":
            # Try PyMuPDF first, fall back to pdfplumber
            pages = self.extract_text_pymupdf(pdf_path)
            if not pages:
                logger.info("PyMuPDF failed, trying pdfplumber...")
                pages = self.extract_text_pdfplumber(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        if not pages:
            logger.warning(f"No text extracted from {pdf_path}")
        else:
            total_chars = sum(len(page['text']) for page in pages)
            logger.info(f"Extracted {len(pages)} pages, {total_chars} characters")
        
        return pages
    
    def get_preview(self, pdf_path: Path, max_chars: int = 500) -> str:
        """
        Get a preview of the PDF content.
        
        Args:
            pdf_path: Path to the PDF file
            max_chars: Maximum characters to return
            
        Returns:
            Preview text
        """
        try:
            pages = self.extract_text(pdf_path)
            if not pages:
                return "No text could be extracted from this PDF."
            
            # Combine text from all pages
            full_text = " ".join(page['text'] for page in pages)
            
            if len(full_text) <= max_chars:
                return full_text
            else:
                return full_text[:max_chars] + "..."
        
        except Exception as e:
            return f"Error previewing PDF: {str(e)}"
    
    def get_text_stats(self, pages: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Get statistics about extracted text.
        
        Args:
            pages: List of page data
            
        Returns:
            Dictionary with text statistics
        """
        if not pages:
            return {
                'total_pages': 0,
                'total_characters': 0,
                'total_words': 0,
                'average_words_per_page': 0
            }
        
        total_chars = sum(len(page['text']) for page in pages)
        total_words = sum(len(page['text'].split()) for page in pages)
        
        return {
            'total_pages': len(pages),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_words_per_page': total_words // len(pages) if pages else 0
        }