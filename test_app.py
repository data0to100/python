#!/usr/bin/env python3
"""
Test script for PDF to Video Creator
Tests all major components of the application.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from app import PDFProcessor, AISummarizer, RenderForestAPI

class TestPDFProcessor(unittest.TestCase):
    """Test PDF processing functionality"""
    
    def setUp(self):
        self.processor = PDFProcessor()
    
    def test_extract_text_from_pdf(self):
        """Test PDF text extraction"""
        # Create a mock PDF file
        mock_file = Mock()
        mock_file.seek.return_value = None
        
        # Mock PyPDF2 response
        with patch('PyPDF2.PdfReader') as mock_pdf_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample PDF content"
            mock_pdf_reader.return_value.pages = [mock_page]
            
            result = self.processor.extract_text_from_pdf(mock_file)
            self.assertIn("Sample PDF content", result)
    
    def test_extract_text_fallback(self):
        """Test fallback to pdfplumber when PyPDF2 fails"""
        mock_file = Mock()
        mock_file.seek.return_value = None
        
        # Mock PyPDF2 to return minimal text
        with patch('PyPDF2.PdfReader') as mock_pdf_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "a"  # Minimal text
            mock_pdf_reader.return_value.pages = [mock_page]
            
            # Mock pdfplumber
            with patch('pdfplumber.open') as mock_pdfplumber:
                mock_pdf = Mock()
                mock_page_plumber = Mock()
                mock_page_plumber.extract_text.return_value = "Better extracted content"
                mock_pdf.pages = [mock_page_plumber]
                mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
                
                result = self.processor.extract_text_from_pdf(mock_file)
                self.assertIn("Better extracted content", result)

class TestAISummarizer(unittest.TestCase):
    """Test AI summarization functionality"""
    
    def setUp(self):
        self.summarizer = AISummarizer()
    
    @patch('transformers.pipeline')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSeq2SeqGeneration.from_pretrained')
    def test_model_loading(self, mock_model, mock_tokenizer, mock_pipeline):
        """Test AI model loading"""
        mock_pipeline.return_value = Mock()
        
        # Reinitialize to test loading
        summarizer = AISummarizer()
        self.assertIsNotNone(summarizer.pipeline)
    
    def test_summarize_text(self):
        """Test text summarization"""
        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{'summary_text': 'Summarized content'}]
        self.summarizer.pipeline = mock_pipeline
        
        text = "This is a long text that needs to be summarized into a shorter version."
        result = self.summarizer.summarize_text(text)
        
        self.assertIn("Summarized content", result)
    
    def test_summarize_text_no_pipeline(self):
        """Test summarization when pipeline is not loaded"""
        self.summarizer.pipeline = None
        result = self.summarizer.summarize_text("Test text")
        self.assertEqual(result, "Error: AI model not loaded")

class TestRenderForestAPI(unittest.TestCase):
    """Test RenderForest API functionality"""
    
    def setUp(self):
        self.api = RenderForestAPI()
    
    @patch.dict(os.environ, {'RENDERFOREST_API_KEY': 'test_key'})
    def test_create_video_with_api_key(self):
        """Test video creation with API key"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'id': 'video_123', 'status': 'processing'}
            mock_post.return_value = mock_response
            
            result = self.api.create_video("Test script", "Test video")
            self.assertEqual(result['id'], 'video_123')
    
    def test_create_video_no_api_key(self):
        """Test video creation without API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = self.api.create_video("Test script", "Test video")
            self.assertIn("error", result)
            self.assertIn("API key not found", result['error'])
    
    @patch.dict(os.environ, {'RENDERFOREST_API_KEY': 'test_key'})
    def test_create_video_api_error(self):
        """Test video creation with API error"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_post.return_value = mock_response
            
            result = self.api.create_video("Test script", "Test video")
            self.assertIn("error", result)
            self.assertIn("API request failed", result['error'])

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test the complete workflow"""
        # This would test the actual integration
        # For now, just verify components can be instantiated
        pdf_processor = PDFProcessor()
        ai_summarizer = AISummarizer()
        renderforest = RenderForestAPI()
        
        self.assertIsNotNone(pdf_processor)
        self.assertIsNotNone(ai_summarizer)
        self.assertIsNotNone(renderforest)

def run_tests():
    """Run all tests"""
    print("🧪 Running PDF to Video Creator Tests")
    print("=" * 40)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPDFProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAISummarizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRenderForestAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 40)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)