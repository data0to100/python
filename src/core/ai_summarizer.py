"""
AI-powered text summarization module using open-source models.

This module provides text summarization capabilities using various
transformer models like BART, T5, and other open-source alternatives.
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import re
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AISummarizer:
    """
    AI-powered text summarization using open-source transformer models.
    
    Supports multiple models including BART, T5, and distilled variants
    for different quality/speed trade-offs.
    """
    
    SUPPORTED_MODELS = {
        'bart-large-cnn': {
            'model_name': 'facebook/bart-large-cnn',
            'type': 'bart',
            'description': 'High quality summarization, slower',
            'max_length': 1024
        },
        'bart-base': {
            'model_name': 'facebook/bart-base',
            'type': 'bart',
            'description': 'Medium quality, faster than large',
            'max_length': 1024
        },
        't5-small': {
            'model_name': 't5-small',
            'type': 't5',
            'description': 'Fast, lower quality',
            'max_length': 512
        },
        't5-base': {
            'model_name': 't5-base',
            'type': 't5',
            'description': 'Good balance of speed and quality',
            'max_length': 512
        },
        'distilbart-cnn': {
            'model_name': 'sshleifer/distilbart-cnn-12-6',
            'type': 'bart',
            'description': 'Distilled BART, good speed/quality balance',
            'max_length': 1024
        }
    }
    
    def __init__(self, model_name: str = 'distilbart-cnn', device: str = 'auto'):
        """
        Initialize the AI summarizer.
        
        Args:
            model_name: Name of the model to use (from SUPPORTED_MODELS)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS.get(model_name)
        
        if not self.model_config:
            raise ValueError(f"Unsupported model: {model_name}. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing AI summarizer with model: {self.model_config['model_name']}")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the selected model and tokenizer."""
        try:
            model_path = self.model_config['model_name']
            
            if self.model_config['type'] == 'bart':
                if 'facebook/bart' in model_path:
                    self.tokenizer = BartTokenizer.from_pretrained(model_path)
                    self.model = BartForConditionalGeneration.from_pretrained(model_path)
                else:
                    # For distilled BART
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            elif self.model_config['type'] == 't5':
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            # Move model to device
            self.model.to(self.device)
            
            # Create pipeline for easier use
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
            
            logger.info(f"Successfully loaded model: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better summarization.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\bPage \d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*$', '', text)  # Remove trailing page numbers
        
        # Remove repetitive patterns
        text = re.sub(r'(\w+\s+){10,}', '', text)  # Remove very repetitive content
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Split text into chunks that fit within model limits.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (auto-detected if None)
            
        Returns:
            List of text chunks
        """
        if max_tokens is None:
            max_tokens = self.model_config['max_length'] - 100  # Reserve for special tokens
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        max_chars = max_tokens * 4
        
        # If text is short enough, return as single chunk
        if len(text) <= max_chars:
            return [text]
        
        # Split by sentences for better coherence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        length_penalty: float = 2.0,
        num_beams: int = 4,
        early_stopping: bool = True
    ) -> str:
        """
        Summarize a single text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            length_penalty: Length penalty for beam search
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early in beam search
            
        Returns:
            Summarized text
        """
        if not text or not text.strip():
            return ""
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        if len(processed_text.split()) < min_length:
            logger.warning("Text too short for meaningful summarization")
            return processed_text
        
        try:
            # For T5 models, add the task prefix
            if self.model_config['type'] == 't5':
                input_text = f"summarize: {processed_text}"
            else:
                input_text = processed_text
            
            # Generate summary
            result = self.pipeline(
                input_text,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping,
                do_sample=False
            )
            
            summary = result[0]['summary_text']
            logger.info(f"Generated summary of length: {len(summary.split())} words")
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            # Return truncated original text as fallback
            words = processed_text.split()
            if len(words) > max_length:
                return " ".join(words[:max_length]) + "..."
            return processed_text
    
    def summarize_long_text(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        strategy: str = 'chunk_then_summarize'
    ) -> str:
        """
        Summarize long text using chunking strategies.
        
        Args:
            text: Long text to summarize
            max_length: Maximum length of final summary
            min_length: Minimum length of final summary
            strategy: Summarization strategy ('chunk_then_summarize' or 'extract_then_summarize')
            
        Returns:
            Summarized text
        """
        processed_text = self._preprocess_text(text)
        
        if strategy == 'chunk_then_summarize':
            # Split text into chunks and summarize each
            chunks = self._chunk_text(processed_text)
            
            if len(chunks) == 1:
                return self.summarize_text(chunks[0], max_length, min_length)
            
            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                summary = self.summarize_text(
                    chunk,
                    max_length=max_length // len(chunks) + 50,
                    min_length=min(20, min_length // len(chunks))
                )
                if summary:
                    chunk_summaries.append(summary)
            
            # Combine and re-summarize if needed
            combined_summary = " ".join(chunk_summaries)
            
            if len(combined_summary.split()) > max_length:
                return self.summarize_text(combined_summary, max_length, min_length)
            else:
                return combined_summary
        
        else:  # extract_then_summarize
            # Extract key sentences then summarize
            sentences = re.split(r'(?<=[.!?])\s+', processed_text)
            
            # Simple extractive approach: take every nth sentence
            key_sentences = sentences[::max(1, len(sentences) // 10)]
            extracted_text = " ".join(key_sentences)
            
            return self.summarize_text(extracted_text, max_length, min_length)
    
    def summarize_pages(
        self,
        pages: List[Dict[str, str]],
        summarize_individually: bool = False,
        max_length: int = 150,
        min_length: int = 30
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Summarize multiple pages from PDF extraction.
        
        Args:
            pages: List of page dictionaries with 'page' and 'text' keys
            summarize_individually: Whether to summarize each page separately
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Single summary string or list of page summaries
        """
        if not pages:
            return "" if not summarize_individually else []
        
        if summarize_individually:
            # Summarize each page separately
            page_summaries = []
            for page_data in pages:
                page_num = page_data.get('page', 0)
                text = page_data.get('text', '')
                
                if text.strip():
                    logger.info(f"Summarizing page {page_num}")
                    summary = self.summarize_text(text, max_length, min_length)
                    page_summaries.append({
                        'page': page_num,
                        'text': text,
                        'summary': summary
                    })
                else:
                    page_summaries.append({
                        'page': page_num,
                        'text': text,
                        'summary': ''
                    })
            
            return page_summaries
        
        else:
            # Combine all pages and summarize together
            combined_text = " ".join([page.get('text', '') for page in pages if page.get('text', '').strip()])
            
            if not combined_text.strip():
                return ""
            
            logger.info(f"Summarizing combined text from {len(pages)} pages")
            return self.summarize_long_text(combined_text, max_length, min_length)
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'name': self.model_name,
            'description': self.model_config['description'],
            'model_path': self.model_config['model_name'],
            'max_length': self.model_config['max_length'],
            'device': self.device,
            'type': self.model_config['type']
        }
    
    @classmethod
    def get_available_models(cls) -> Dict:
        """Get information about all available models."""
        return cls.SUPPORTED_MODELS