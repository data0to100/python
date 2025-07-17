"""
Subtitle generation utilities for PDF Script to Speech/Video Converter.

This module provides utilities for creating subtitle files and managing
subtitle timing and formatting.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    text: str          # Subtitle text
    index: int = 0     # Subtitle index (for SRT format)

class SubtitleGenerator:
    """
    Generates subtitle files from text and timing information.
    
    Supports multiple subtitle formats including SRT, VTT, and ASS.
    """
    
    def __init__(self):
        self.max_chars_per_line = 60
        self.max_lines_per_subtitle = 2
        self.min_duration = 1.0  # Minimum subtitle duration in seconds
        self.max_duration = 6.0  # Maximum subtitle duration in seconds
    
    def create_subtitles_from_pages(self, pages: List[Dict], 
                                   audio_durations: List[float]) -> List[SubtitleEntry]:
        """
        Create subtitle entries from page text and audio durations.
        
        Args:
            pages: List of page data with text
            audio_durations: List of audio durations for each page
            
        Returns:
            List of SubtitleEntry objects
        """
        if len(pages) != len(audio_durations):
            logger.error("Number of pages and audio durations must match")
            return []
        
        subtitles = []
        current_time = 0.0
        
        for page, duration in zip(pages, audio_durations):
            text = page['text'].strip()
            if not text:
                current_time += duration
                continue
            
            # Split text into subtitle-sized chunks
            chunks = self._split_text_for_subtitles(text, duration)
            
            chunk_duration = duration / len(chunks) if chunks else duration
            
            for i, chunk in enumerate(chunks):
                start_time = current_time + (i * chunk_duration)
                end_time = min(start_time + chunk_duration, current_time + duration)
                
                # Ensure minimum duration
                if end_time - start_time < self.min_duration:
                    end_time = start_time + self.min_duration
                
                subtitle = SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    text=chunk,
                    index=len(subtitles) + 1
                )
                subtitles.append(subtitle)
            
            current_time += duration
        
        return subtitles
    
    def create_subtitles_from_text(self, text: str, 
                                  total_duration: float) -> List[SubtitleEntry]:
        """
        Create subtitle entries from continuous text and total duration.
        
        Args:
            text: Complete text content
            total_duration: Total duration in seconds
            
        Returns:
            List of SubtitleEntry objects
        """
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Calculate duration per sentence
        chars_per_second = len(text) / total_duration
        subtitles = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            # Estimate duration based on character count
            estimated_duration = len(sentence) / chars_per_second
            
            # Apply min/max constraints
            duration = max(self.min_duration, 
                          min(self.max_duration, estimated_duration))
            
            # Wrap text if too long
            wrapped_text = self._wrap_text(sentence)
            
            subtitle = SubtitleEntry(
                start_time=current_time,
                end_time=current_time + duration,
                text=wrapped_text,
                index=i + 1
            )
            subtitles.append(subtitle)
            current_time += duration
        
        # Adjust timing if total doesn't match
        if current_time != total_duration:
            scale_factor = total_duration / current_time
            for subtitle in subtitles:
                subtitle.start_time *= scale_factor
                subtitle.end_time *= scale_factor
        
        return subtitles
    
    def _split_text_for_subtitles(self, text: str, duration: float) -> List[str]:
        """
        Split text into appropriately sized chunks for subtitles.
        
        Args:
            text: Text to split
            duration: Available duration for the text
            
        Returns:
            List of text chunks
        """
        # For very short durations, use the whole text
        if duration <= self.max_duration:
            return [self._wrap_text(text)]
        
        # Split by sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the limit, start a new chunk
            if (current_length + sentence_length > self.max_chars_per_line * self.max_lines_per_subtitle 
                and current_chunk):
                chunks.append(self._wrap_text(' '.join(current_chunk)))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add the last chunk
        if current_chunk:
            chunks.append(self._wrap_text(' '.join(current_chunk)))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex to split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up and filter empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _wrap_text(self, text: str) -> str:
        """
        Wrap text to fit subtitle display constraints.
        
        Args:
            text: Text to wrap
            
        Returns:
            Wrapped text
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word would exceed the line length
            if current_length + len(word) + 1 <= self.max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                # Start a new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to maximum lines per subtitle
        if len(lines) > self.max_lines_per_subtitle:
            lines = lines[:self.max_lines_per_subtitle]
            # Add ellipsis to indicate continuation
            if not lines[-1].endswith('...'):
                lines[-1] += '...'
        
        return '\n'.join(lines)
    
    def export_srt(self, subtitles: List[SubtitleEntry], output_path: Path) -> bool:
        """
        Export subtitles in SRT format.
        
        Args:
            subtitles: List of subtitle entries
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for subtitle in subtitles:
                    # Format timing
                    start_time = self._format_srt_time(subtitle.start_time)
                    end_time = self._format_srt_time(subtitle.end_time)
                    
                    # Write subtitle entry
                    f.write(f"{subtitle.index}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{subtitle.text}\n\n")
            
            logger.info(f"SRT subtitles exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export SRT subtitles: {str(e)}")
            return False
    
    def export_vtt(self, subtitles: List[SubtitleEntry], output_path: Path) -> bool:
        """
        Export subtitles in WebVTT format.
        
        Args:
            subtitles: List of subtitle entries
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for subtitle in subtitles:
                    # Format timing
                    start_time = self._format_vtt_time(subtitle.start_time)
                    end_time = self._format_vtt_time(subtitle.end_time)
                    
                    # Write subtitle entry
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{subtitle.text}\n\n")
            
            logger.info(f"VTT subtitles exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export VTT subtitles: {str(e)}")
            return False
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def get_subtitle_statistics(self, subtitles: List[SubtitleEntry]) -> Dict[str, any]:
        """
        Get statistics about the subtitles.
        
        Args:
            subtitles: List of subtitle entries
            
        Returns:
            Dictionary with subtitle statistics
        """
        if not subtitles:
            return {
                'total_subtitles': 0,
                'total_duration': 0,
                'average_duration': 0,
                'total_characters': 0,
                'average_characters': 0
            }
        
        total_duration = max(sub.end_time for sub in subtitles)
        total_chars = sum(len(sub.text) for sub in subtitles)
        subtitle_durations = [sub.end_time - sub.start_time for sub in subtitles]
        
        return {
            'total_subtitles': len(subtitles),
            'total_duration': total_duration,
            'average_duration': sum(subtitle_durations) / len(subtitle_durations),
            'total_characters': total_chars,
            'average_characters': total_chars / len(subtitles),
            'min_duration': min(subtitle_durations),
            'max_duration': max(subtitle_durations)
        }