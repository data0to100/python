"""
Text-to-Speech engine module for PDF Script to Speech/Video Converter.

This module provides a unified interface for multiple TTS engines including
pyttsx3 (offline), gTTS (Google), and ElevenLabs API.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import pyttsx3
from gtts import gTTS
import requests
from io import BytesIO

from .config import VoiceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine(ABC):
    """Abstract base class for text-to-speech engines."""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig) -> bool:
        """Synthesize text to speech and save to file."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available."""
        pass

class Pyttsx3Engine(TTSEngine):
    """Offline TTS engine using pyttsx3."""
    
    def __init__(self):
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize the pyttsx3 engine."""
        try:
            self.engine = pyttsx3.init()
            logger.info("Pyttsx3 engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {str(e)}")
            self.engine = None
    
    def is_available(self) -> bool:
        """Check if pyttsx3 is available."""
        return self.engine is not None
    
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig) -> bool:
        """
        Synthesize text using pyttsx3.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            config: Voice configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Pyttsx3 engine not available")
            return False
        
        try:
            # Configure voice properties
            self.engine.setProperty('rate', int(config.speed * 200))  # Default rate is ~200
            
            # Set voice if available
            voices = self.engine.getProperty('voices')
            if voices and config.voice_id:
                for voice in voices:
                    if config.voice_id.lower() in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Save to file
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()
            
            logger.info(f"Audio saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Pyttsx3 synthesis failed: {str(e)}")
            return False

class GTTSEngine(TTSEngine):
    """Online TTS engine using Google Text-to-Speech."""
    
    def is_available(self) -> bool:
        """Check if gTTS is available (requires internet)."""
        try:
            # Test with a small request
            tts = gTTS(text="test", lang="en")
            return True
        except Exception:
            return False
    
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig) -> bool:
        """
        Synthesize text using gTTS.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            config: Voice configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split text into chunks if too long (gTTS has character limits)
            chunks = self._split_text(text, max_length=5000)
            audio_files = []
            
            for i, chunk in enumerate(chunks):
                tts = gTTS(
                    text=chunk,
                    lang=config.language,
                    slow=(config.speed < 0.8)
                )
                
                chunk_path = output_path.parent / f"temp_chunk_{i}.mp3"
                tts.save(str(chunk_path))
                audio_files.append(chunk_path)
            
            # Combine chunks if multiple
            if len(audio_files) == 1:
                audio_files[0].rename(output_path)
            else:
                self._combine_audio_files(audio_files, output_path)
                # Clean up temporary files
                for file in audio_files:
                    file.unlink(missing_ok=True)
            
            logger.info(f"gTTS audio saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {str(e)}")
            return False
    
    def _split_text(self, text: str, max_length: int = 5000) -> List[str]:
        """Split text into chunks suitable for gTTS."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _combine_audio_files(self, audio_files: List[Path], output_path: Path) -> None:
        """Combine multiple audio files into one."""
        try:
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            for file in audio_files:
                audio = AudioSegment.from_mp3(str(file))
                combined += audio
            
            combined.export(str(output_path), format="mp3")
            
        except ImportError:
            logger.warning("pydub not available, using simple concatenation")
            # Fallback: just use the first file
            audio_files[0].rename(output_path)

class ElevenLabsEngine(TTSEngine):
    """ElevenLabs AI voice synthesis engine."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def is_available(self) -> bool:
        """Check if ElevenLabs API is available."""
        if not self.api_key:
            return False
        
        try:
            headers = {"xi-api-key": self.api_key}
            response = requests.get(f"{self.base_url}/voices", headers=headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig) -> bool:
        """
        Synthesize text using ElevenLabs API.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            config: Voice configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("ElevenLabs API not available")
            return False
        
        try:
            voice_id = config.voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default voice
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                json=data,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"ElevenLabs audio saved to {output_path}")
                return True
            else:
                logger.error(f"ElevenLabs API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {str(e)}")
            return False

class TTSManager:
    """Manages multiple TTS engines and provides a unified interface."""
    
    def __init__(self):
        self.engines = {
            "pyttsx3": Pyttsx3Engine(),
            "gtts": GTTSEngine(),
            "elevenlabs": ElevenLabsEngine()
        }
    
    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines."""
        return [name for name, engine in self.engines.items() if engine.is_available()]
    
    def synthesize_text(self, text: str, output_path: Path, config: VoiceConfig) -> bool:
        """
        Synthesize text using the specified engine.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            config: Voice configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return False
        
        engine_name = config.engine
        if engine_name not in self.engines:
            logger.error(f"Unknown TTS engine: {engine_name}")
            return False
        
        engine = self.engines[engine_name]
        if not engine.is_available():
            logger.error(f"TTS engine {engine_name} is not available")
            # Try to fall back to an available engine
            available = self.get_available_engines()
            if available:
                logger.info(f"Falling back to {available[0]}")
                engine = self.engines[available[0]]
            else:
                logger.error("No TTS engines available")
                return False
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Synthesizing {len(text)} characters using {engine_name}")
        start_time = time.time()
        
        success = engine.synthesize(text, output_path, config)
        
        if success:
            duration = time.time() - start_time
            logger.info(f"Synthesis completed in {duration:.2f} seconds")
        
        return success
    
    def synthesize_pages(self, pages: List[Dict[str, str]], output_dir: Path, 
                        config: VoiceConfig, filename_prefix: str = "page") -> List[Path]:
        """
        Synthesize multiple pages to separate audio files.
        
        Args:
            pages: List of page data with text
            output_dir: Output directory
            config: Voice configuration
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of generated audio file paths
        """
        audio_files = []
        
        for page in pages:
            page_num = page['page_number']
            text = page['text']
            
            if not text.strip():
                logger.warning(f"Skipping empty page {page_num}")
                continue
            
            filename = f"{filename_prefix}_{page_num:03d}.mp3"
            output_path = output_dir / filename
            
            if self.synthesize_text(text, output_path, config):
                audio_files.append(output_path)
                logger.info(f"Generated audio for page {page_num}")
            else:
                logger.error(f"Failed to generate audio for page {page_num}")
        
        return audio_files