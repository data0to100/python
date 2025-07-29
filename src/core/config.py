"""
Configuration management for PDF Script to Speech/Video Converter.

This module handles all configuration settings including voice parameters,
file paths, and API keys.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class VoiceConfig:
    """Configuration for voice synthesis parameters."""
    engine: str = "pyttsx3"  # "pyttsx3", "gtts", or "elevenlabs"
    speed: float = 1.0  # 0.5 to 2.0
    pitch: int = 0  # -50 to 50
    language: str = "en"  # Language code for gTTS
    voice_id: Optional[str] = None  # For specific voice selection

@dataclass
class AIConfig:
    """Configuration for AI summarization."""
    summarization_model: str = "distilbart-cnn"  # Model to use for summarization
    max_summary_length: int = 150  # Maximum words in summary
    min_summary_length: int = 30   # Minimum words in summary
    device: str = "auto"  # "auto", "cpu", or "cuda"
    summarize_individually: bool = False  # Summarize pages individually

@dataclass
class RenderforestConfig:
    """Configuration for Renderforest integration."""
    api_key: Optional[str] = None
    template: str = "minimal_typography"  # Default template
    video_quality: str = "high"  # "low", "medium", "high"
    include_voiceover: bool = True
    style_options: Dict[str, Any] = None

@dataclass
class OutputConfig:
    """Configuration for output settings."""
    audio_format: str = "mp3"  # "mp3" or "wav"
    video_format: str = "mp4"
    audio_quality: str = "high"  # "low", "medium", "high"
    video_resolution: tuple = (1920, 1080)
    fps: int = 24

@dataclass
class PathConfig:
    """Configuration for file paths."""
    pdf_input_dir: Path = Path("pdf_scripts")
    audio_output_dir: Path = Path("audio_output")
    video_output_dir: Path = Path("video_output")
    temp_dir: Path = Path("temp")
    background_image: Optional[Path] = None

class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        self.voice = VoiceConfig()
        self.output = OutputConfig()
        self.paths = PathConfig()
        self.ai = AIConfig()
        self.renderforest = RenderforestConfig()
        self.api_keys = self._load_api_keys()
        self._ensure_directories()
    
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from environment variables."""
        return {
            "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.paths.pdf_input_dir,
            self.paths.audio_output_dir,
            self.paths.video_output_dir,
            self.paths.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def update_voice_config(self, **kwargs) -> None:
        """Update voice configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.voice, key):
                setattr(self.voice, key, value)
    
    def update_output_config(self, **kwargs) -> None:
        """Update output configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.output, key):
                setattr(self.output, key, value)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for serialization."""
        return {
            "voice": {
                "engine": self.voice.engine,
                "speed": self.voice.speed,
                "pitch": self.voice.pitch,
                "language": self.voice.language,
                "voice_id": self.voice.voice_id
            },
            "output": {
                "audio_format": self.output.audio_format,
                "video_format": self.output.video_format,
                "audio_quality": self.output.audio_quality,
                "video_resolution": self.output.video_resolution,
                "fps": self.output.fps
            },
            "paths": {
                "pdf_input_dir": str(self.paths.pdf_input_dir),
                "audio_output_dir": str(self.paths.audio_output_dir),
                "video_output_dir": str(self.paths.video_output_dir),
                "temp_dir": str(self.paths.temp_dir)
            }
        }

# Global configuration instance
config = AppConfig()