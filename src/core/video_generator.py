"""
Video generation module for PDF Script to Speech/Video Converter.

This module creates video files by combining audio with background images
and optionally adding subtitles using moviepy.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile

from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
    TextClip, concatenate_videoclips, concatenate_audioclips
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .config import OutputConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    """
    Generates video files with audio and optional subtitles.
    
    Uses moviepy to create videos by combining audio files with
    background images and subtitle overlays.
    """
    
    def __init__(self):
        self.default_background_color = (30, 30, 40)  # Dark blue-gray
        self.subtitle_style = {
            'fontsize': 36,
            'color': 'white',
            'font': 'Arial-Bold',
            'stroke_color': 'black',
            'stroke_width': 2
        }
    
    def create_background_image(self, size: Tuple[int, int], 
                              color: Tuple[int, int, int] = None,
                              title: str = None) -> Path:
        """
        Create a simple background image with optional title.
        
        Args:
            size: Image dimensions (width, height)
            color: RGB color tuple
            title: Optional title text
            
        Returns:
            Path to created background image
        """
        color = color or self.default_background_color
        
        # Create image
        img = Image.new('RGB', size, color)
        draw = ImageDraw.Draw(img)
        
        if title:
            try:
                # Try to use a nice font
                font_size = min(size) // 15
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                # Fall back to default font
                font = ImageFont.load_default()
            
            # Calculate text position (centered)
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2
            
            # Draw text with shadow for better visibility
            draw.text((x + 2, y + 2), title, fill=(0, 0, 0), font=font)
            draw.text((x, y), title, fill=(255, 255, 255), font=font)
        
        # Save to temporary file
        temp_path = Path(tempfile.mktemp(suffix='.png'))
        img.save(temp_path)
        
        return temp_path
    
    def create_video_from_audio(self, audio_path: Path, output_path: Path,
                               background_image: Optional[Path] = None,
                               config: OutputConfig = None) -> bool:
        """
        Create video from audio file with background image.
        
        Args:
            audio_path: Path to audio file
            output_path: Output video path
            background_image: Optional background image
            config: Output configuration
            
        Returns:
            True if successful, False otherwise
        """
        config = config or OutputConfig()
        
        try:
            # Load audio
            audio = AudioFileClip(str(audio_path))
            duration = audio.duration
            
            # Create or load background
            if background_image and background_image.exists():
                background = ImageClip(str(background_image))
            else:
                # Create default background
                bg_path = self.create_background_image(
                    config.video_resolution,
                    title="Audio Script"
                )
                background = ImageClip(str(bg_path))
                # Clean up temporary file later
                bg_path.unlink(missing_ok=True)
            
            # Set duration and resize
            background = background.set_duration(duration).resize(config.video_resolution)
            
            # Combine video and audio
            video = background.set_audio(audio)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write video file
            video.write_videofile(
                str(output_path),
                fps=config.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            audio.close()
            background.close()
            video.close()
            
            logger.info(f"Video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create video: {str(e)}")
            return False
    
    def create_video_with_subtitles(self, audio_path: Path, output_path: Path,
                                   subtitles: List[Dict], 
                                   background_image: Optional[Path] = None,
                                   config: OutputConfig = None) -> bool:
        """
        Create video with subtitles.
        
        Args:
            audio_path: Path to audio file
            output_path: Output video path
            subtitles: List of subtitle dictionaries with 'start', 'end', 'text'
            background_image: Optional background image
            config: Output configuration
            
        Returns:
            True if successful, False otherwise
        """
        config = config or OutputConfig()
        
        try:
            # Load audio
            audio = AudioFileClip(str(audio_path))
            duration = audio.duration
            
            # Create or load background
            if background_image and background_image.exists():
                background = ImageClip(str(background_image))
            else:
                bg_path = self.create_background_image(
                    config.video_resolution,
                    title="Script with Subtitles"
                )
                background = ImageClip(str(bg_path))
                bg_path.unlink(missing_ok=True)
            
            # Set duration and resize
            background = background.set_duration(duration).resize(config.video_resolution)
            
            # Create subtitle clips
            subtitle_clips = []
            for subtitle in subtitles:
                if subtitle['text'].strip():
                    txt_clip = TextClip(
                        subtitle['text'],
                        fontsize=self.subtitle_style['fontsize'],
                        color=self.subtitle_style['color'],
                        font=self.subtitle_style['font'],
                        stroke_color=self.subtitle_style['stroke_color'],
                        stroke_width=self.subtitle_style['stroke_width']
                    ).set_position(('center', 'bottom')).set_start(
                        subtitle['start']
                    ).set_end(subtitle['end'])
                    
                    subtitle_clips.append(txt_clip)
            
            # Combine background and subtitles
            if subtitle_clips:
                video = CompositeVideoClip([background] + subtitle_clips)
            else:
                video = background
            
            # Add audio
            video = video.set_audio(audio)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write video file
            video.write_videofile(
                str(output_path),
                fps=config.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            audio.close()
            background.close()
            video.close()
            for clip in subtitle_clips:
                clip.close()
            
            logger.info(f"Video with subtitles created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create video with subtitles: {str(e)}")
            return False
    
    def combine_audio_files(self, audio_files: List[Path], output_path: Path) -> bool:
        """
        Combine multiple audio files into one.
        
        Args:
            audio_files: List of audio file paths
            output_path: Output audio path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(audio_files) == 1:
                # Just copy the single file
                import shutil
                shutil.copy2(audio_files[0], output_path)
                return True
            
            # Load all audio clips
            clips = [AudioFileClip(str(path)) for path in audio_files]
            
            # Concatenate
            final_audio = concatenate_audioclips(clips)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio file
            final_audio.write_audiofile(str(output_path))
            
            # Clean up
            for clip in clips:
                clip.close()
            final_audio.close()
            
            logger.info(f"Combined audio saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to combine audio files: {str(e)}")
            return False
    
    def create_video_from_pages(self, pages: List[Dict], audio_files: List[Path],
                               output_path: Path, background_image: Optional[Path] = None,
                               config: OutputConfig = None, 
                               include_subtitles: bool = True) -> bool:
        """
        Create a complete video from multiple pages and audio files.
        
        Args:
            pages: List of page data
            audio_files: List of corresponding audio files
            output_path: Output video path
            background_image: Optional background image
            config: Output configuration
            include_subtitles: Whether to include subtitles
            
        Returns:
            True if successful, False otherwise
        """
        config = config or OutputConfig()
        
        try:
            if len(pages) != len(audio_files):
                logger.error("Number of pages and audio files must match")
                return False
            
            video_clips = []
            current_time = 0
            
            for page, audio_path in zip(pages, audio_files):
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue
                
                # Load audio
                audio = AudioFileClip(str(audio_path))
                duration = audio.duration
                
                # Create background for this segment
                if background_image and background_image.exists():
                    background = ImageClip(str(background_image))
                else:
                    bg_path = self.create_background_image(
                        config.video_resolution,
                        title=f"Page {page['page_number']}"
                    )
                    background = ImageClip(str(bg_path))
                    bg_path.unlink(missing_ok=True)
                
                # Set duration and resize
                background = background.set_duration(duration).resize(config.video_resolution)
                
                # Add subtitles if requested
                if include_subtitles and page['text'].strip():
                    # Create simple subtitle for the entire page
                    # For more advanced subtitles, word-level timing would be needed
                    subtitle_text = self._wrap_text(page['text'], max_width=60)
                    
                    txt_clip = TextClip(
                        subtitle_text,
                        fontsize=self.subtitle_style['fontsize'],
                        color=self.subtitle_style['color'],
                        font=self.subtitle_style['font'],
                        stroke_color=self.subtitle_style['stroke_color'],
                        stroke_width=self.subtitle_style['stroke_width']
                    ).set_position(('center', 'bottom')).set_duration(duration)
                    
                    video_segment = CompositeVideoClip([background, txt_clip])
                else:
                    video_segment = background
                
                # Add audio
                video_segment = video_segment.set_audio(audio)
                video_clips.append(video_segment)
                
                current_time += duration
            
            if not video_clips:
                logger.error("No video clips created")
                return False
            
            # Concatenate all video clips
            final_video = concatenate_videoclips(video_clips)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write final video
            final_video.write_videofile(
                str(output_path),
                fps=config.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            for clip in video_clips:
                clip.close()
            final_video.close()
            
            logger.info(f"Complete video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create complete video: {str(e)}")
            return False
    
    def _wrap_text(self, text: str, max_width: int = 60) -> str:
        """
        Wrap text to specified width for better subtitle display.
        
        Args:
            text: Text to wrap
            max_width: Maximum characters per line
            
        Returns:
            Wrapped text
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to 3 lines for subtitle display
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] += "..."
        
        return '\n'.join(lines)