#!/usr/bin/env python3
"""
Command-line interface for PDF Script to Speech/Video Converter.

This module provides a comprehensive CLI for converting PDF scripts
to audio and video with various configuration options.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config, VoiceConfig, OutputConfig
from core.pdf_processor import PDFProcessor
from core.tts_engine import TTSManager
from core.video_generator import VideoGenerator
from utils.file_manager import FileManager
from utils.subtitle_generator import SubtitleGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFConverterCLI:
    """Command-line interface for PDF to Speech/Video conversion."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.tts_manager = TTSManager()
        self.video_generator = VideoGenerator()
        self.file_manager = FileManager()
        self.subtitle_generator = SubtitleGenerator()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Convert PDF scripts to speech and video",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic audio conversion
  python cli.py --input script.pdf --output-type audio
  
  # Video with subtitles
  python cli.py --input script.pdf --output-type video --subtitles
  
  # Custom voice settings
  python cli.py --input script.pdf --voice-engine gtts --voice-speed 1.2 --language es
  
  # Preview script before conversion
  python cli.py --input script.pdf --preview
            """
        )
        
        # Input/Output options
        parser.add_argument(
            '--input', '-i',
            type=Path,
            required=True,
            help='Input PDF file path'
        )
        
        parser.add_argument(
            '--output-dir', '-o',
            type=Path,
            default=Path('.'),
            help='Output directory (default: current directory)'
        )
        
        parser.add_argument(
            '--output-type',
            choices=['audio', 'video', 'both'],
            default='audio',
            help='Output type (default: audio)'
        )
        
        # Voice configuration
        parser.add_argument(
            '--voice-engine',
            choices=['pyttsx3', 'gtts', 'elevenlabs'],
            default='pyttsx3',
            help='TTS engine to use (default: pyttsx3)'
        )
        
        parser.add_argument(
            '--voice-speed',
            type=float,
            default=1.0,
            help='Speech speed (0.5-2.0, default: 1.0)'
        )
        
        parser.add_argument(
            '--voice-pitch',
            type=int,
            default=0,
            help='Voice pitch (-50 to 50, default: 0)'
        )
        
        parser.add_argument(
            '--language',
            default='en',
            help='Language code for gTTS (default: en)'
        )
        
        parser.add_argument(
            '--voice-id',
            help='Specific voice ID (for supported engines)'
        )
        
        # Output configuration
        parser.add_argument(
            '--audio-format',
            choices=['mp3', 'wav'],
            default='mp3',
            help='Audio output format (default: mp3)'
        )
        
        parser.add_argument(
            '--video-resolution',
            default='1920x1080',
            help='Video resolution (default: 1920x1080)'
        )
        
        parser.add_argument(
            '--fps',
            type=int,
            default=24,
            help='Video frame rate (default: 24)'
        )
        
        # Feature options
        parser.add_argument(
            '--subtitles',
            action='store_true',
            help='Include subtitles in video output'
        )
        
        parser.add_argument(
            '--background-image',
            type=Path,
            help='Custom background image for video'
        )
        
        parser.add_argument(
            '--preview',
            action='store_true',
            help='Preview PDF text without conversion'
        )
        
        parser.add_argument(
            '--separate-pages',
            action='store_true',
            help='Create separate audio files for each page'
        )
        
        # PDF processing options
        parser.add_argument(
            '--extraction-method',
            choices=['auto', 'pymupdf', 'pdfplumber'],
            default='auto',
            help='PDF text extraction method (default: auto)'
        )
        
        # Utility options
        parser.add_argument(
            '--list-voices',
            action='store_true',
            help='List available TTS engines and exit'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--clean-temp',
            action='store_true',
            help='Clean temporary files after processing'
        )
        
        return parser
    
    def validate_args(self, args) -> bool:
        """Validate command-line arguments."""
        # Check input file
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            return False
        
        if not args.input.suffix.lower() == '.pdf':
            logger.error(f"Input file must be a PDF: {args.input}")
            return False
        
        # Validate voice speed
        if not 0.5 <= args.voice_speed <= 2.0:
            logger.error("Voice speed must be between 0.5 and 2.0")
            return False
        
        # Validate voice pitch
        if not -50 <= args.voice_pitch <= 50:
            logger.error("Voice pitch must be between -50 and 50")
            return False
        
        # Validate video resolution
        try:
            width, height = map(int, args.video_resolution.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError
        except ValueError:
            logger.error("Invalid video resolution format. Use WIDTHxHEIGHT (e.g., 1920x1080)")
            return False
        
        # Check background image if provided
        if args.background_image and not args.background_image.exists():
            logger.error(f"Background image not found: {args.background_image}")
            return False
        
        return True
    
    def setup_configuration(self, args) -> None:
        """Setup configuration from command-line arguments."""
        # Voice configuration
        config.update_voice_config(
            engine=args.voice_engine,
            speed=args.voice_speed,
            pitch=args.voice_pitch,
            language=args.language,
            voice_id=args.voice_id
        )
        
        # Output configuration
        width, height = map(int, args.video_resolution.split('x'))
        config.update_output_config(
            audio_format=args.audio_format,
            video_resolution=(width, height),
            fps=args.fps
        )
        
        # Set background image
        if args.background_image:
            config.paths.background_image = args.background_image
    
    def list_voices(self) -> None:
        """List available TTS engines and their status."""
        print("\nAvailable TTS Engines:")
        print("=" * 50)
        
        available_engines = self.tts_manager.get_available_engines()
        
        for engine_name, engine in self.tts_manager.engines.items():
            status = "✓ Available" if engine.is_available() else "✗ Not Available"
            print(f"{engine_name:12} - {status}")
            
            if engine_name == 'pyttsx3' and engine.is_available():
                try:
                    voices = engine.engine.getProperty('voices')
                    if voices:
                        print(f"             Voices: {len(voices)} available")
                        for i, voice in enumerate(voices[:3]):  # Show first 3
                            print(f"             - {voice.id}")
                        if len(voices) > 3:
                            print(f"             ... and {len(voices) - 3} more")
                except:
                    pass
        
        print(f"\nRecommended engine: {available_engines[0] if available_engines else 'None available'}")
    
    def preview_pdf(self, pdf_path: Path, extraction_method: str) -> None:
        """Preview PDF content without conversion."""
        print(f"\nPreviewing PDF: {pdf_path}")
        print("=" * 50)
        
        try:
            # Extract text
            pages = self.pdf_processor.extract_text(pdf_path, extraction_method)
            
            if not pages:
                print("No text could be extracted from this PDF.")
                return
            
            # Show statistics
            stats = self.pdf_processor.get_text_stats(pages)
            print(f"Pages: {stats['total_pages']}")
            print(f"Characters: {stats['total_characters']:,}")
            print(f"Words: {stats['total_words']:,}")
            print(f"Average words per page: {stats['average_words_per_page']}")
            
            # Show preview of first page
            print(f"\nFirst page preview:")
            print("-" * 30)
            first_page_text = pages[0]['text']
            preview_text = first_page_text[:500]
            if len(first_page_text) > 500:
                preview_text += "..."
            print(preview_text)
            
            # Estimate processing time
            total_chars = stats['total_characters']
            estimated_minutes = total_chars / 1000  # Rough estimate
            print(f"\nEstimated processing time: {estimated_minutes:.1f} minutes")
            
        except Exception as e:
            logger.error(f"Failed to preview PDF: {str(e)}")
    
    def convert_pdf(self, args) -> bool:
        """Main conversion function."""
        try:
            # Extract text from PDF
            logger.info(f"Extracting text from {args.input}")
            pages = self.pdf_processor.extract_text(args.input, args.extraction_method)
            
            if not pages:
                logger.error("No text extracted from PDF")
                return False
            
            # Show statistics
            stats = self.pdf_processor.get_text_stats(pages)
            logger.info(f"Extracted {stats['total_pages']} pages, {stats['total_words']} words")
            
            # Determine output paths
            base_name = args.input.stem
            audio_output_dir = args.output_dir / "audio_output"
            video_output_dir = args.output_dir / "video_output"
            
            self.file_manager.ensure_directory(audio_output_dir)
            self.file_manager.ensure_directory(video_output_dir)
            
            audio_files = []
            video_file = None
            
            # Generate audio
            if args.output_type in ['audio', 'both']:
                logger.info("Generating audio...")
                
                if args.separate_pages:
                    # Create separate audio files for each page
                    audio_files = self.tts_manager.synthesize_pages(
                        pages, audio_output_dir, config.voice, base_name
                    )
                else:
                    # Combine all text into one audio file
                    combined_text = " ".join(page['text'] for page in pages)
                    combined_audio_path = audio_output_dir / f"{base_name}_complete.{config.output.audio_format}"
                    
                    if self.tts_manager.synthesize_text(combined_text, combined_audio_path, config.voice):
                        audio_files = [combined_audio_path]
                
                if audio_files:
                    logger.info(f"Generated {len(audio_files)} audio file(s)")
                else:
                    logger.error("Failed to generate audio")
                    return False
            
            # Generate video
            if args.output_type in ['video', 'both'] and audio_files:
                logger.info("Generating video...")
                
                video_file = video_output_dir / f"{base_name}_complete.{config.output.video_format}"
                
                if args.separate_pages and len(audio_files) > 1:
                    # Create video from multiple audio files
                    success = self.video_generator.create_video_from_pages(
                        pages, audio_files, video_file,
                        background_image=config.paths.background_image,
                        config=config.output,
                        include_subtitles=args.subtitles
                    )
                else:
                    # Create video from single audio file
                    if args.subtitles:
                        # Generate subtitles
                        from moviepy.editor import AudioFileClip
                        audio_duration = AudioFileClip(str(audio_files[0])).duration
                        
                        combined_text = " ".join(page['text'] for page in pages)
                        subtitles = self.subtitle_generator.create_subtitles_from_text(
                            combined_text, audio_duration
                        )
                        
                        # Convert to format expected by video generator
                        subtitle_data = [
                            {
                                'start': sub.start_time,
                                'end': sub.end_time,
                                'text': sub.text
                            }
                            for sub in subtitles
                        ]
                        
                        success = self.video_generator.create_video_with_subtitles(
                            audio_files[0], video_file, subtitle_data,
                            background_image=config.paths.background_image,
                            config=config.output
                        )
                    else:
                        success = self.video_generator.create_video_from_audio(
                            audio_files[0], video_file,
                            background_image=config.paths.background_image,
                            config=config.output
                        )
                
                if not success:
                    logger.error("Failed to generate video")
                    video_file = None
            
            # Generate subtitle files if requested
            if args.subtitles and audio_files:
                logger.info("Generating subtitle files...")
                
                subtitle_dir = args.output_dir / "subtitles"
                self.file_manager.ensure_directory(subtitle_dir)
                
                # Get audio duration for subtitle timing
                from moviepy.editor import AudioFileClip
                if args.separate_pages:
                    # Create subtitles for each page
                    for i, (page, audio_file) in enumerate(zip(pages, audio_files)):
                        audio_duration = AudioFileClip(str(audio_file)).duration
                        subtitles = self.subtitle_generator.create_subtitles_from_text(
                            page['text'], audio_duration
                        )
                        
                        srt_path = subtitle_dir / f"{base_name}_page_{i+1:03d}.srt"
                        self.subtitle_generator.export_srt(subtitles, srt_path)
                else:
                    # Create subtitles for combined audio
                    audio_duration = AudioFileClip(str(audio_files[0])).duration
                    combined_text = " ".join(page['text'] for page in pages)
                    subtitles = self.subtitle_generator.create_subtitles_from_text(
                        combined_text, audio_duration
                    )
                    
                    srt_path = subtitle_dir / f"{base_name}_complete.srt"
                    self.subtitle_generator.export_srt(subtitles, srt_path)
            
            # Organize output files
            organized_files = self.file_manager.organize_output_files(
                base_name, audio_files, video_file, args.output_dir
            )
            
            # Create project manifest
            metadata = {
                'project_name': base_name,
                'source_pdf': str(args.input),
                'total_pages': len(pages),
                'total_audio_files': len(audio_files),
                'has_video': video_file is not None,
                'voice_config': config.get_config_dict()['voice'],
                'output_config': config.get_config_dict()['output'],
                'audio_files': audio_files,
                'video_file': video_file
            }
            
            self.file_manager.create_project_manifest(
                organized_files['project_dir'], metadata
            )
            
            # Summary
            print("\n" + "=" * 50)
            print("CONVERSION COMPLETE")
            print("=" * 50)
            print(f"Project directory: {organized_files['project_dir']}")
            print(f"Audio files: {len(audio_files)}")
            if video_file:
                print(f"Video file: {video_file.name}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return False
        finally:
            if args.clean_temp:
                self.file_manager.clean_temp_files()
    
    def run(self) -> int:
        """Main entry point for the CLI."""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Setup logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Handle special commands
        if args.list_voices:
            self.list_voices()
            return 0
        
        # Validate arguments
        if not self.validate_args(args):
            return 1
        
        # Setup configuration
        self.setup_configuration(args)
        
        # Handle preview
        if args.preview:
            self.preview_pdf(args.input, args.extraction_method)
            return 0
        
        # Main conversion
        success = self.convert_pdf(args)
        return 0 if success else 1

def main():
    """CLI entry point."""
    cli = PDFConverterCLI()
    return cli.run()

if __name__ == '__main__':
    sys.exit(main())