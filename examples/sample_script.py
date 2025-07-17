#!/usr/bin/env python3
"""
Sample script demonstrating PDF Script to Speech/Video Converter usage.

This script shows how to use the library programmatically without the CLI or web interface.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import config
from core.pdf_processor import PDFProcessor
from core.tts_engine import TTSManager
from core.video_generator import VideoGenerator
from utils.file_manager import FileManager
from utils.subtitle_generator import SubtitleGenerator

def main():
    """Demonstrate basic usage of the PDF converter."""
    
    # Initialize components
    pdf_processor = PDFProcessor()
    tts_manager = TTSManager()
    video_generator = VideoGenerator()
    file_manager = FileManager()
    subtitle_generator = SubtitleGenerator()
    
    # Check if we have a sample PDF
    sample_pdf = Path("pdf_scripts/sample.pdf")
    if not sample_pdf.exists():
        print("No sample PDF found. Please place a PDF file at: pdf_scripts/sample.pdf")
        return 1
    
    try:
        print("🔍 Extracting text from PDF...")
        # Extract text from PDF
        pages = pdf_processor.extract_text(sample_pdf)
        
        if not pages:
            print("❌ No text could be extracted from the PDF")
            return 1
        
        # Show statistics
        stats = pdf_processor.get_text_stats(pages)
        print(f"📊 Extracted {stats['total_pages']} pages, {stats['total_words']} words")
        
        # Configure voice settings
        print("🎙️ Configuring voice settings...")
        config.update_voice_config(
            engine="pyttsx3",  # Use offline TTS
            speed=1.0,
            language="en"
        )
        
        # Generate audio
        print("🎵 Generating audio...")
        audio_output_dir = Path("audio_output")
        file_manager.ensure_directory(audio_output_dir)
        
        # Combine all text into one audio file
        combined_text = " ".join(page['text'] for page in pages)
        audio_path = audio_output_dir / "sample_complete.mp3"
        
        success = tts_manager.synthesize_text(combined_text, audio_path, config.voice)
        
        if success:
            print(f"✅ Audio generated: {audio_path}")
            
            # Generate video with subtitles
            print("🎬 Generating video with subtitles...")
            video_output_dir = Path("video_output")
            file_manager.ensure_directory(video_output_dir)
            
            video_path = video_output_dir / "sample_complete.mp4"
            
            # Get audio duration for subtitle timing
            from moviepy.editor import AudioFileClip
            audio_duration = AudioFileClip(str(audio_path)).duration
            
            # Generate subtitles
            subtitles = subtitle_generator.create_subtitles_from_text(
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
            
            # Create video with subtitles
            video_success = video_generator.create_video_with_subtitles(
                audio_path, video_path, subtitle_data,
                config=config.output
            )
            
            if video_success:
                print(f"✅ Video generated: {video_path}")
                
                # Export subtitle file
                subtitle_path = video_output_dir / "sample_complete.srt"
                subtitle_generator.export_srt(subtitles, subtitle_path)
                print(f"✅ Subtitles exported: {subtitle_path}")
                
            else:
                print("❌ Failed to generate video")
        else:
            print("❌ Failed to generate audio")
        
        print("\n🎉 Conversion complete!")
        print("\nGenerated files:")
        print(f"  Audio: {audio_path}")
        if video_success:
            print(f"  Video: {video_path}")
            print(f"  Subtitles: {subtitle_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())