"""
Streamlit web interface for PDF Script to Speech/Video Converter.

This module provides a user-friendly web interface for converting
PDF scripts to audio and video with real-time configuration.
"""

import streamlit as st
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import io

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config, VoiceConfig, OutputConfig
from core.pdf_processor import PDFProcessor
from core.tts_engine import TTSManager
from core.video_generator import VideoGenerator
from utils.file_manager import FileManager
from utils.subtitle_generator import SubtitleGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitApp:
    """Streamlit web application for PDF conversion."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.tts_manager = TTSManager()
        self.video_generator = VideoGenerator()
        self.file_manager = FileManager()
        self.subtitle_generator = SubtitleGenerator()
        
        # Initialize session state
        if 'pdf_pages' not in st.session_state:
            st.session_state.pdf_pages = None
        if 'conversion_complete' not in st.session_state:
            st.session_state.conversion_complete = False
        if 'output_files' not in st.session_state:
            st.session_state.output_files = {}
    
    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="PDF Script to Speech/Video Converter",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎬 PDF Script to Speech/Video Converter")
        st.markdown("""
        Convert your PDF scripts into natural-sounding audio and engaging videos with subtitles.
        Upload a PDF, configure your preferences, and let the magic happen!
        """)
    
    def render_sidebar(self) -> Dict:
        """Render sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # Voice Configuration
        st.sidebar.subheader("🎙️ Voice Settings")
        
        available_engines = self.tts_manager.get_available_engines()
        if not available_engines:
            st.sidebar.error("No TTS engines available!")
            return {}
        
        voice_engine = st.sidebar.selectbox(
            "TTS Engine",
            available_engines,
            help="Text-to-speech engine to use"
        )
        
        voice_speed = st.sidebar.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Speech speed multiplier"
        )
        
        if voice_engine == 'gtts':
            language = st.sidebar.selectbox(
                "Language",
                ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                help="Language for Google TTS"
            )
        else:
            language = "en"
        
        voice_pitch = st.sidebar.slider(
            "Voice Pitch",
            min_value=-50,
            max_value=50,
            value=0,
            help="Voice pitch adjustment (-50 to 50)"
        )
        
        # Output Configuration
        st.sidebar.subheader("📤 Output Settings")
        
        output_type = st.sidebar.selectbox(
            "Output Type",
            ["Audio Only", "Video Only", "Both Audio and Video"],
            index=0
        )
        
        audio_format = st.sidebar.selectbox(
            "Audio Format",
            ["mp3", "wav"],
            help="Audio output format"
        )
        
        if output_type in ["Video Only", "Both Audio and Video"]:
            video_resolution = st.sidebar.selectbox(
                "Video Resolution",
                ["1920x1080", "1280x720", "854x480"],
                help="Video resolution"
            )
            
            include_subtitles = st.sidebar.checkbox(
                "Include Subtitles",
                value=True,
                help="Add subtitles to video"
            )
            
            fps = st.sidebar.slider(
                "Frame Rate",
                min_value=15,
                max_value=60,
                value=24,
                help="Video frame rate (fps)"
            )
        else:
            video_resolution = "1920x1080"
            include_subtitles = False
            fps = 24
        
        # Advanced Options
        with st.sidebar.expander("🔧 Advanced Options"):
            separate_pages = st.checkbox(
                "Separate Audio per Page",
                value=False,
                help="Create separate audio files for each page"
            )
            
            extraction_method = st.selectbox(
                "PDF Extraction Method",
                ["auto", "pymupdf", "pdfplumber"],
                help="Method for extracting text from PDF"
            )
            
            background_image = st.file_uploader(
                "Custom Background Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a custom background for video"
            )
        
        return {
            'voice_engine': voice_engine,
            'voice_speed': voice_speed,
            'voice_pitch': voice_pitch,
            'language': language,
            'output_type': output_type,
            'audio_format': audio_format,
            'video_resolution': video_resolution,
            'include_subtitles': include_subtitles,
            'fps': fps,
            'separate_pages': separate_pages,
            'extraction_method': extraction_method,
            'background_image': background_image
        }
    
    def handle_pdf_upload(self, extraction_method: str):
        """Handle PDF file upload and processing."""
        uploaded_file = st.file_uploader(
            "📄 Upload PDF Script",
            type=['pdf'],
            help="Upload your PDF script file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)
            
            # Extract text
            try:
                with st.spinner("Extracting text from PDF..."):
                    pages = self.pdf_processor.extract_text(tmp_path, extraction_method)
                
                if pages:
                    st.session_state.pdf_pages = pages
                    st.session_state.pdf_path = tmp_path
                    
                    # Show PDF statistics
                    stats = self.pdf_processor.get_text_stats(pages)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Pages", stats['total_pages'])
                    with col2:
                        st.metric("Words", f"{stats['total_words']:,}")
                    with col3:
                        st.metric("Characters", f"{stats['total_characters']:,}")
                    with col4:
                        estimated_time = stats['total_characters'] / 1000
                        st.metric("Est. Time (min)", f"{estimated_time:.1f}")
                    
                    # Show preview
                    with st.expander("📖 Preview PDF Content"):
                        preview_text = pages[0]['text'][:1000]
                        if len(pages[0]['text']) > 1000:
                            preview_text += "..."
                        st.text_area(
                            "First page preview:",
                            value=preview_text,
                            height=200,
                            disabled=True
                        )
                    
                    return True
                else:
                    st.error("Could not extract text from the PDF. Please check the file format.")
                    return False
                    
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return False
            finally:
                # Clean up temporary file
                try:
                    tmp_path.unlink()
                except:
                    pass
        
        return False
    
    def update_configuration(self, settings: Dict):
        """Update global configuration from settings."""
        # Voice configuration
        config.update_voice_config(
            engine=settings['voice_engine'],
            speed=settings['voice_speed'],
            pitch=settings['voice_pitch'],
            language=settings['language']
        )
        
        # Output configuration
        width, height = map(int, settings['video_resolution'].split('x'))
        config.update_output_config(
            audio_format=settings['audio_format'],
            video_resolution=(width, height),
            fps=settings['fps']
        )
    
    def save_background_image(self, background_file) -> Optional[Path]:
        """Save uploaded background image to temporary file."""
        if background_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(background_file.getvalue())
                return Path(tmp_file.name)
        return None
    
    def perform_conversion(self, settings: Dict) -> bool:
        """Perform the PDF conversion with progress tracking."""
        if st.session_state.pdf_pages is None:
            st.error("Please upload a PDF file first.")
            return False
        
        try:
            pages = st.session_state.pdf_pages
            base_name = "converted_script"
            
            # Create temporary output directories
            temp_dir = Path(tempfile.mkdtemp())
            audio_dir = temp_dir / "audio"
            video_dir = temp_dir / "video"
            
            self.file_manager.ensure_directory(audio_dir)
            self.file_manager.ensure_directory(video_dir)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            audio_files = []
            video_file = None
            
            # Generate audio
            if settings['output_type'] in ["Audio Only", "Both Audio and Video"]:
                status_text.text("🎙️ Generating audio...")
                progress_bar.progress(10)
                
                if settings['separate_pages']:
                    # Create separate audio files for each page
                    for i, page in enumerate(pages):
                        progress = 10 + (i + 1) * 30 // len(pages)
                        progress_bar.progress(progress)
                        status_text.text(f"🎙️ Generating audio for page {i + 1}/{len(pages)}...")
                        
                        audio_path = audio_dir / f"page_{i+1:03d}.{settings['audio_format']}"
                        if self.tts_manager.synthesize_text(page['text'], audio_path, config.voice):
                            audio_files.append(audio_path)
                else:
                    # Combine all text into one audio file
                    combined_text = " ".join(page['text'] for page in pages)
                    audio_path = audio_dir / f"complete.{settings['audio_format']}"
                    
                    if self.tts_manager.synthesize_text(combined_text, audio_path, config.voice):
                        audio_files = [audio_path]
                
                if not audio_files:
                    st.error("Failed to generate audio files.")
                    return False
                
                progress_bar.progress(40)
            
            # Generate video
            if settings['output_type'] in ["Video Only", "Both Audio and Video"] and audio_files:
                status_text.text("🎬 Generating video...")
                progress_bar.progress(50)
                
                # Save background image if provided
                background_path = None
                if settings['background_image']:
                    background_path = self.save_background_image(settings['background_image'])
                    config.paths.background_image = background_path
                
                video_file = video_dir / f"complete.mp4"
                
                if settings['separate_pages'] and len(audio_files) > 1:
                    # Create video from multiple audio files
                    success = self.video_generator.create_video_from_pages(
                        pages, audio_files, video_file,
                        background_image=background_path,
                        config=config.output,
                        include_subtitles=settings['include_subtitles']
                    )
                else:
                    # Create video from single audio file
                    if settings['include_subtitles']:
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
                            background_image=background_path,
                            config=config.output
                        )
                    else:
                        success = self.video_generator.create_video_from_audio(
                            audio_files[0], video_file,
                            background_image=background_path,
                            config=config.output
                        )
                
                if not success:
                    st.error("Failed to generate video.")
                    video_file = None
                
                progress_bar.progress(80)
            
            # Store results in session state
            st.session_state.output_files = {
                'audio_files': audio_files,
                'video_file': video_file,
                'temp_dir': temp_dir
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Conversion complete!")
            
            return True
            
        except Exception as e:
            st.error(f"Conversion failed: {str(e)}")
            logger.error(f"Conversion error: {str(e)}")
            return False
    
    def display_results(self):
        """Display conversion results and download options."""
        if not st.session_state.output_files:
            return
        
        st.header("🎉 Conversion Results")
        
        output_files = st.session_state.output_files
        audio_files = output_files.get('audio_files', [])
        video_file = output_files.get('video_file')
        
        # Audio downloads
        if audio_files:
            st.subheader("🎵 Audio Files")
            
            for i, audio_file in enumerate(audio_files):
                if audio_file.exists():
                    file_info = self.file_manager.get_file_info(audio_file)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {audio_file.name} ({file_info['size_mb']} MB)")
                    with col2:
                        with open(audio_file, 'rb') as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=audio_file.name,
                                mime="audio/mpeg" if audio_file.suffix == '.mp3' else "audio/wav",
                                key=f"audio_{i}"
                            )
        
        # Video download
        if video_file and video_file.exists():
            st.subheader("🎬 Video File")
            
            file_info = self.file_manager.get_file_info(video_file)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"🎥 {video_file.name} ({file_info['size_mb']} MB)")
                
                # Display video preview
                try:
                    st.video(str(video_file))
                except:
                    st.info("Video preview not available, but file can be downloaded.")
            
            with col2:
                with open(video_file, 'rb') as f:
                    st.download_button(
                        label="Download Video",
                        data=f.read(),
                        file_name=video_file.name,
                        mime="video/mp4",
                        key="video_download"
                    )
    
    def render_main_content(self, settings: Dict):
        """Render main content area."""
        # PDF Upload and Processing
        pdf_uploaded = self.handle_pdf_upload(settings['extraction_method'])
        
        # Conversion Button
        if pdf_uploaded and st.session_state.pdf_pages:
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "🚀 Start Conversion",
                    type="primary",
                    use_container_width=True
                ):
                    self.update_configuration(settings)
                    
                    with st.spinner("Converting PDF..."):
                        success = self.perform_conversion(settings)
                    
                    if success:
                        st.session_state.conversion_complete = True
                        st.rerun()
        
        # Display Results
        if st.session_state.conversion_complete:
            st.markdown("---")
            self.display_results()
    
    def render_footer(self):
        """Render footer with information."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit, PyMuPDF, MoviePy, and various TTS engines.</p>
            <p>For best results, use clear, well-formatted PDF scripts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner."""
        self.setup_page()
        
        # Render sidebar configuration
        settings = self.render_sidebar()
        
        if not settings:
            st.error("Application configuration error. Please check TTS engines.")
            return
        
        # Render main content
        self.render_main_content(settings)
        
        # Render footer
        self.render_footer()

def main():
    """Main entry point for Streamlit app."""
    app = StreamlitApp()
    app.run()

if __name__ == '__main__':
    main()