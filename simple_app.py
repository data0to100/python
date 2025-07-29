#!/usr/bin/env python3
"""
Simplified PDF to Video Creator
A lightweight version that works with minimal dependencies.
"""

import streamlit as st
import os
import json
import requests
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="PDF to Video Creator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimplePDFProcessor:
    """Simple PDF processor that works with basic dependencies"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Try to import PyPDF2
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            except ImportError:
                st.warning("PyPDF2 not available. Please install it with: pip install PyPDF2")
                return ""
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

class SimpleAISummarizer:
    """Simple AI summarizer using basic text processing"""
    
    @staticmethod
    def summarize_text(text: str, max_length: int = 150) -> str:
        """Create a simple summary by extracting key sentences"""
        if not text:
            return ""
        
        # Simple summarization: take first few sentences
        sentences = text.split('.')
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and current_length + len(sentence) < max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        summary = '. '.join(summary_sentences)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary

class SimpleRenderForestAPI:
    """Simple RenderForest API wrapper"""
    
    def __init__(self):
        self.api_key = os.getenv("RENDERFOREST_API_KEY")
        self.base_url = "https://api.renderforest.com/v1"
    
    def create_video(self, script: str, title: str, template_id: str = "1") -> Dict[str, Any]:
        """Create a video using RenderForest API"""
        if not self.api_key:
            return {"error": "RenderForest API key not found. Please set RENDERFOREST_API_KEY environment variable."}
        
        try:
            # Prepare the video creation request
            payload = {
                "template": template_id,
                "name": title,
                "duration": 30,
                "scenes": [
                    {
                        "type": "text",
                        "text": script[:500],
                        "duration": 5
                    }
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                f"{self.base_url}/videos",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Error creating video: {str(e)}"}

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 PDF to Video Creator</h1>', unsafe_allow_html=True)
    st.markdown("Extract scripts from PDF webinars, summarize with AI, and create videos with RenderForest")
    
    # Initialize classes
    pdf_processor = SimplePDFProcessor()
    ai_summarizer = SimpleAISummarizer()
    renderforest = SimpleRenderForestAPI()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Model Settings
        st.subheader("AI Summarization Settings")
        max_length = st.slider("Max Summary Length", 50, 300, 150)
        
        # RenderForest Settings
        st.subheader("Video Settings")
        template_id = st.text_input("Template ID", value="1")
        video_title = st.text_input("Video Title", value="Webinar Summary")
        
        # API Key input
        st.subheader("API Configuration")
        renderforest_key = st.text_input("RenderForest API Key", type="password", help="Enter your RenderForest API key")
        if renderforest_key:
            os.environ["RENDERFOREST_API_KEY"] = renderforest_key
        
        # Installation instructions
        st.subheader("📦 Installation")
        st.markdown("""
        If you encounter missing dependencies:
        ```bash
        pip install streamlit PyPDF2 pdfplumber requests python-dotenv
        ```
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📄 Upload PDF</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file containing webinar script or content"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.write("**File Details:**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                extracted_text = pdf_processor.extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.success(f"✅ Extracted {len(extracted_text)} characters from PDF")
                
                # Display extracted text
                with st.expander("📝 View Extracted Text"):
                    st.text_area("Extracted Text", extracted_text, height=200)
                
                # Summarize text
                if st.button("🤖 Generate AI Summary", type="primary"):
                    with st.spinner("Generating AI summary..."):
                        summary = ai_summarizer.summarize_text(extracted_text, max_length=max_length)
                    
                    st.markdown('<h3 class="section-header">📋 AI Summary</h3>', unsafe_allow_html=True)
                    st.text_area("Summary", summary, height=150)
                    
                    # Store summary in session state
                    st.session_state['summary'] = summary
                    st.session_state['original_text'] = extracted_text
                    
                    st.success("✅ Summary generated successfully!")
            else:
                st.error("❌ Failed to extract text from PDF. Please try a different file.")
    
    with col2:
        st.markdown('<h2 class="section-header">🎬 Create Video</h2>', unsafe_allow_html=True)
        
        if 'summary' in st.session_state:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**Available Summary:**")
            st.write(st.session_state['summary'][:200] + "...")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Video creation options
            st.subheader("Video Creation Options")
            
            # Customize video script
            video_script = st.text_area(
                "Video Script (modify if needed)",
                value=st.session_state['summary'],
                height=100,
                help="Edit the script that will be used in the video"
            )
            
            # Video settings
            video_duration = st.slider("Video Duration (seconds)", 10, 120, 30)
            
            if st.button("🎬 Create Video with RenderForest", type="primary"):
                if not os.getenv("RENDERFOREST_API_KEY"):
                    st.error("❌ RenderForest API key not found. Please enter it in the sidebar.")
                else:
                    with st.spinner("Creating video with RenderForest..."):
                        result = renderforest.create_video(
                            script=video_script,
                            title=video_title,
                            template_id=template_id
                        )
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success("✅ Video creation request sent to RenderForest!")
                        st.json(result)
                        
                        # Display video info
                        if "id" in result:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.write(f"**Video ID:** {result['id']}")
                            st.write("Check your RenderForest dashboard for the final video.")
                            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("📝 Upload a PDF and generate a summary first to create a video.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit and RenderForest</p>
        <p>Upload PDF → Extract Script → AI Summarize → Create Video</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()