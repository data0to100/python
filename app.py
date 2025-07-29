import streamlit as st
import PyPDF2
import pdfplumber
import requests
import json
import os
from typing import List, Dict, Any
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF to Video Creator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

class PDFProcessor:
    """Class to handle PDF processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Try with PyPDF2 first
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # If PyPDF2 doesn't extract much text, try pdfplumber
            if len(text.strip()) < 100:
                pdf_file.seek(0)  # Reset file pointer
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

class AISummarizer:
    """Class to handle AI-powered text summarization"""
    
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model"""
        try:
            with st.spinner("Loading AI model for summarization..."):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqGeneration.from_pretrained(self.model_name)
                
                self.pipeline = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
        except Exception as e:
            st.error(f"Error loading AI model: {str(e)}")
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Summarize the given text using AI"""
        if not self.pipeline:
            return "Error: AI model not loaded"
        
        try:
            # Split text into chunks if it's too long
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_text(text)
            
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only summarize meaningful chunks
                    summary = self.pipeline(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)
            
            return " ".join(summaries)
        except Exception as e:
            st.error(f"Error during summarization: {str(e)}")
            return text[:500] + "..." if len(text) > 500 else text

class RenderForestAPI:
    """Class to handle RenderForest API interactions"""
    
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
                "duration": 30,  # Default duration in seconds
                "scenes": [
                    {
                        "type": "text",
                        "text": script[:500],  # Limit text length
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
    pdf_processor = PDFProcessor()
    ai_summarizer = AISummarizer()
    renderforest = RenderForestAPI()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Model Settings
        st.subheader("AI Summarization Settings")
        max_length = st.slider("Max Summary Length", 50, 300, 150)
        min_length = st.slider("Min Summary Length", 10, 100, 50)
        
        # RenderForest Settings
        st.subheader("Video Settings")
        template_id = st.text_input("Template ID", value="1")
        video_title = st.text_input("Video Title", value="Webinar Summary")
        
        # API Key input
        st.subheader("API Configuration")
        renderforest_key = st.text_input("RenderForest API Key", type="password", help="Enter your RenderForest API key")
        if renderforest_key:
            os.environ["RENDERFOREST_API_KEY"] = renderforest_key
    
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
                        summary = ai_summarizer.summarize_text(
                            extracted_text, 
                            max_length=max_length, 
                            min_length=min_length
                        )
                    
                    st.markdown('<h3 class="section-header">📋 AI Summary</h3>', unsafe_allow_html=True)
                    st.text_area("Summary", summary, height=150)
                    
                    # Store summary in session state
                    st.session_state['summary'] = summary
                    st.session_state['original_text'] = extracted_text
                    
                    st.success("✅ Summary generated successfully!")
    
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
        <p>Built with ❤️ using Streamlit, AI, and RenderForest</p>
        <p>Upload PDF → Extract Script → AI Summarize → Create Video</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()