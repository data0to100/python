#!/usr/bin/env python3
"""
Demo script for PDF to Video Creator
This script demonstrates how to use the application classes programmatically
and provides sample data for testing.
"""

import os
import tempfile
from app import PDFProcessor, AISummarizer, RenderForestAPI

def create_sample_pdf():
    """Create a sample PDF file for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            c = canvas.Canvas(tmp_file.name, pagesize=letter)
            width, height = letter
            
            # Add title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, height - 100, "Sample Webinar: Digital Marketing Strategies")
            
            # Add content
            c.setFont("Helvetica", 12)
            y_position = height - 150
            
            content = [
                "Introduction to Digital Marketing",
                "Digital marketing encompasses all marketing efforts that use electronic devices or the internet. Businesses leverage digital channels such as search engines, social media, email, and other websites to connect with current and prospective customers.",
                "",
                "Key Components of Digital Marketing:",
                "1. Search Engine Optimization (SEO)",
                "2. Content Marketing",
                "3. Social Media Marketing",
                "4. Email Marketing",
                "5. Pay-Per-Click Advertising",
                "",
                "Benefits of Digital Marketing:",
                "- Cost-effective compared to traditional marketing",
                "- Measurable results and analytics",
                "- Global reach and targeting capabilities",
                "- Real-time optimization and adjustments",
                "",
                "Best Practices:",
                "- Create valuable, relevant content",
                "- Optimize for mobile devices",
                "- Use data-driven insights",
                "- Maintain consistent branding",
                "- Engage with your audience regularly"
            ]
            
            for line in content:
                if y_position < 100:  # Start new page if needed
                    c.showPage()
                    y_position = height - 100
                
                c.drawString(100, y_position, line)
                y_position -= 20
            
            c.save()
            return tmp_file.name
            
    except ImportError:
        print("reportlab not available, creating text file instead")
        # Fallback to text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            content = """
Sample Webinar: Digital Marketing Strategies

Introduction to Digital Marketing
Digital marketing encompasses all marketing efforts that use electronic devices or the internet. Businesses leverage digital channels such as search engines, social media, email, and other websites to connect with current and prospective customers.

Key Components of Digital Marketing:
1. Search Engine Optimization (SEO)
2. Content Marketing
3. Social Media Marketing
4. Email Marketing
5. Pay-Per-Click Advertising

Benefits of Digital Marketing:
- Cost-effective compared to traditional marketing
- Measurable results and analytics
- Global reach and targeting capabilities
- Real-time optimization and adjustments

Best Practices:
- Create valuable, relevant content
- Optimize for mobile devices
- Use data-driven insights
- Maintain consistent branding
- Engage with your audience regularly
            """
            tmp_file.write(content.encode())
            return tmp_file.name

def demo_pdf_processing():
    """Demonstrate PDF processing functionality"""
    print("🔍 Testing PDF Processing...")
    
    # Create sample PDF
    sample_file = create_sample_pdf()
    print(f"✅ Created sample file: {sample_file}")
    
    # Test PDF processor
    pdf_processor = PDFProcessor()
    
    # Simulate file upload
    class MockFile:
        def __init__(self, filename):
            self.name = filename
            self.size = os.path.getsize(filename)
            self.type = "application/pdf"
        
        def seek(self, pos):
            pass
        
        def read(self):
            with open(self.name, 'rb') as f:
                return f.read()
    
    mock_file = MockFile(sample_file)
    
    # Extract text
    extracted_text = pdf_processor.extract_text_from_pdf(mock_file)
    print(f"✅ Extracted {len(extracted_text)} characters")
    print(f"📝 Sample text: {extracted_text[:100]}...")
    
    return extracted_text

def demo_ai_summarization(text):
    """Demonstrate AI summarization functionality"""
    print("\n🤖 Testing AI Summarization...")
    
    ai_summarizer = AISummarizer()
    
    # Test summarization
    summary = ai_summarizer.summarize_text(text, max_length=150, min_length=50)
    print(f"✅ Generated summary: {summary}")
    
    return summary

def demo_renderforest_integration(script):
    """Demonstrate RenderForest API integration"""
    print("\n🎬 Testing RenderForest Integration...")
    
    renderforest = RenderForestAPI()
    
    # Test video creation (will fail without API key, but shows the structure)
    result = renderforest.create_video(
        script=script,
        title="Demo Video",
        template_id="1"
    )
    
    if "error" in result:
        print(f"⚠️ Expected error (no API key): {result['error']}")
    else:
        print(f"✅ Video creation result: {result}")
    
    return result

def main():
    """Run the complete demo"""
    print("🎬 PDF to Video Creator - Demo")
    print("=" * 50)
    
    try:
        # Step 1: PDF Processing
        extracted_text = demo_pdf_processing()
        
        # Step 2: AI Summarization
        summary = demo_ai_summarization(extracted_text)
        
        # Step 3: RenderForest Integration
        demo_renderforest_integration(summary)
        
        print("\n✅ Demo completed successfully!")
        print("\nTo use the full application:")
        print("1. Run: streamlit run app.py")
        print("2. Open: http://localhost:8501")
        print("3. Upload a PDF and start creating videos!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()