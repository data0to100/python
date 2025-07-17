#!/usr/bin/env python3
"""
Create a sample PDF for testing the PDF Script to Speech/Video Converter.

This script generates a sample PDF with script content that can be used
to test the conversion functionality.
"""

import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
except ImportError:
    print("❌ ReportLab is required to create sample PDFs")
    print("Install with: pip install reportlab")
    sys.exit(1)

def create_sample_script_content():
    """Create sample script content."""
    return [
        {
            "title": "The Digital Age",
            "content": """
            Welcome to our presentation about the digital age. In today's rapidly evolving world, 
            technology has become an integral part of our daily lives. From smartphones to artificial 
            intelligence, we are witnessing unprecedented changes in how we communicate, work, and 
            interact with the world around us.
            
            The digital revolution has transformed every aspect of human society. We now live in 
            an interconnected world where information travels at the speed of light, and geographical 
            boundaries have become less relevant than ever before.
            """
        },
        {
            "title": "Impact on Communication",
            "content": """
            One of the most significant impacts of the digital age has been on communication. 
            Social media platforms, instant messaging, and video conferencing have revolutionized 
            how we connect with others. We can now communicate with people across the globe in 
            real-time, sharing ideas, experiences, and emotions instantaneously.
            
            This transformation has brought both opportunities and challenges. While we can maintain 
            relationships across vast distances, we also face issues like digital addiction, 
            privacy concerns, and the spread of misinformation.
            """
        },
        {
            "title": "The Future of Work",
            "content": """
            The digital age has also fundamentally changed the nature of work. Remote work, which 
            was once a rare privilege, has become commonplace. Cloud computing, collaboration tools, 
            and virtual meetings have made it possible for teams to work together regardless of 
            their physical location.
            
            Artificial intelligence and automation are reshaping entire industries. While these 
            technologies promise increased efficiency and new opportunities, they also raise 
            questions about the future of employment and the skills workers will need to remain 
            relevant in the digital economy.
            """
        },
        {
            "title": "Challenges and Opportunities",
            "content": """
            As we navigate this digital landscape, we face numerous challenges. Cybersecurity threats 
            are becoming more sophisticated, requiring constant vigilance and adaptation. The digital 
            divide continues to separate those with access to technology from those without, creating 
            new forms of inequality.
            
            However, the opportunities are equally significant. Digital technologies are enabling 
            breakthrough solutions in healthcare, education, environmental protection, and scientific 
            research. We have the potential to solve some of humanity's greatest challenges through 
            innovative applications of technology.
            """
        },
        {
            "title": "Conclusion",
            "content": """
            The digital age is not just about technology; it's about human adaptation and evolution. 
            As we continue to integrate digital tools into our lives, we must remain mindful of 
            their impact on our society, relationships, and well-being.
            
            The key to thriving in the digital age is not just technical literacy, but also 
            emotional intelligence, critical thinking, and the ability to maintain human connections 
            in an increasingly digital world. By embracing these principles, we can harness the 
            power of technology to create a better future for all.
            
            Thank you for your attention. The digital age is here, and together, we can shape 
            its trajectory for the benefit of humanity.
            """
        }
    ]

def create_sample_pdf():
    """Create a sample PDF with script content."""
    # Ensure output directory exists
    output_dir = Path(__file__).parent.parent / "pdf_scripts"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "sample_script.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    normal_style = styles['Normal']
    
    # Build content
    story = []
    
    # Add main title
    story.append(Paragraph("The Digital Age: A Script Sample", title_style))
    story.append(Spacer(1, 20))
    
    # Add subtitle
    story.append(Paragraph("Sample Script for PDF to Speech/Video Converter", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    # Add content sections
    content = create_sample_script_content()
    
    for section in content:
        # Add section title
        story.append(Paragraph(section['title'], styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add section content
        # Split content into paragraphs
        paragraphs = section['content'].strip().split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                story.append(Paragraph(paragraph, normal_style))
                story.append(Spacer(1, 12))
        
        story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    
    return output_file

def main():
    """Main function to create sample PDF."""
    print("📄 Creating sample PDF script...")
    
    try:
        pdf_file = create_sample_pdf()
        print(f"✅ Sample PDF created: {pdf_file}")
        print(f"📊 File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        
        print("\nYou can now test the converter with:")
        print(f"  python main.py --interface cli --input {pdf_file} --preview")
        print(f"  python main.py --interface cli --input {pdf_file} --output-type audio")
        print(f"  python examples/sample_script.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating sample PDF: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())