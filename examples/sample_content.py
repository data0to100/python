#!/usr/bin/env python3
"""
Sample Content Generation Demo

This script demonstrates the PDF content generation workflow
using example content to show how the system works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.content_generator import ContentGenerator, ContentGenerationConfig

def main():
    """Demonstrate the content generation workflow with sample content."""
    
    # Sample PDF content (simulating extracted text)
    sample_text = """
    Artificial Intelligence in Modern Business: A Comprehensive Guide
    
    Introduction
    Artificial Intelligence (AI) has revolutionized the way businesses operate in the 21st century. 
    From automating routine tasks to providing sophisticated data analytics, AI technologies are 
    reshaping industries across the globe. This comprehensive guide explores the practical 
    applications, benefits, and challenges of implementing AI solutions in modern business environments.
    
    Key Applications
    Machine learning algorithms are being used for predictive analytics, helping companies forecast 
    market trends and customer behavior. Natural language processing enables automated customer 
    service through chatbots and virtual assistants. Computer vision technology is transforming 
    quality control in manufacturing and enabling autonomous vehicle development.
    
    Benefits and ROI
    Companies implementing AI solutions report significant improvements in operational efficiency, 
    with automation reducing manual work by up to 40%. Customer satisfaction scores increase due 
    to personalized experiences and faster response times. Cost savings are substantial, with many 
    organizations seeing ROI within 12-18 months of implementation.
    
    Implementation Challenges
    Despite the benefits, AI implementation faces several obstacles. Data quality and availability 
    remain primary concerns, as AI systems require large amounts of clean, relevant data. Skills 
    gaps in the workforce necessitate significant training investments. Ethical considerations 
    around bias and privacy must be carefully addressed.
    
    Future Outlook
    The future of AI in business looks promising, with emerging technologies like quantum computing 
    and advanced neural networks opening new possibilities. Integration with IoT devices will 
    create more comprehensive business intelligence systems. However, regulatory frameworks and 
    ethical guidelines will continue to evolve alongside these technological advances.
    
    Conclusion
    AI adoption in business is not just a trend but a fundamental shift in how organizations operate. 
    Companies that embrace AI technologies strategically, while addressing implementation challenges 
    thoughtfully, will gain significant competitive advantages in the digital economy.
    """
    
    print("🎥 PDF Content Generation Workflow Demo")
    print("=" * 60)
    
    # Create configuration
    config = ContentGenerationConfig(
        max_script_words=90,
        key_points_count=3,
        key_point_max_words=12,
        voice_style="friendly",
        language="American English",
        avatar_style="Professional",
        image_style="Infographic-style"
    )
    
    print(f"📄 Sample Content: AI in Modern Business Guide")
    print(f"⚙️  Configuration: {config.max_script_words} words, {config.voice_style} voice")
    print()
    
    # Generate workflow
    generator = ContentGenerator(config)
    workflow = generator.generate_content_workflow(sample_text)
    
    # Display results
    print("🚀 Generated AI Prompts:")
    print("=" * 60)
    
    print("\n1️⃣  PDF SUMMARIZATION PROMPT")
    print("-" * 40)
    print(workflow['1_pdf_summarization_prompt'])
    
    print("\n2️⃣  KEY POINTS EXTRACTION PROMPT")
    print("-" * 40)
    print(workflow['2_key_points_extraction_prompt'])
    
    print("\n3️⃣  VOICEOVER PROMPT (ElevenLabs)")
    print("-" * 40)
    print(workflow['3_voiceover_prompt'])
    
    print("\n4️⃣  AVATAR VIDEO PROMPT (D-ID)")
    print("-" * 40)
    print(workflow['4_avatar_video_prompt'])
    
    print("\n5️⃣  IMAGE GENERATION PROMPT (Midjourney/DALL-E)")
    print("-" * 40)
    print(workflow['5_image_generation_prompt'])
    
    print("\n📊 EXTRACTED CONCEPTS")
    print("-" * 40)
    concepts = workflow['extracted_concepts'][:10]  # Show top 10
    print(f"Top concepts: {', '.join(concepts)}")
    
    print("\n✅ WORKFLOW COMPLETE!")
    print("=" * 60)
    print("📋 Next Steps:")
    print("1. Copy each prompt to your respective AI tools")
    print("2. Generate content using the prompts")
    print("3. Combine assets for final video production")
    print("4. Review and refine as needed")
    
    # Generate markdown report
    markdown_content = generator.create_markdown_report(workflow, "Sample_AI_Business_Guide.pdf")
    
    # Save to examples directory
    output_path = Path(__file__).parent / "sample_workflow_report.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n📝 Sample report saved: {output_path}")
    print("\nThis demonstrates the complete workflow for any PDF!")

if __name__ == '__main__':
    main()