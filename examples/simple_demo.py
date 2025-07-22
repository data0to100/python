#!/usr/bin/env python3
"""
Simple Content Generation Demo (No Dependencies)

This script demonstrates the PDF content generation workflow
using your exact prompts with example content.
"""

def generate_pdf_summary_prompt(text, max_words=90):
    """Generate the PDF summarization prompt."""
    return f'''Summarize this entire PDF in a short script of no more than {max_words} words.

PDF Content:
{text}

Requirements:
- Maximum {max_words} words
- Clear and engaging script format
- Suitable for voiceover narration
- Capture the main message and key insights'''

def generate_key_points_prompt(text, num_points=3, max_words_per_point=12):
    """Generate the key points extraction prompt."""
    return f'''Give me the top {num_points} key takeaways in under {max_words_per_point} words each.

PDF Content:
{text}

Format each takeaway as a concise bullet point, maximum {max_words_per_point} words per point.'''

def generate_voiceover_prompt(script, language="American English"):
    """Generate the ElevenLabs voiceover prompt."""
    return f'''Generate a friendly and clear narration of this script in {language}:
'{script}' '''

def generate_avatar_prompt(style="Professional"):
    """Generate the D-ID avatar video prompt."""
    return f'''Create a talking avatar using this audio and match lip sync:
Style: {style}'''

def generate_image_prompt(concepts):
    """Generate the Midjourney/DALL-E image prompt."""
    concept_text = ", ".join(concepts[:3])
    return f'''Infographic-style background showing {concept_text}, with clean layout and white space.'''

def extract_key_concepts(text):
    """Extract key concepts from text (simplified version)."""
    # Simple keyword extraction
    words = text.lower().split()
    
    # Filter out common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Count word frequency
    word_count = {}
    for word in words:
        word = word.strip('.,!?()[]{}":;')
        if len(word) > 3 and word not in stop_words:
            word_count[word] = word_count.get(word, 0) + 1
    
    # Return top concepts
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:10]]

def main():
    """Demonstrate the complete content generation workflow."""
    
    # Sample PDF content
    sample_text = """
    Artificial Intelligence in Modern Business: A Comprehensive Guide
    
    Artificial Intelligence (AI) has revolutionized business operations in the 21st century. 
    From automating routine tasks to providing sophisticated data analytics, AI technologies 
    reshape industries globally. Machine learning algorithms enable predictive analytics, 
    helping companies forecast market trends and customer behavior. Natural language processing 
    powers automated customer service through chatbots and virtual assistants.
    
    Companies implementing AI solutions report significant operational efficiency improvements, 
    with automation reducing manual work by up to 40%. Customer satisfaction scores increase 
    due to personalized experiences and faster response times. Cost savings are substantial, 
    with many organizations seeing ROI within 12-18 months.
    
    Implementation challenges include data quality concerns, workforce skills gaps, and ethical 
    considerations around bias and privacy. The future looks promising with quantum computing 
    and advanced neural networks opening new possibilities. Companies embracing AI strategically 
    will gain significant competitive advantages in the digital economy.
    """
    
    print("🎥 PDF CONTENT GENERATION WORKFLOW")
    print("   Transform PDFs into AI-ready prompts for video creation")
    print("=" * 70)
    
    # Extract concepts
    concepts = extract_key_concepts(sample_text)
    
    print(f"📄 Sample Content: AI in Modern Business")
    print(f"🎯 Top Concepts: {', '.join(concepts[:5])}")
    print()
    
    # Generate all prompts using your exact specifications
    print("🚀 GENERATED AI PROMPTS:")
    print("=" * 70)
    
    # 1. PDF Summarization Prompt
    print("\n1️⃣  PDF SUMMARIZATION PROMPT")
    print("   > \"Summarize this entire PDF in a short script of no more than 90 words.\"")
    print("-" * 50)
    summary_prompt = generate_pdf_summary_prompt(sample_text)
    print(summary_prompt)
    
    # 2. Key Points Extraction
    print("\n\n2️⃣  KEY POINTS EXTRACTION PROMPT")
    print("   > \"Give me the top 3 key takeaways in under 12 words each.\"")
    print("-" * 50)
    key_points_prompt = generate_key_points_prompt(sample_text)
    print(key_points_prompt)
    
    # 3. Voiceover Prompt
    print("\n\n3️⃣  VOICEOVER PROMPT (ElevenLabs)")
    print("   > \"Generate a friendly and clear narration of this script in American English\"")
    print("-" * 50)
    placeholder_script = "[AI-generated summary script - max 90 words]"
    voiceover_prompt = generate_voiceover_prompt(placeholder_script)
    print(voiceover_prompt)
    
    # 4. Avatar Video Prompt
    print("\n\n4️⃣  AVATAR VIDEO PROMPT (D-ID)")
    print("   > \"Create a talking avatar using this audio and match lip sync\"")
    print("-" * 50)
    avatar_prompt = generate_avatar_prompt("Professional")
    print(avatar_prompt)
    
    # 5. Image Generation Prompt
    print("\n\n5️⃣  IMAGE GENERATION PROMPT (Midjourney/DALL-E)")
    print("   > \"Infographic-style background showing [concept], with clean layout and white space.\"")
    print("-" * 50)
    image_prompt = generate_image_prompt(concepts)
    print(image_prompt)
    
    print("\n\n✅ COMPLETE WORKFLOW GENERATED!")
    print("=" * 70)
    print("📋 WORKFLOW STEPS:")
    print("1. Extract PDF Content → Use PDF processor to get full text")
    print("2. Generate Summary → Apply summarization prompt to AI model")
    print("3. Extract Key Points → Generate bullet points from content")
    print("4. Create Voiceover → Use ElevenLabs with generated script")
    print("5. Generate Avatar Video → Use D-ID with audio and style preferences")
    print("6. Create Background Image → Use Midjourney/DALL-E with concept keywords")
    print("7. Combine Assets → Merge avatar, background, and audio for final video")
    
    print("\n🎬 YOUR EXACT PROMPTS IMPLEMENTED:")
    print("✓ PDF Summarization: 90 words maximum")
    print("✓ Key Points: Top 3 takeaways, 12 words each")
    print("✓ Voiceover: Friendly, clear, American English")
    print("✓ Avatar: Professional style with lip sync")
    print("✓ Images: Infographic-style with clean layout")
    
    print("\n🚀 READY TO USE!")
    print("Copy each prompt above to your AI tools and start creating!")

if __name__ == '__main__':
    main()