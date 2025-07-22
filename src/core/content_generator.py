"""
PDF Content Generation Module for AI-Powered Summarization and Media Creation.

This module implements a comprehensive workflow for converting PDFs into
summarized content with voiceovers, avatars, and visual elements.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentGenerationConfig:
    """Configuration for content generation workflow."""
    max_script_words: int = 90
    key_points_count: int = 3
    key_point_max_words: int = 12
    voice_style: str = "friendly"  # friendly, professional, casual
    language: str = "American English"
    avatar_style: str = "Professional"  # Professional, Cartoon, Friendly
    image_style: str = "Infographic-style"
    
@dataclass
class GeneratedContent:
    """Container for all generated content."""
    original_text: str
    summary_script: str
    key_points: List[str]
    voiceover_prompt: str
    avatar_prompt: str
    image_prompt: str
    word_count: int
    concept_keywords: List[str]

class ContentGenerator:
    """
    Generates AI-ready prompts and content from PDF text following
    a structured workflow for video content creation.
    """
    
    def __init__(self, config: Optional[ContentGenerationConfig] = None):
        self.config = config or ContentGenerationConfig()
        
    def generate_summary_prompt(self, text: str) -> str:
        """
        Generate the PDF summarization prompt.
        
        Args:
            text: Full text from PDF
            
        Returns:
            Formatted prompt for AI summarization
        """
        return f"""Summarize this entire PDF in a short script of no more than {self.config.max_script_words} words.

PDF Content:
{text}

Requirements:
- Maximum {self.config.max_script_words} words
- Clear and engaging script format
- Suitable for voiceover narration
- Capture the main message and key insights
"""

    def generate_key_points_prompt(self, text: str) -> str:
        """
        Generate the key points extraction prompt.
        
        Args:
            text: Full text from PDF
            
        Returns:
            Formatted prompt for key points extraction
        """
        return f"""Give me the top {self.config.key_points_count} key takeaways in under {self.config.key_point_max_words} words each.

PDF Content:
{text}

Format each takeaway as a concise bullet point, maximum {self.config.key_point_max_words} words per point.
"""

    def generate_voiceover_prompt(self, script: str) -> str:
        """
        Generate the ElevenLabs voiceover prompt.
        
        Args:
            script: The summarized script
            
        Returns:
            Formatted prompt for voiceover generation
        """
        return f"""Generate a {self.config.voice_style} and clear narration of this script in {self.config.language}:

'{script}'

Voice characteristics:
- Tone: {self.config.voice_style}
- Language: {self.config.language}
- Pace: Natural and engaging
- Clarity: High priority for comprehension
"""

    def generate_avatar_prompt(self, audio_description: str = "narration audio") -> str:
        """
        Generate the D-ID avatar video prompt.
        
        Args:
            audio_description: Description of the audio to sync with
            
        Returns:
            Formatted prompt for avatar video creation
        """
        return f"""Create a talking avatar using this audio and match lip sync:
Style: {self.config.avatar_style}

Audio: {audio_description}

Requirements:
- Perfect lip synchronization
- {self.config.avatar_style} appearance and demeanor
- Natural facial expressions
- Professional quality output
"""

    def generate_image_prompt(self, concept_keywords: List[str]) -> str:
        """
        Generate the Midjourney/DALL-E image prompt.
        
        Args:
            concept_keywords: Key concepts from the PDF content
            
        Returns:
            Formatted prompt for image generation
        """
        concept_text = ", ".join(concept_keywords[:3])  # Use top 3 concepts
        
        return f"""{self.config.image_style} background showing {concept_text}, with clean layout and white space.

Visual elements:
- Style: {self.config.image_style}
- Layout: Clean and organized
- Concepts: {concept_text}
- Design: Professional with ample white space
- Colors: Harmonious and readable
- Format: Suitable for video background
"""

    def extract_concept_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key concepts and keywords from text for image generation.
        
        Args:
            text: Source text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of relevant keywords/concepts
        """
        # Simple keyword extraction - in production, use NLP libraries
        # Remove common words and extract meaningful terms
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Clean and tokenize text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords

    def generate_content_workflow(self, text: str) -> Dict[str, str]:
        """
        Generate all prompts for the complete content creation workflow.
        
        Args:
            text: Full PDF text content
            
        Returns:
            Dictionary containing all generated prompts and content
        """
        logger.info("Generating complete content workflow...")
        
        # Extract concepts for image generation
        concept_keywords = self.extract_concept_keywords(text)
        
        # Generate all prompts
        summary_prompt = self.generate_summary_prompt(text)
        key_points_prompt = self.generate_key_points_prompt(text)
        
        # For demonstration, create placeholder script and key points
        # In production, these would be generated by AI models
        placeholder_script = f"[AI-generated summary script - max {self.config.max_script_words} words]"
        placeholder_key_points = [f"[Key point {i+1} - max {self.config.key_point_max_words} words]" 
                                 for i in range(self.config.key_points_count)]
        
        voiceover_prompt = self.generate_voiceover_prompt(placeholder_script)
        avatar_prompt = self.generate_avatar_prompt("generated narration audio")
        image_prompt = self.generate_image_prompt(concept_keywords)
        
        workflow = {
            "1_pdf_summarization_prompt": summary_prompt,
            "2_key_points_extraction_prompt": key_points_prompt,
            "3_voiceover_prompt": voiceover_prompt,
            "4_avatar_video_prompt": avatar_prompt,
            "5_image_generation_prompt": image_prompt,
            "extracted_concepts": concept_keywords,
            "config": {
                "max_script_words": self.config.max_script_words,
                "key_points_count": self.config.key_points_count,
                "voice_style": self.config.voice_style,
                "avatar_style": self.config.avatar_style,
                "language": self.config.language
            }
        }
        
        logger.info("Content workflow generation completed")
        return workflow

    def save_workflow_to_file(self, workflow: Dict[str, str], output_path: Path) -> None:
        """
        Save the generated workflow to a JSON file.
        
        Args:
            workflow: Generated workflow dictionary
            output_path: Path to save the workflow file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(workflow, f, indent=2, ensure_ascii=False)
            logger.info(f"Workflow saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save workflow: {str(e)}")
            raise

    def create_markdown_report(self, workflow: Dict[str, str], pdf_name: str) -> str:
        """
        Create a formatted markdown report of the content generation workflow.
        
        Args:
            workflow: Generated workflow dictionary
            pdf_name: Name of the source PDF
            
        Returns:
            Formatted markdown string
        """
        concepts_list = ", ".join(workflow.get("extracted_concepts", [])[:5])
        
        markdown = f"""# PDF Content Generation Workflow

**Source PDF:** {pdf_name}
**Generated:** {logger.name}

## Configuration
- **Script Length:** {workflow['config']['max_script_words']} words maximum
- **Key Points:** {workflow['config']['key_points_count']} takeaways
- **Voice Style:** {workflow['config']['voice_style']}
- **Avatar Style:** {workflow['config']['avatar_style']}
- **Language:** {workflow['config']['language']}

## Extracted Concepts
{concepts_list}

---

## 1. PDF Summarization Prompt

```
{workflow['1_pdf_summarization_prompt']}
```

---

## 2. Key Points Extraction Prompt

```
{workflow['2_key_points_extraction_prompt']}
```

---

## 3. Voiceover Prompt (ElevenLabs)

```
{workflow['3_voiceover_prompt']}
```

---

## 4. Avatar Video Prompt (D-ID)

```
{workflow['4_avatar_video_prompt']}
```

---

## 5. Image Generation Prompt (Midjourney/DALL-E)

```
{workflow['5_image_generation_prompt']}
```

---

## Workflow Steps

1. **Extract PDF Content** → Use PDF processor to get full text
2. **Generate Summary** → Apply summarization prompt to AI model
3. **Extract Key Points** → Generate bullet points from content
4. **Create Voiceover** → Use ElevenLabs with generated script
5. **Generate Avatar Video** → Use D-ID with audio and style preferences
6. **Create Background Image** → Use Midjourney/DALL-E with concept keywords
7. **Combine Assets** → Merge avatar, background, and audio for final video

## Next Steps

1. Copy the prompts above to your AI tools
2. Process each step in sequence
3. Use generated assets to create final video content
4. Review and refine as needed

---

*Generated by PDF Content Generator - {pdf_name}*
"""
        
        return markdown