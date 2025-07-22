# PDF Content Generation Workflow

Transform any PDF into AI-ready prompts for creating engaging video content with voiceovers, avatars, and visuals.

## 🎯 What This Does

This system takes your PDF documents and generates structured prompts for:

1. **PDF Summarization** (90 words max) - Perfect for video scripts
2. **Key Points Extraction** (3 points, 12 words each) - For highlights and bullet points
3. **Voiceover Generation** (ElevenLabs) - Natural narration prompts
4. **Avatar Video Creation** (D-ID) - Talking avatar with lip sync
5. **Background Images** (Midjourney/DALL-E) - Professional infographic-style visuals

## 🚀 Quick Start

### Method 1: Simple Demo (No Dependencies)
```bash
cd examples
python3 simple_demo.py
```

### Method 2: Full System (Requires Dependencies)
```bash
# Install dependencies (if not in managed environment)
pip install -r requirements.txt

# Run content generation workflow
python main.py --interface content-gen --input your_document.pdf

# Or use directly
python src/interfaces/content_generator_cli.py --input your_document.pdf
```

## 📋 Your Original Prompts Implemented

The system implements your exact prompt specifications:

### 1. PDF Summarization Prompt
```
"Summarize this entire PDF in a short script of no more than 90 words."
```

### 2. Key Points Extraction
```
"Give me the top 3 key takeaways in under 12 words each."
```

### 3. Voiceover Prompt (ElevenLabs)
```
"Generate a friendly and clear narration of this script in American English:
'[Final script]'"
```

### 4. Avatar Video Prompt (D-ID)
```
"Create a talking avatar using this audio and match lip sync:
Style: [Professional / Cartoon / Friendly]"
```

### 5. Image Prompt for Midjourney or DALL·E
```
"Infographic-style background showing [concept], with clean layout and white space."
```

## 🛠️ Usage Examples

### Basic Usage
```bash
# Generate workflow from PDF with default settings
python main.py --interface content-gen --input document.pdf
```

### Custom Configuration
```bash
# Professional style with custom word limits
python main.py --interface content-gen \
  --input document.pdf \
  --script-words 75 \
  --voice-style professional \
  --avatar-style Professional
```

### Preview PDF First
```bash
# Check PDF content before processing
python main.py --interface content-gen \
  --input document.pdf \
  --preview-only
```

### Save to Specific Directory
```bash
# Generate and save to custom location
python main.py --interface content-gen \
  --input document.pdf \
  --output-dir ./my_workflows \
  --format both
```

## ⚙️ Configuration Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--script-words` | 50-150 | 90 | Maximum words in summary script |
| `--key-points` | 1-5 | 3 | Number of key points to extract |
| `--key-point-words` | 5-20 | 12 | Maximum words per key point |
| `--voice-style` | friendly, professional, casual | friendly | Voice narration style |
| `--avatar-style` | Professional, Cartoon, Friendly | Professional | Avatar appearance |
| `--language` | Any language | American English | Voiceover language |
| `--format` | json, markdown, both | both | Output format |

## 📁 Output Structure

The system generates organized outputs:

```
content_workflows/
├── document_workflow.json      # Machine-readable workflow
├── document_workflow.md        # Human-readable report
└── [timestamp]_backup/         # Backup copies
```

### JSON Output
```json
{
  "1_pdf_summarization_prompt": "...",
  "2_key_points_extraction_prompt": "...",
  "3_voiceover_prompt": "...",
  "4_avatar_video_prompt": "...",
  "5_image_generation_prompt": "...",
  "extracted_concepts": ["ai", "business", "automation"],
  "config": { "max_script_words": 90, "voice_style": "friendly" }
}
```

### Markdown Report
- Complete workflow overview
- Ready-to-copy prompts
- Configuration summary
- Step-by-step instructions

## 🎬 Complete Workflow Process

1. **Extract PDF Content** → System reads and processes your PDF
2. **Generate Summary Prompt** → Creates 90-word script prompt
3. **Extract Key Points** → Identifies top 3 takeaways (12 words each)
4. **Create Voiceover Prompt** → ElevenLabs-ready narration prompt
5. **Generate Avatar Prompt** → D-ID avatar creation prompt
6. **Create Image Prompt** → Midjourney/DALL-E background prompt
7. **Output Organized Files** → JSON + Markdown reports

## 🔧 Advanced Features

### Concept Extraction
- Automatically identifies key concepts from PDF content
- Uses concepts for image generation prompts
- Filters out common words for better relevance

### Multiple Output Formats
- **JSON**: For automation and integration
- **Markdown**: For human review and copying
- **Both**: Complete documentation

### Flexible Configuration
- Customizable word limits
- Multiple voice and avatar styles
- Language options for international content

## 💡 Tips for Best Results

### PDF Preparation
- Use clear, well-structured PDFs
- Avoid image-heavy documents (text extraction works best)
- Keep content focused (under 2000 words recommended)

### AI Tool Integration
1. Copy the generated prompts to your AI tools
2. Use the exact prompts for consistent results
3. Adjust based on AI tool-specific requirements

### Content Optimization
- Review extracted concepts for accuracy
- Adjust word limits based on content complexity
- Test different voice styles for best fit

## 🚨 Troubleshooting

### Common Issues

**"No text extracted from PDF"**
- PDF might be image-based or encrypted
- Try different extraction method: `--extract-method pdfplumber`

**"Module not found" errors**
- Install dependencies: `pip install -r requirements.txt`
- Or use the simple demo: `python3 examples/simple_demo.py`

**Large document warnings**
- Consider splitting large PDFs
- Or increase script word limit: `--script-words 120`

## 🎯 Integration with AI Tools

### ElevenLabs Setup
1. Copy the voiceover prompt
2. Paste into ElevenLabs interface
3. Select voice and generate audio

### D-ID Avatar Creation
1. Upload generated audio to D-ID
2. Copy avatar prompt for style selection
3. Generate talking avatar video

### Midjourney/DALL-E Images
1. Copy image generation prompt
2. Paste into image generation tool
3. Use as video background

## 📊 Example Output

For a PDF about "AI in Business", the system generates:

**Summary Prompt**: Ready-to-use prompt for 90-word script
**Key Points**: 3 bullet points, 12 words each
**Voiceover**: "Generate friendly narration in American English..."
**Avatar**: "Professional style talking avatar with lip sync..."
**Images**: "Infographic showing AI, business, automation with clean layout..."

## 🔄 Workflow Integration

This system integrates with the existing PDF-to-speech converter:
- Use existing PDF processing capabilities
- Extend with AI content generation
- Maintain compatibility with current features

## 📈 Future Enhancements

- Direct API integration with AI services
- Batch processing for multiple PDFs
- Template customization for different industries
- Real-time preview of generated content

---

## 🏃‍♂️ Get Started Now

1. **Try the demo**: `python3 examples/simple_demo.py`
2. **Process your PDF**: `python main.py --interface content-gen --input your_file.pdf`
3. **Copy the prompts** to your AI tools
4. **Create amazing video content**!

Your PDF content generation workflow is ready to transform documents into engaging video content! 🎥✨