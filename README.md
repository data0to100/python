# AI-Enhanced PDF Script to Video Converter

A comprehensive Python application that uses AI to summarize PDF scripts and creates professional videos using Renderforest templates.

## Features

### 🤖 AI-Powered Summarization
- **Open-Source AI Models**: Uses transformer models (BART, T5, DistilBART) for intelligent text summarization
- **Multiple Summarization Strategies**: Chunk-based and extractive summarization options
- **Flexible Configuration**: Adjustable summary length and individual page processing

### 📄 PDF Text Extraction
- **Robust Extraction**: Uses PyMuPDF and pdfplumber for maximum compatibility
- **Multiple Methods**: Fallback extraction methods for various PDF formats

### 🎤 Text-to-Speech
- **Multiple TTS Engines**: pyttsx3 (offline), gTTS (online), ElevenLabs API support
- **Voice Customization**: Adjustable speed, pitch, and language settings
- **High-Quality Audio**: Multiple audio formats (MP3/WAV)

### 🎬 Professional Video Creation
- **Renderforest Integration**: Creates professional videos using Renderforest API
- **Multiple Templates**: Corporate, typography, slideshow, and explainer templates
- **Voiceover Support**: Automatically adds generated audio to videos
- **Quality Options**: Low, medium, and high-quality rendering

### 💻 User Interfaces
- **Streamlit Web App**: User-friendly web interface with real-time configuration
- **CLI Interface**: Command-line interface for automation and scripting
- **Progress Tracking**: Real-time progress updates for all operations

## Project Structure

```
pdf_scripts/          # Input PDF files
audio_output/         # Generated audio files
video_output/         # Generated video files with subtitles
src/
├── core/
│   ├── pdf_processor.py           # PDF text extraction
│   ├── ai_summarizer.py           # AI-powered text summarization
│   ├── tts_engine.py              # Text-to-speech conversion
│   ├── video_generator.py         # Local video creation with subtitles
│   ├── renderforest_integration.py # Renderforest API integration
│   └── config.py                  # Configuration management
├── interfaces/
│   ├── cli.py                     # Command-line interface
│   └── streamlit_app.py           # Enhanced web interface
└── utils/
    ├── file_manager.py            # File operations
    └── subtitle_generator.py      # Subtitle creation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-script-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p pdf_scripts audio_output video_output
```

## Configuration

Create a `.env` file for API keys:
```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
# Required for Renderforest video creation:
RENDERFOREST_API_KEY=your_renderforest_api_key_here

# Optional for premium TTS:
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### Getting API Keys

1. **Renderforest API**: Sign up at [renderforest.com](https://www.renderforest.com) and get your API key from the developer section
2. **ElevenLabs (optional)**: Get premium voice synthesis at [elevenlabs.io](https://elevenlabs.io)

## Usage

### Command Line Interface

```bash
# Basic audio conversion with AI summarization
python main.py --interface cli --input pdf_scripts/script.pdf --output-type audio

# Create Renderforest video with AI summary
python main.py --interface cli --input pdf_scripts/script.pdf --summarize --renderforest

# Preview script before conversion
python main.py --interface cli --input pdf_scripts/script.pdf --preview
```

### Streamlit Web Interface

```bash
# Run the enhanced web interface
python main.py --interface web

# Or directly with streamlit
streamlit run src/interfaces/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Complete AI-Enhanced Workflow

1. **Upload PDF**: Use the web interface to upload your webinar script
2. **AI Summarization**: Enable AI summarization to create concise content
3. **Configure Voice**: Choose TTS engine and voice settings
4. **Renderforest Video**: Enable professional video creation
5. **Download Results**: Get your AI-summarized audio and professional video

## Voice Parameters

- **Speed**: 0.5 to 2.0 (default: 1.0)
- **Pitch**: -50 to 50 (default: 0)
- **Language**: Various languages supported by gTTS
- **Voice Engine**: pyttsx3 (offline) or gTTS (online)

## Output Formats

- **Audio**: MP3, WAV
- **Video**: MP4 with audio and subtitles

## License

MIT License
