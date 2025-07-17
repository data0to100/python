# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or run the automated setup
python scripts/install_dependencies.py
```

### 2. Create a Sample PDF (Optional)
```bash
# Install reportlab for PDF creation
pip install reportlab

# Create sample PDF
python scripts/create_sample_pdf.py
```

### 3. Run the Application

#### Option A: Web Interface (Recommended)
```bash
python main.py --interface web
```
Then open http://localhost:8501 in your browser.

#### Option B: Command Line Interface
```bash
# Preview PDF content
python main.py --interface cli --input pdf_scripts/sample_script.pdf --preview

# Convert to audio
python main.py --interface cli --input pdf_scripts/sample_script.pdf --output-type audio

# Convert to video with subtitles
python main.py --interface cli --input pdf_scripts/sample_script.pdf --output-type video --subtitles
```

#### Option C: Programmatic Usage
```bash
python examples/sample_script.py
```

## 📁 Project Structure

```
pdf-script-converter/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── config.py            # Configuration management
│   │   ├── pdf_processor.py     # PDF text extraction
│   │   ├── tts_engine.py        # Text-to-speech engines
│   │   └── video_generator.py   # Video creation
│   ├── interfaces/              # User interfaces
│   │   ├── cli.py              # Command-line interface
│   │   └── streamlit_app.py    # Web interface
│   └── utils/                   # Utility modules
│       ├── file_manager.py     # File operations
│       └── subtitle_generator.py # Subtitle creation
├── pdf_scripts/                 # Input PDF files
├── audio_output/               # Generated audio files
├── video_output/               # Generated video files
├── examples/                   # Example scripts
├── scripts/                    # Utility scripts
└── main.py                     # Main entry point
```

## 🎛️ Configuration Options

### Voice Settings
- **Engine**: pyttsx3 (offline), gTTS (online), ElevenLabs (API)
- **Speed**: 0.5 to 2.0 (default: 1.0)
- **Language**: Various languages for gTTS
- **Pitch**: -50 to 50 (default: 0)

### Output Options
- **Audio Formats**: MP3, WAV
- **Video Resolution**: 1920x1080, 1280x720, 854x480
- **Subtitles**: SRT, VTT formats
- **Background**: Custom images supported

## 🔧 System Requirements

### Required
- Python 3.8+
- FFmpeg (for video processing)

### Linux Additional
- espeak (for offline TTS)

### Installation Commands
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg espeak

# macOS
brew install ffmpeg

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
```

## 🎯 Common Use Cases

### 1. Script Narration
Convert presentation scripts to professional voice-over audio.

### 2. Accessibility
Create audio versions of written content for visually impaired users.

### 3. Video Content
Generate videos with synchronized audio and subtitles for social media.

### 4. Language Learning
Create pronunciation examples from text materials.

### 5. Content Creation
Batch convert multiple scripts for podcast or video production.

## 🆘 Troubleshooting

### "No TTS engines available"
- Install pyttsx3: `pip install pyttsx3`
- For Linux: Install espeak system package

### "FFmpeg not found"
- Install FFmpeg system package
- Add FFmpeg to system PATH

### "Failed to extract text from PDF"
- Try different extraction method: `--extraction-method pdfplumber`
- Ensure PDF contains selectable text (not just images)

### Web interface won't start
- Install Streamlit: `pip install streamlit`
- Check if port 8501 is available

## 📚 Next Steps

1. **API Keys**: Set up ElevenLabs API for premium voices in `.env`
2. **Custom Backgrounds**: Add your own background images for videos
3. **Batch Processing**: Use the programmatic API for multiple files
4. **Integration**: Embed the modules in your own applications

## 🤝 Support

- Check the full README.md for detailed documentation
- Run `python main.py --interface cli --help` for CLI options
- Use the built-in preview feature to test PDF extraction