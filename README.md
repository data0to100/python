# PDF Script to Speech/Video Converter

A comprehensive Python application that converts PDF scripts into natural-sounding audio and optionally creates videos with subtitles.

## Features

- **PDF Text Extraction**: Uses PyMuPDF and pdfplumber for robust text extraction
- **Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS, ElevenLabs API support)
- **Video Generation**: Creates videos with audio and subtitles using moviepy
- **Voice Customization**: Adjustable speed, pitch, and language settings
- **Multiple Output Formats**: Audio-only (MP3/WAV) or video with subtitles
- **User Interfaces**: Both CLI and Streamlit web interface
- **Organized Output**: Structured folder organization for inputs and outputs

## Project Structure

```
pdf_scripts/          # Input PDF files
audio_output/         # Generated audio files
video_output/         # Generated video files with subtitles
src/
├── core/
│   ├── pdf_processor.py      # PDF text extraction
│   ├── tts_engine.py         # Text-to-speech conversion
│   ├── video_generator.py    # Video creation with subtitles
│   └── config.py             # Configuration management
├── interfaces/
│   ├── cli.py               # Command-line interface
│   └── streamlit_app.py     # Web interface
└── utils/
    ├── file_manager.py      # File operations
    └── subtitle_generator.py # Subtitle creation
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

Create a `.env` file for API keys (optional):
```
ELEVENLABS_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

```bash
# Basic audio conversion
python src/interfaces/cli.py --input pdf_scripts/script.pdf --output-type audio

# Video with subtitles
python src/interfaces/cli.py --input pdf_scripts/script.pdf --output-type video --voice-speed 1.2

# Preview script before conversion
python src/interfaces/cli.py --input pdf_scripts/script.pdf --preview
```

### Streamlit Web Interface

```bash
streamlit run src/interfaces/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

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
