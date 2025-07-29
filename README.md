# 🎬 PDF to Video Creator

A powerful Streamlit application that extracts scripts from PDF webinar files, summarizes them using AI, and creates professional videos using RenderForest.

## ✨ Features

- **PDF Text Extraction**: Extract text from PDF files using multiple methods (PyPDF2 + pdfplumber)
- **AI-Powered Summarization**: Use state-of-the-art AI models to summarize webinar content
- **RenderForest Integration**: Create professional videos directly from summarized content
- **User-Friendly Interface**: Beautiful Streamlit UI with real-time feedback
- **Customizable Settings**: Adjust summary length, video duration, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- RenderForest API key (get it from [RenderForest](https://app.renderforest.com/api-keys))

### Option 1: Simple Installation (Recommended)

1. **Install minimal dependencies**:
   ```bash
   pip install streamlit PyPDF2 requests
   ```

2. **Run the simplified version**:
   ```bash
   streamlit run simple_app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

### Option 2: Full Installation (Advanced Features)

1. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your RenderForest API key:
   ```
   RENDERFOREST_API_KEY=your_actual_api_key_here
   ```

3. **Run the full application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## 📖 How to Use

### Step 1: Upload PDF
- Click "Browse files" to upload your PDF webinar file
- The app will automatically extract text from the PDF
- View the extracted text in the expandable section

### Step 2: Generate AI Summary
- Click "🤖 Generate AI Summary" button
- Adjust summary settings in the sidebar if needed:
  - **Max Summary Length**: 50-300 characters (default: 150)
  - **Min Summary Length**: 10-100 characters (default: 50) - Full version only
- The AI will create a concise summary of your webinar content

### Step 3: Create Video
- Review and edit the generated summary if needed
- Configure video settings in the sidebar:
  - **Template ID**: RenderForest template to use (default: "1")
  - **Video Title**: Title for your video
  - **Video Duration**: Length in seconds (10-120)
- Click "🎬 Create Video with RenderForest"
- Check your RenderForest dashboard for the final video

## ⚙️ Configuration

### RenderForest API Setup

1. Sign up for a RenderForest account at [renderforest.com](https://renderforest.com)
2. Go to [API Keys](https://app.renderforest.com/api-keys)
3. Generate a new API key
4. Add the key to your `.env` file or enter it in the sidebar

### Version Differences

| Feature | Simple Version | Full Version |
|---------|----------------|--------------|
| PDF Processing | Basic PyPDF2 | PyPDF2 + pdfplumber |
| AI Summarization | Simple text processing | Advanced AI models |
| Dependencies | Minimal | Full AI stack |
| Installation | Easy | Requires more setup |

## 🛠️ Technical Details

### Architecture

- **PDFProcessor**: Handles PDF text extraction using PyPDF2 and pdfplumber
- **AISummarizer**: Manages AI-powered text summarization using HuggingFace transformers
- **RenderForestAPI**: Handles video creation via RenderForest API
- **Streamlit UI**: Beautiful, responsive web interface

### Dependencies

**Simple Version:**
- **Streamlit**: Web application framework
- **PyPDF2**: PDF text extraction
- **Requests**: HTTP requests for API calls

**Full Version:**
- **Streamlit**: Web application framework
- **PyPDF2 & pdfplumber**: PDF text extraction
- **Transformers**: AI model for summarization
- **LangChain**: Text processing utilities
- **Requests**: HTTP requests for API calls

### File Structure

```
pdf-to-video-creator/
├── app.py              # Full Streamlit application
├── simple_app.py       # Simplified version
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .env               # Your environment variables (create this)
├── demo.py            # Demo script
├── test_app.py        # Test suite
├── setup.py           # Setup script
└── README.md          # This file
```

## 🔧 Troubleshooting

### Common Issues

1. **"PyPDF2 not available" error**:
   ```bash
   pip install PyPDF2
   ```

2. **"RenderForest API key not found"**:
   - Check that you've added your API key to the `.env` file
   - Or enter it directly in the sidebar

3. **PDF text extraction fails**:
   - Try a different PDF file
   - Some PDFs may be image-based and require OCR

4. **Video creation fails**:
   - Verify your RenderForest API key is valid
   - Check your RenderForest account has available credits
   - Ensure the template ID exists in your account

5. **Installation issues**:
   - Use the simple version if you encounter dependency conflicts
   - Try installing with `--user` flag: `pip install --user streamlit`

### Performance Tips

- Use smaller PDF files for faster processing
- Adjust summary length based on your needs
- The AI model loads once and stays in memory (full version)

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_app.py
```

Or run the demo:

```bash
python demo.py
```

## 🤝 Contributing

Feel free to contribute to this project by:

1. Reporting bugs
2. Suggesting new features
3. Improving the code
4. Adding new AI models or video platforms

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [HuggingFace](https://huggingface.co/) for the AI models
- [RenderForest](https://renderforest.com/) for video creation API
- [PyPDF2](https://pypdf2.readthedocs.io/) and [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF processing

---

**Made with ❤️ for content creators who want to turn their webinar PDFs into engaging videos!**
