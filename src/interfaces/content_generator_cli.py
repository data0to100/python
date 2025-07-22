#!/usr/bin/env python3
"""
Content Generator CLI for PDF to AI Content Workflow.

This CLI tool processes PDFs and generates structured prompts for:
1. PDF Summarization (90 words max)
2. Key Points Extraction (3 points, 12 words each)
3. Voiceover Generation (ElevenLabs)
4. Avatar Video Creation (D-ID)
5. Background Image Generation (Midjourney/DALL-E)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pdf_processor import PDFProcessor
from core.content_generator import ContentGenerator, ContentGenerationConfig

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PDF Content Generation Workflow - Generate AI prompts for video content creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate workflow from PDF with default settings
  python content_generator_cli.py --input document.pdf

  # Custom configuration
  python content_generator_cli.py --input document.pdf --script-words 75 --voice-style professional

  # Generate workflow and save to specific directory
  python content_generator_cli.py --input document.pdf --output-dir ./workflows --format both

  # Preview PDF content first
  python content_generator_cli.py --input document.pdf --preview-only

Available Styles:
  Voice: friendly, professional, casual
  Avatar: Professional, Cartoon, Friendly
  Image: Infographic-style, Minimalist, Corporate
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input PDF file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./content_workflows'),
        help='Output directory for generated workflows (default: ./content_workflows)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'markdown', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    
    # Content configuration
    parser.add_argument(
        '--script-words',
        type=int,
        default=90,
        help='Maximum words in summary script (default: 90)'
    )
    
    parser.add_argument(
        '--key-points',
        type=int,
        default=3,
        help='Number of key points to extract (default: 3)'
    )
    
    parser.add_argument(
        '--key-point-words',
        type=int,
        default=12,
        help='Maximum words per key point (default: 12)'
    )
    
    parser.add_argument(
        '--voice-style',
        choices=['friendly', 'professional', 'casual'],
        default='friendly',
        help='Voice style for narration (default: friendly)'
    )
    
    parser.add_argument(
        '--language',
        default='American English',
        help='Language for voiceover (default: American English)'
    )
    
    parser.add_argument(
        '--avatar-style',
        choices=['Professional', 'Cartoon', 'Friendly'],
        default='Professional',
        help='Avatar style for video (default: Professional)'
    )
    
    parser.add_argument(
        '--image-style',
        default='Infographic-style',
        help='Image style for background (default: Infographic-style)'
    )
    
    # Processing options
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='Only preview PDF content without generating workflow'
    )
    
    parser.add_argument(
        '--extract-method',
        choices=['auto', 'pymupdf', 'pdfplumber'],
        default='auto',
        help='PDF text extraction method (default: auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser

def preview_pdf_content(pdf_path: Path, extract_method: str = 'auto') -> None:
    """Preview PDF content before processing."""
    print(f"\n📄 Previewing PDF: {pdf_path.name}")
    print("=" * 60)
    
    try:
        processor = PDFProcessor()
        preview_text = processor.get_preview(pdf_path, max_chars=1000)
        
        # Get basic stats
        pages = processor.extract_text(pdf_path, method=extract_method)
        stats = processor.get_text_stats(pages)
        
        print(f"📊 Document Statistics:")
        print(f"   Pages: {stats['total_pages']}")
        print(f"   Words: {stats['total_words']:,}")
        print(f"   Characters: {stats['total_characters']:,}")
        print(f"   Avg words/page: {stats['average_words_per_page']}")
        
        print(f"\n📖 Content Preview:")
        print("-" * 40)
        print(preview_text)
        print("-" * 40)
        
        if stats['total_words'] > 2000:
            print(f"\n⚠️  Large document ({stats['total_words']:,} words)")
            print("   Consider using a shorter document for better AI processing")
        
    except Exception as e:
        print(f"❌ Error previewing PDF: {str(e)}")

def generate_workflow(
    pdf_path: Path,
    output_dir: Path,
    config: ContentGenerationConfig,
    extract_method: str = 'auto',
    output_format: str = 'both',
    verbose: bool = False
) -> None:
    """Generate the complete content workflow."""
    
    print(f"\n🚀 Generating Content Workflow for: {pdf_path.name}")
    print("=" * 60)
    
    try:
        # Extract PDF text
        if verbose:
            print("📄 Extracting PDF content...")
        
        processor = PDFProcessor()
        pages = processor.extract_text(pdf_path, method=extract_method)
        
        if not pages:
            print("❌ No text could be extracted from the PDF")
            return
        
        # Combine all text
        full_text = " ".join(page['text'] for page in pages)
        
        # Generate workflow
        if verbose:
            print("🤖 Generating AI prompts...")
        
        generator = ContentGenerator(config)
        workflow = generator.generate_content_workflow(full_text)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        base_name = pdf_path.stem
        
        # Save outputs based on format choice
        if output_format in ['json', 'both']:
            json_path = output_dir / f"{base_name}_workflow.json"
            generator.save_workflow_to_file(workflow, json_path)
            print(f"💾 JSON workflow saved: {json_path}")
        
        if output_format in ['markdown', 'both']:
            markdown_path = output_dir / f"{base_name}_workflow.md"
            markdown_content = generator.create_markdown_report(workflow, pdf_path.name)
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"📝 Markdown report saved: {markdown_path}")
        
        # Display summary
        print(f"\n✅ Workflow Generation Complete!")
        print(f"   📊 Extracted concepts: {len(workflow['extracted_concepts'])}")
        print(f"   🎯 Top concepts: {', '.join(workflow['extracted_concepts'][:3])}")
        print(f"   ⚙️  Configuration: {config.max_script_words} words, {config.voice_style} voice")
        
        print(f"\n🎬 Ready-to-Use AI Prompts Generated:")
        print(f"   1. PDF Summarization ({config.max_script_words} words max)")
        print(f"   2. Key Points Extraction ({config.key_points_count} points)")
        print(f"   3. ElevenLabs Voiceover ({config.voice_style} style)")
        print(f"   4. D-ID Avatar Video ({config.avatar_style} avatar)")
        print(f"   5. Midjourney/DALL-E Images ({config.image_style})")
        
        print(f"\n📁 Output Location: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error generating workflow: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()

def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"❌ PDF file not found: {args.input}")
        return 1
    
    if args.input.suffix.lower() != '.pdf':
        print(f"❌ Input file must be a PDF: {args.input}")
        return 1
    
    # Create configuration
    config = ContentGenerationConfig(
        max_script_words=args.script_words,
        key_points_count=args.key_points,
        key_point_max_words=args.key_point_words,
        voice_style=args.voice_style,
        language=args.language,
        avatar_style=args.avatar_style,
        image_style=args.image_style
    )
    
    print("🎥 PDF Content Generation Workflow")
    print("   Transform PDFs into AI-ready prompts for video creation")
    
    if args.verbose:
        print(f"\n⚙️  Configuration:")
        print(f"   Script length: {config.max_script_words} words")
        print(f"   Key points: {config.key_points_count} points ({config.key_point_max_words} words each)")
        print(f"   Voice style: {config.voice_style}")
        print(f"   Avatar style: {config.avatar_style}")
        print(f"   Language: {config.language}")
    
    try:
        # Preview mode
        if args.preview_only:
            preview_pdf_content(args.input, args.extract_method)
            return 0
        
        # Generate workflow
        generate_workflow(
            pdf_path=args.input,
            output_dir=args.output_dir,
            config=config,
            extract_method=args.extract_method,
            output_format=args.format,
            verbose=args.verbose
        )
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())