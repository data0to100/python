#!/usr/bin/env python3
"""
Main entry point for PDF Script to Speech/Video Converter.

This script provides a simple way to run either the CLI or Streamlit interface.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Main entry point with interface selection."""
    parser = argparse.ArgumentParser(
        description="PDF Script to Speech/Video Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Streamlit web interface
  python main.py --interface web

  # Run CLI interface  
  python main.py --interface cli --input script.pdf --output-type audio

  # Show CLI help
  python main.py --interface cli --help
        """
    )
    
    parser.add_argument(
        '--interface',
        choices=['web', 'cli'],
        default='web',
        help='Interface to run (default: web)'
    )
    
    # Parse only known args to allow CLI args to pass through
    args, remaining = parser.parse_known_args()
    
    if args.interface == 'web':
        # Check for ML platform flag
        app_choice = input("Choose application:\n1. PDF to Speech/Video Converter\n2. Enterprise AI/ML Platform\nEnter choice (1 or 2): ")
        
        if app_choice == '2':
            # Run ML Platform
            import subprocess
            import os
            
            streamlit_script = Path(__file__).parent / 'src' / 'interfaces' / 'ml_platform_app.py'
            
            # Set environment variable for Streamlit
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent / 'src')
            
            cmd = ['streamlit', 'run', str(streamlit_script)]
            
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running ML Platform: {e}")
                print("Make sure all dependencies are installed: pip install -r requirements.txt")
                return 1
            except KeyboardInterrupt:
                print("\nML Platform stopped.")
                return 0
        else:
            # Run PDF Converter (default)
            import subprocess
            import os
            
            streamlit_script = Path(__file__).parent / 'src' / 'interfaces' / 'streamlit_app.py'
            
            # Set environment variable for Streamlit
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent / 'src')
            
            cmd = ['streamlit', 'run', str(streamlit_script)]
            
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running Streamlit: {e}")
                print("Make sure Streamlit is installed: pip install streamlit")
                return 1
            except KeyboardInterrupt:
                print("\nStreamlit app stopped.")
                return 0
    
    elif args.interface == 'cli':
        # Run CLI interface with remaining arguments
        sys.argv = ['cli.py'] + remaining
        
        try:
            from interfaces.cli import main as cli_main
            return cli_main()
        except ImportError as e:
            print(f"Error importing CLI module: {e}")
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())