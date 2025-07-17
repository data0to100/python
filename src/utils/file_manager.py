"""
File management utilities for PDF Script to Speech/Video Converter.

This module provides utilities for file operations, organization,
and cleanup tasks.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileManager:
    """
    Handles file operations and organization for the application.
    
    Provides utilities for creating directories, managing temporary files,
    and organizing output files.
    """
    
    def __init__(self):
        self.temp_files = []  # Track temporary files for cleanup
    
    def ensure_directory(self, directory: Path) -> bool:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            directory: Path to the directory
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            return False
    
    def get_unique_filename(self, base_path: Path, extension: str = "") -> Path:
        """
        Generate a unique filename by adding timestamp or counter.
        
        Args:
            base_path: Base path without extension
            extension: File extension (with or without dot)
            
        Returns:
            Unique file path
        """
        if not extension.startswith('.') and extension:
            extension = '.' + extension
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_path = base_path.parent / f"{base_path.stem}_{timestamp}{extension}"
        
        # If still exists, add counter
        counter = 1
        while unique_path.exists():
            unique_path = base_path.parent / f"{base_path.stem}_{timestamp}_{counter}{extension}"
            counter += 1
        
        return unique_path
    
    def organize_output_files(self, base_name: str, audio_files: List[Path], 
                             video_file: Optional[Path] = None, 
                             output_dir: Path = Path("output")) -> Dict[str, Path]:
        """
        Organize output files into structured directories.
        
        Args:
            base_name: Base name for the project
            audio_files: List of audio file paths
            video_file: Optional video file path
            output_dir: Base output directory
            
        Returns:
            Dictionary with organized file paths
        """
        # Create timestamped project directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = output_dir / f"{base_name}_{timestamp}"
        
        # Create subdirectories
        audio_dir = project_dir / "audio"
        video_dir = project_dir / "video"
        
        self.ensure_directory(audio_dir)
        self.ensure_directory(video_dir)
        
        organized_files = {
            'project_dir': project_dir,
            'audio_dir': audio_dir,
            'video_dir': video_dir,
            'audio_files': [],
            'video_file': None
        }
        
        try:
            # Copy audio files
            for i, audio_file in enumerate(audio_files):
                if audio_file.exists():
                    new_name = f"{base_name}_page_{i+1:03d}{audio_file.suffix}"
                    new_path = audio_dir / new_name
                    shutil.copy2(audio_file, new_path)
                    organized_files['audio_files'].append(new_path)
                    logger.info(f"Copied audio file: {new_path}")
            
            # Copy video file
            if video_file and video_file.exists():
                new_name = f"{base_name}_complete{video_file.suffix}"
                new_path = video_dir / new_name
                shutil.copy2(video_file, new_path)
                organized_files['video_file'] = new_path
                logger.info(f"Copied video file: {new_path}")
            
            logger.info(f"Output files organized in: {project_dir}")
            
        except Exception as e:
            logger.error(f"Failed to organize output files: {str(e)}")
        
        return organized_files
    
    def clean_temp_files(self) -> None:
        """Clean up temporary files created during processing."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean temp file {temp_file}: {str(e)}")
        
        self.temp_files.clear()
    
    def add_temp_file(self, file_path: Path) -> None:
        """Add a file to the temporary files list for later cleanup."""
        self.temp_files.append(file_path)
    
    def get_file_info(self, file_path: Path) -> Dict[str, any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            return {'exists': False}
        
        stat = file_path.stat()
        
        return {
            'exists': True,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'suffix': file_path.suffix,
            'name': file_path.name
        }
    
    def find_pdf_files(self, directory: Path) -> List[Path]:
        """
        Find all PDF files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return pdf_files
        
        try:
            for file_path in directory.rglob("*.pdf"):
                if file_path.is_file():
                    pdf_files.append(file_path)
            
            logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
            
        except Exception as e:
            logger.error(f"Error searching for PDF files: {str(e)}")
        
        return sorted(pdf_files)
    
    def validate_output_paths(self, output_dir: Path, 
                             required_space_mb: float = 100) -> Tuple[bool, str]:
        """
        Validate that output directory is writable and has enough space.
        
        Args:
            output_dir: Output directory path
            required_space_mb: Required space in megabytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Ensure directory exists
            if not self.ensure_directory(output_dir):
                return False, f"Cannot create output directory: {output_dir}"
            
            # Check if writable
            test_file = output_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception:
                return False, f"Output directory is not writable: {output_dir}"
            
            # Check available space
            stat = shutil.disk_usage(output_dir)
            available_mb = stat.free / (1024 * 1024)
            
            if available_mb < required_space_mb:
                return False, f"Insufficient disk space. Required: {required_space_mb}MB, Available: {available_mb:.1f}MB"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating output directory: {str(e)}"
    
    def create_project_manifest(self, project_dir: Path, metadata: Dict) -> Path:
        """
        Create a manifest file with project information.
        
        Args:
            project_dir: Project directory
            metadata: Project metadata
            
        Returns:
            Path to manifest file
        """
        import json
        
        manifest_path = project_dir / "manifest.json"
        
        manifest_data = {
            'created': datetime.datetime.now().isoformat(),
            'project_name': metadata.get('project_name', 'Unnamed Project'),
            'source_pdf': str(metadata.get('source_pdf', '')),
            'total_pages': metadata.get('total_pages', 0),
            'total_audio_files': metadata.get('total_audio_files', 0),
            'has_video': metadata.get('has_video', False),
            'voice_config': metadata.get('voice_config', {}),
            'output_config': metadata.get('output_config', {}),
            'file_list': {
                'audio_files': [str(f) for f in metadata.get('audio_files', [])],
                'video_file': str(metadata.get('video_file', '')) if metadata.get('video_file') else None
            }
        }
        
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            logger.info(f"Project manifest created: {manifest_path}")
            
        except Exception as e:
            logger.error(f"Failed to create manifest: {str(e)}")
        
        return manifest_path
    
    def load_project_manifest(self, manifest_path: Path) -> Optional[Dict]:
        """
        Load project manifest file.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            Manifest data or None if failed
        """
        import json
        
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest: {str(e)}")
            return None