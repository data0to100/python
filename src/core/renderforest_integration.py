"""
Renderforest API integration for video generation.

This module provides integration with Renderforest's video creation API
to generate professional videos from summarized script content.
"""

import logging
import requests
import time
import json
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RenderforestConfig:
    """Configuration for Renderforest API."""
    api_key: str
    base_url: str = "https://api.renderforest.com"
    timeout: int = 300  # 5 minutes timeout for rendering
    max_retries: int = 3

@dataclass
class VideoTemplate:
    """Renderforest video template configuration."""
    template_id: int
    name: str
    description: str
    duration_range: tuple  # (min_seconds, max_seconds)
    supports_text: bool = True
    supports_audio: bool = True
    supports_voiceover: bool = True

class RenderforestIntegration:
    """
    Integration with Renderforest API for video generation.
    
    Provides methods to create videos from text content using
    various Renderforest templates and customization options.
    """
    
    # Popular templates for script-based videos
    SCRIPT_TEMPLATES = {
        'minimal_typography': VideoTemplate(
            template_id=701,
            name="Minimal Typography",
            description="Clean text-focused video with minimal design",
            duration_range=(10, 180),
            supports_text=True,
            supports_audio=True,
            supports_voiceover=True
        ),
        'corporate_presentation': VideoTemplate(
            template_id=888,
            name="Corporate Presentation",
            description="Professional business presentation style",
            duration_range=(30, 300),
            supports_text=True,
            supports_audio=True,
            supports_voiceover=True
        ),
        'modern_slideshow': VideoTemplate(
            template_id=1056,
            name="Modern Slideshow",
            description="Modern slideshow with text and transitions",
            duration_range=(20, 240),
            supports_text=True,
            supports_audio=True,
            supports_voiceover=True
        ),
        'kinetic_typography': VideoTemplate(
            template_id=1234,
            name="Kinetic Typography",
            description="Animated text with dynamic movements",
            duration_range=(15, 120),
            supports_text=True,
            supports_audio=True,
            supports_voiceover=True
        ),
        'explainer_video': VideoTemplate(
            template_id=1567,
            name="Explainer Video",
            description="Educational content with icons and text",
            duration_range=(30, 180),
            supports_text=True,
            supports_audio=True,
            supports_voiceover=True
        )
    }
    
    def __init__(self, api_key: str = None, config: RenderforestConfig = None):
        """
        Initialize Renderforest integration.
        
        Args:
            api_key: Renderforest API key (can also be set via environment)
            config: Custom configuration object
        """
        if config:
            self.config = config
        else:
            # Get API key from parameter or environment
            api_key = api_key or os.getenv('RENDERFOREST_API_KEY')
            if not api_key:
                raise ValueError("Renderforest API key is required. Set RENDERFOREST_API_KEY environment variable or pass api_key parameter.")
            
            self.config = RenderforestConfig(api_key=api_key)
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })
        
        logger.info("Renderforest integration initialized")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make authenticated request to Renderforest API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response object
        """
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                response.raise_for_status()
                return response
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information and credits.
        
        Returns:
            Account information dictionary
        """
        try:
            response = self._make_request('GET', '/api/v2/users/current')
            return response.json()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def list_templates(self, category: str = None) -> List[Dict[str, Any]]:
        """
        List available templates.
        
        Args:
            category: Template category filter
            
        Returns:
            List of template information
        """
        try:
            endpoint = '/api/v2/templates'
            params = {'category': category} if category else {}
            
            response = self._make_request('GET', endpoint, params=params)
            return response.json().get('data', [])
        
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
    
    def get_template_info(self, template_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific template.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template information dictionary
        """
        try:
            response = self._make_request('GET', f'/api/v2/templates/{template_id}')
            return response.json()
        except Exception as e:
            logger.error(f"Error getting template info: {e}")
            return {}
    
    def _prepare_text_scenes(self, text_content: str, max_scenes: int = 10) -> List[Dict[str, Any]]:
        """
        Prepare text content for video scenes.
        
        Args:
            text_content: Text to convert to scenes
            max_scenes: Maximum number of scenes
            
        Returns:
            List of scene data
        """
        # Split text into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text_content.strip())
        
        # Group sentences into scenes (2-3 sentences per scene)
        scenes = []
        scene_text = ""
        sentence_count = 0
        
        for sentence in sentences:
            if sentence_count < 3 and len(scene_text) < 150:  # Max chars per scene
                scene_text += sentence + " "
                sentence_count += 1
            else:
                if scene_text.strip():
                    scenes.append({
                        'text': scene_text.strip(),
                        'duration': min(max(len(scene_text.split()) * 0.5, 3), 8)  # Duration based on word count
                    })
                scene_text = sentence + " "
                sentence_count = 1
        
        # Add the last scene
        if scene_text.strip():
            scenes.append({
                'text': scene_text.strip(),
                'duration': min(max(len(scene_text.split()) * 0.5, 3), 8)
            })
        
        # Limit to max_scenes
        return scenes[:max_scenes]
    
    def create_video_project(
        self,
        template_name: str,
        text_content: str,
        title: str = "AI Generated Video",
        style_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a video project with text content.
        
        Args:
            template_name: Name of template from SCRIPT_TEMPLATES
            text_content: Text content for the video
            title: Video title
            style_options: Additional styling options
            
        Returns:
            Project creation response
        """
        if template_name not in self.SCRIPT_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(self.SCRIPT_TEMPLATES.keys())}")
        
        template = self.SCRIPT_TEMPLATES[template_name]
        style_options = style_options or {}
        
        # Prepare scenes from text
        scenes = self._prepare_text_scenes(text_content)
        
        if not scenes:
            raise ValueError("No valid scenes could be created from the text content")
        
        # Calculate total duration
        total_duration = sum(scene['duration'] for scene in scenes)
        
        # Ensure duration is within template limits
        min_duration, max_duration = template.duration_range
        if total_duration < min_duration:
            # Extend scene durations proportionally
            factor = min_duration / total_duration
            for scene in scenes:
                scene['duration'] *= factor
        elif total_duration > max_duration:
            # Reduce scene durations proportionally
            factor = max_duration / total_duration
            for scene in scenes:
                scene['duration'] *= factor
        
        # Prepare project data
        project_data = {
            'templateId': template.template_id,
            'title': title,
            'scenes': []
        }
        
        # Add scenes to project
        for i, scene in enumerate(scenes):
            scene_data = {
                'id': i + 1,
                'text': scene['text'],
                'duration': scene['duration']
            }
            
            # Add style options
            if 'font_family' in style_options:
                scene_data['fontFamily'] = style_options['font_family']
            if 'font_size' in style_options:
                scene_data['fontSize'] = style_options['font_size']
            if 'text_color' in style_options:
                scene_data['textColor'] = style_options['text_color']
            if 'background_color' in style_options:
                scene_data['backgroundColor'] = style_options['background_color']
            
            project_data['scenes'].append(scene_data)
        
        try:
            response = self._make_request('POST', '/api/v2/projects', json=project_data)
            result = response.json()
            
            logger.info(f"Created video project with ID: {result.get('projectId')}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating video project: {e}")
            raise
    
    def add_voiceover_to_project(
        self,
        project_id: int,
        audio_file_path: Path,
        voiceover_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Add voiceover audio to a video project.
        
        Args:
            project_id: Project ID
            audio_file_path: Path to audio file
            voiceover_options: Voiceover configuration options
            
        Returns:
            Response from voiceover addition
        """
        voiceover_options = voiceover_options or {}
        
        # First, upload the audio file
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                upload_response = self._make_request(
                    'POST',
                    '/api/v2/assets/upload',
                    files=files
                )
                audio_asset = upload_response.json()
            
            # Add voiceover to project
            voiceover_data = {
                'projectId': project_id,
                'audioAssetId': audio_asset['assetId'],
                'volume': voiceover_options.get('volume', 1.0),
                'fadeIn': voiceover_options.get('fade_in', 0.5),
                'fadeOut': voiceover_options.get('fade_out', 0.5)
            }
            
            response = self._make_request('POST', '/api/v2/projects/voiceover', json=voiceover_data)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error adding voiceover: {e}")
            raise
    
    def render_video(
        self,
        project_id: int,
        quality: str = 'high',
        format: str = 'mp4'
    ) -> Dict[str, Any]:
        """
        Start video rendering process.
        
        Args:
            project_id: Project ID to render
            quality: Video quality ('low', 'medium', 'high')
            format: Output format ('mp4', 'mov')
            
        Returns:
            Render job information
        """
        render_data = {
            'projectId': project_id,
            'quality': quality,
            'format': format
        }
        
        try:
            response = self._make_request('POST', '/api/v2/renders', json=render_data)
            result = response.json()
            
            logger.info(f"Started rendering job with ID: {result.get('renderId')}")
            return result
            
        except Exception as e:
            logger.error(f"Error starting render: {e}")
            raise
    
    def get_render_status(self, render_id: int) -> Dict[str, Any]:
        """
        Get status of a render job.
        
        Args:
            render_id: Render job ID
            
        Returns:
            Render status information
        """
        try:
            response = self._make_request('GET', f'/api/v2/renders/{render_id}')
            return response.json()
        except Exception as e:
            logger.error(f"Error getting render status: {e}")
            return {}
    
    def wait_for_render_completion(
        self,
        render_id: int,
        check_interval: int = 30,
        max_wait_time: int = None
    ) -> Dict[str, Any]:
        """
        Wait for render to complete and return final status.
        
        Args:
            render_id: Render job ID
            check_interval: Seconds between status checks
            max_wait_time: Maximum time to wait (uses config timeout if None)
            
        Returns:
            Final render status
        """
        max_wait_time = max_wait_time or self.config.timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_render_status(render_id)
            
            if not status:
                logger.error("Could not get render status")
                break
            
            state = status.get('state', 'unknown')
            progress = status.get('progress', 0)
            
            logger.info(f"Render progress: {progress}% (state: {state})")
            
            if state == 'completed':
                logger.info("Render completed successfully")
                return status
            elif state == 'failed':
                logger.error(f"Render failed: {status.get('error', 'Unknown error')}")
                break
            
            time.sleep(check_interval)
        
        logger.error("Render timeout or failed")
        return {}
    
    def download_video(self, download_url: str, output_path: Path) -> bool:
        """
        Download rendered video from URL.
        
        Args:
            download_url: Video download URL
            output_path: Local path to save video
            
        Returns:
            True if download successful
        """
        try:
            # Make sure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Video downloaded to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return False
    
    def create_complete_video(
        self,
        text_content: str,
        template_name: str = 'minimal_typography',
        title: str = "AI Generated Video",
        audio_file_path: Path = None,
        output_path: Path = None,
        style_options: Dict[str, Any] = None,
        quality: str = 'high'
    ) -> Optional[Path]:
        """
        Complete workflow: create project, add audio, render, and download video.
        
        Args:
            text_content: Text content for video
            template_name: Template to use
            title: Video title
            audio_file_path: Optional audio file to add
            output_path: Where to save the video
            style_options: Visual styling options
            quality: Render quality
            
        Returns:
            Path to downloaded video file or None if failed
        """
        try:
            # Create project
            logger.info("Creating video project...")
            project_result = self.create_video_project(
                template_name=template_name,
                text_content=text_content,
                title=title,
                style_options=style_options
            )
            
            project_id = project_result.get('projectId')
            if not project_id:
                logger.error("Failed to create project")
                return None
            
            # Add voiceover if provided
            if audio_file_path and audio_file_path.exists():
                logger.info("Adding voiceover...")
                self.add_voiceover_to_project(project_id, audio_file_path)
            
            # Start rendering
            logger.info("Starting video render...")
            render_result = self.render_video(project_id, quality=quality)
            
            render_id = render_result.get('renderId')
            if not render_id:
                logger.error("Failed to start render")
                return None
            
            # Wait for completion
            logger.info("Waiting for render to complete...")
            final_status = self.wait_for_render_completion(render_id)
            
            download_url = final_status.get('downloadUrl')
            if not download_url:
                logger.error("Render failed or no download URL provided")
                return None
            
            # Download video
            if output_path is None:
                output_path = Path(f"renderforest_video_{project_id}.mp4")
            
            if self.download_video(download_url, output_path):
                return output_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in complete video creation: {e}")
            return None
    
    @classmethod
    def get_available_templates(cls) -> Dict[str, VideoTemplate]:
        """Get available script templates."""
        return cls.SCRIPT_TEMPLATES