"""
Data Analyst Toolkit - Comprehensive analysis tools for data professionals.

This package provides a complete toolkit for data analysis including:
- Data loading and validation
- Data profiling and quality assessment
- Advanced visualizations
- Machine learning pipelines
- Model interpretation
"""

__version__ = "1.0.0"
__author__ = "Data Analysis Team"
__email__ = "data@example.com"

# Convenient imports for users
from .utils.data_loader import DataLoader, DataValidator
from .utils.data_quality_monitor import DataQualityMonitor
from .analysis.data_profiler import DataProfiler, DataComparer
from .analysis.automated_insights import AutomatedInsightsGenerator
from .visualization.advanced_plots import AdvancedVisualizer, InteractiveDashboard
from .ml_models.model_pipeline import AutoMLPipeline, ModelInterpreter

__all__ = [
    "DataLoader",
    "DataValidator",
    "DataQualityMonitor", 
    "DataProfiler",
    "DataComparer",
    "AutomatedInsightsGenerator",
    "AdvancedVisualizer",
    "InteractiveDashboard",
    "AutoMLPipeline",
    "ModelInterpreter"
]

# Package metadata
PACKAGE_INFO = {
    "name": "data-analyst-toolkit",
    "version": __version__,
    "description": "Comprehensive toolkit for data analysts and data scientists",
    "author": __author__,
    "email": __email__,
    "license": "MIT",
    "url": "https://github.com/example/data-analyst-toolkit"
}