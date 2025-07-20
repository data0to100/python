"""
Advanced ML Pipeline for Large-Scale Data Processing and AutoML

This package provides a production-ready, scalable machine learning pipeline
that automatically handles:
- Large dataset processing with big data frameworks
- Automated outlier detection and data quality assessment
- Time-series vs cross-sectional data detection
- AutoML model selection and training
- Feature engineering and preprocessing
- Model monitoring and versioning
- Production deployment utilities

Author: Expert Data Science Assistant
"""

__version__ = "1.0.0"
__author__ = "Expert Data Science Assistant"

from .core.pipeline import MLPipeline
from .core.data_loader import DataLoader
from .core.preprocessor import DataPreprocessor
from .core.feature_engineer import FeatureEngineer
from .core.model_selector import ModelSelector
from .core.time_series_detector import TimeSeriesDetector
from .core.outlier_detector import OutlierDetector
from .utils.logger import get_logger
from .utils.config import Config
from .utils.visualizer import AdvancedVisualizer

__all__ = [
    "MLPipeline",
    "DataLoader", 
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelSelector",
    "TimeSeriesDetector",
    "OutlierDetector",
    "get_logger",
    "Config",
    "AdvancedVisualizer"
]