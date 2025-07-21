"""
Enterprise AI/ML Platform Module

A production-ready machine learning platform with automated insights,
predictive modeling, and advanced visualization capabilities.
"""

from .config import MLConfig
from .exceptions import MLPlatformError, DataLoadError, ModelError
from .logger import get_logger

__version__ = "1.0.0"
__author__ = "Enterprise AI/ML Platform"

__all__ = [
    "MLConfig",
    "MLPlatformError", 
    "DataLoadError",
    "ModelError",
    "get_logger"
]