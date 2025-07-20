"""
Advanced Logging System for ML Pipeline

This module provides structured, production-ready logging with:
- Performance monitoring and timing
- Error tracking and alerting
- Model training metrics logging
- Structured JSON logging for production
- Automatic log rotation and retention
- Integration with monitoring systems
"""

import sys
import json
import time
import functools
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager

from loguru import logger
import mlflow


class MLPipelineLogger:
    """
    Advanced logger for ML Pipeline with MLflow integration and performance monitoring
    
    Features:
    - Structured logging with context
    - Performance timing decorators
    - MLflow integration for experiment tracking
    - Error tracking with stack traces
    - Configurable output formats
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 json_logging: bool = False,
                 mlflow_tracking: bool = True):
        """
        Initialize the ML Pipeline Logger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            json_logging: Enable JSON structured logging
            mlflow_tracking: Enable MLflow experiment tracking
        """
        self.log_level = log_level
        self.log_file = log_file
        self.json_logging = json_logging
        self.mlflow_tracking = mlflow_tracking
        
        # Remove default logger
        logger.remove()
        
        # Configure console logging
        self._setup_console_logging()
        
        # Configure file logging if specified
        if log_file:
            self._setup_file_logging()
        
        # MLflow tracking setup
        if mlflow_tracking:
            self._setup_mlflow_tracking()
    
    def _setup_console_logging(self):
        """Setup console logging with appropriate formatting"""
        if self.json_logging:
            # JSON format for production
            logger.add(
                sys.stderr,
                level=self.log_level,
                format=self._json_formatter,
                serialize=False
            )
        else:
            # Human-readable format for development
            logger.add(
                sys.stderr,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
    
    def _setup_file_logging(self):
        """Setup file logging with rotation and retention"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.json_logging:
            # JSON format for log aggregation systems
            logger.add(
                self.log_file,
                level=self.log_level,
                format=self._json_formatter,
                rotation="1 week",
                retention="1 month",
                compression="gz",
                serialize=False
            )
        else:
            # Standard format for human reading
            logger.add(
                self.log_file,
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="1 week",
                retention="1 month",
                compression="gz"
            )
    
    def _setup_mlflow_tracking(self):
        """Setup MLflow for experiment tracking"""
        try:
            # Set tracking URI if not already set
            if not mlflow.get_tracking_uri():
                mlflow.set_tracking_uri("file:./mlruns")
            
            # Create default experiment if it doesn't exist
            experiment_name = "ml_pipeline_default"
            try:
                mlflow.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException:
                pass  # Experiment already exists
            
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.warning(f"Failed to setup MLflow tracking: {e}")
            self.mlflow_tracking = False
    
    def _json_formatter(self, record):
        """Custom JSON formatter for structured logging"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        
        # Add extra context if available
        if hasattr(record, "extra"):
            log_entry.update(record["extra"])
        
        return json.dumps(log_entry)
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """Log the start of an ML experiment"""
        logger.info(f"Starting ML experiment: {experiment_name}")
        
        if self.mlflow_tracking:
            mlflow.start_run(run_name=experiment_name)
            
            # Log configuration parameters
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
    
    def log_experiment_end(self, status: str = "FINISHED"):
        """Log the end of an ML experiment"""
        logger.info(f"ML experiment completed with status: {status}")
        
        if self.mlflow_tracking and mlflow.active_run():
            mlflow.end_run(status=status)
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        logger.info(f"Metric {metric_name}: {value}")
        
        if self.mlflow_tracking and mlflow.active_run():
            mlflow.log_metric(metric_name, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log an artifact (file) to the experiment"""
        logger.info(f"Logging artifact: {artifact_path}")
        
        if self.mlflow_tracking and mlflow.active_run():
            mlflow.log_artifact(artifact_path, artifact_name)
    
    def log_model(self, model, model_name: str, **kwargs):
        """Log a trained model"""
        logger.info(f"Logging model: {model_name}")
        
        if self.mlflow_tracking and mlflow.active_run():
            # Try to determine model type and log accordingly
            model_type = type(model).__name__
            
            if hasattr(mlflow, 'sklearn') and 'sklearn' in str(type(model)):
                mlflow.sklearn.log_model(model, model_name, **kwargs)
            elif hasattr(mlflow, 'tensorflow') and 'tensorflow' in str(type(model)):
                mlflow.tensorflow.log_model(model, model_name, **kwargs)
            elif hasattr(mlflow, 'pytorch') and 'torch' in str(type(model)):
                mlflow.pytorch.log_model(model, model_name, **kwargs)
            else:
                # Fallback to pickle
                mlflow.log_artifact(model_name, f"models/{model_name}")
    
    def log_data_profile(self, data_profile: Dict[str, Any]):
        """Log data profiling information"""
        logger.info("Logging data profile information")
        
        # Log basic data statistics
        for key, value in data_profile.items():
            if isinstance(value, (int, float)):
                self.log_metric(f"data_{key}", value)
            else:
                logger.info(f"Data {key}: {value}")
    
    def log_performance_warning(self, operation: str, duration: float, threshold: float = 60.0):
        """Log performance warnings for slow operations"""
        if duration > threshold:
            logger.warning(
                f"Performance Warning: {operation} took {duration:.2f}s (threshold: {threshold}s)"
            )
    
    @contextmanager
    def log_execution_time(self, operation_name: str, log_level: str = "INFO"):
        """Context manager to log execution time of operations"""
        start_time = time.time()
        logger.log(log_level, f"Starting {operation_name}")
        
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {operation_name} after {duration:.2f}s: {e}")
            raise
        else:
            duration = time.time() - start_time
            logger.log(log_level, f"Completed {operation_name} in {duration:.2f}s")
            
            # Log to MLflow if enabled
            if self.mlflow_tracking and mlflow.active_run():
                mlflow.log_metric(f"{operation_name}_duration_seconds", duration)
            
            # Check for performance warnings
            self.log_performance_warning(operation_name, duration)


def timing_decorator(operation_name: Optional[str] = None):
    """
    Decorator to automatically log execution time of functions
    
    Usage:
        @timing_decorator("data_loading")
        def load_data():
            # function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            logger.info(f"Starting {name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {name} in {duration:.2f}s")
                
                # Log to MLflow if available
                if mlflow.active_run():
                    mlflow.log_metric(f"{name}_duration_seconds", duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {name} after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def error_handler(func: Callable) -> Callable:
    """
    Decorator to handle and log errors with full context
    
    Usage:
        @error_handler
        def risky_function():
            # function implementation
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                f"Error in {func.__module__}.{func.__name__}: {e}",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    return wrapper


# Global logger instance
_global_logger: Optional[MLPipelineLogger] = None


def get_logger(config: Optional[Dict[str, Any]] = None) -> MLPipelineLogger:
    """
    Get or create the global logger instance
    
    Args:
        config: Logger configuration dictionary
    
    Returns:
        MLPipelineLogger: Configured logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        if config is None:
            config = {}
        
        _global_logger = MLPipelineLogger(
            log_level=config.get("level", "INFO"),
            log_file=config.get("log_file"),
            json_logging=config.get("json_logging", False),
            mlflow_tracking=config.get("mlflow_tracking", True)
        )
    
    return _global_logger


def setup_logging(config: Dict[str, Any]) -> MLPipelineLogger:
    """
    Setup logging system with given configuration
    
    Args:
        config: Logging configuration
    
    Returns:
        MLPipelineLogger: Configured logger instance
    """
    global _global_logger
    _global_logger = MLPipelineLogger(
        log_level=config.get("level", "INFO"),
        log_file=config.get("log_file"),
        json_logging=config.get("json_logging", False),
        mlflow_tracking=config.get("mlflow_tracking", True)
    )
    
    return _global_logger