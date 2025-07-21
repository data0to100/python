"""
Logging configuration for the ML Platform.

Provides structured logging with loguru, including rotation, filtering, and formatting.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json


class MLLogger:
    """Enterprise-grade logger for the ML Platform."""
    
    def __init__(self, name: str = "ml_platform", level: str = "INFO", log_dir: str = "logs"):
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Directory for log files
        """
        self.name = name
        self.level = level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default handler
        logger.remove()
        
        # Configure loggers
        self._setup_console_logging()
        self._setup_file_logging()
        self._setup_error_logging()
        self._setup_performance_logging()
    
    def _setup_console_logging(self):
        """Setup console logging with colored output."""
        logger.add(
            sys.stdout,
            level=self.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            filter=lambda record: record["level"].name != "ERROR"
        )
    
    def _setup_file_logging(self):
        """Setup general file logging with rotation."""
        log_file = self.log_dir / f"{self.name}.log"
        logger.add(
            log_file,
            level=self.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            encoding="utf-8"
        )
    
    def _setup_error_logging(self):
        """Setup error-specific logging."""
        error_log = self.log_dir / f"{self.name}_errors.log"
        logger.add(
            error_log,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
            rotation="50 MB",
            retention="90 days",
            compression="gz",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
    
    def _setup_performance_logging(self):
        """Setup performance-specific logging."""
        perf_log = self.log_dir / f"{self.name}_performance.log"
        logger.add(
            perf_log,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | PERF | {message}",
            rotation="50 MB",
            retention="7 days",
            compression="gz",
            encoding="utf-8",
            filter=lambda record: record.get("extra", {}).get("performance", False)
        )
    
    def get_logger(self):
        """Get the configured logger instance."""
        return logger.bind(name=self.name)


# Global logger instance
_logger_instance: Optional[MLLogger] = None


def get_logger(name: str = "ml_platform", level: Optional[str] = None, log_dir: str = "logs"):
    """Get or create logger instance.
    
    Args:
        name: Logger name
        level: Logging level (defaults to environment variable LOG_LEVEL or INFO)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    global _logger_instance
    
    if _logger_instance is None:
        actual_level = level or os.getenv("LOG_LEVEL", "INFO")
        _logger_instance = MLLogger(name=name, level=actual_level, log_dir=log_dir)
    
    return _logger_instance.get_logger()


def log_performance(operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        metadata: Additional metadata to log
    """
    log_data = {
        "operation": operation,
        "duration_seconds": round(duration, 4),
        "metadata": metadata or {}
    }
    
    logger.bind(performance=True).info(
        f"Performance: {operation} completed in {duration:.4f}s",
        extra={"performance_data": log_data}
    )


def log_data_quality(dataset_name: str, quality_metrics: Dict[str, Any]):
    """Log data quality metrics.
    
    Args:
        dataset_name: Name of the dataset
        quality_metrics: Dictionary of quality metrics
    """
    logger.info(
        f"Data Quality Assessment: {dataset_name}",
        extra={
            "data_quality": True,
            "dataset": dataset_name,
            "metrics": quality_metrics
        }
    )


def log_model_metrics(model_name: str, model_type: str, metrics: Dict[str, Any], stage: str = "training"):
    """Log model performance metrics.
    
    Args:
        model_name: Name of the model
        model_type: Type of model (xgboost, lightgbm, etc.)
        metrics: Dictionary of model metrics
        stage: Stage of the model (training, validation, testing)
    """
    logger.info(
        f"Model Metrics: {model_name} ({model_type}) - {stage}",
        extra={
            "model_metrics": True,
            "model_name": model_name,
            "model_type": model_type,
            "stage": stage,
            "metrics": metrics
        }
    )


def log_api_call(endpoint: str, method: str, status_code: int, duration: float, 
                 request_size: Optional[int] = None, response_size: Optional[int] = None):
    """Log API call details.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: Response status code
        duration: Request duration in seconds
        request_size: Size of request in bytes
        response_size: Size of response in bytes
    """
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration_seconds": round(duration, 4)
    }
    
    if request_size is not None:
        log_data["request_size_bytes"] = request_size
    if response_size is not None:
        log_data["response_size_bytes"] = response_size
    
    level = "ERROR" if status_code >= 400 else "INFO"
    logger.log(
        level,
        f"API Call: {method} {endpoint} -> {status_code} ({duration:.4f}s)",
        extra={"api_call": True, "call_data": log_data}
    )


def log_cache_operation(operation: str, key: str, hit: bool, duration: Optional[float] = None):
    """Log cache operation details.
    
    Args:
        operation: Cache operation (get, set, delete)
        key: Cache key
        hit: Whether it was a cache hit
        duration: Operation duration in seconds
    """
    log_data = {
        "operation": operation,
        "key": key,
        "hit": hit
    }
    
    if duration is not None:
        log_data["duration_seconds"] = round(duration, 4)
    
    logger.info(
        f"Cache: {operation} {key} - {'HIT' if hit else 'MISS'}",
        extra={"cache_operation": True, "cache_data": log_data}
    )


class StructuredLogger:
    """Context manager for structured logging with automatic metadata."""
    
    def __init__(self, operation: str, **metadata):
        """Initialize structured logger.
        
        Args:
            operation: Name of the operation
            **metadata: Additional metadata to include in all log messages
        """
        self.operation = operation
        self.metadata = metadata
        self.logger = logger.bind(**metadata)
        self.start_time = None
    
    def __enter__(self):
        """Enter the context."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation} in {duration:.4f}s")
            log_performance(self.operation, duration, self.metadata)
        else:
            self.logger.error(
                f"Failed operation: {self.operation} after {duration:.4f}s",
                extra={"exception": str(exc_val)}
            )


# Convenience function for structured logging
def structured_logging(operation: str, **metadata):
    """Create a structured logging context.
    
    Args:
        operation: Name of the operation
        **metadata: Additional metadata to include
        
    Returns:
        StructuredLogger context manager
    """
    return StructuredLogger(operation, **metadata)