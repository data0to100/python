"""
Custom exceptions for the ML Platform.

Provides specific exception types for different components and error scenarios.
"""

from typing import Optional, Dict, Any


class MLPlatformError(Exception):
    """Base exception for ML Platform errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize ML Platform error.
        
        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        return base_msg


class DataLoadError(MLPlatformError):
    """Exception raised when data loading fails."""
    
    def __init__(self, message: str, source: Optional[str] = None, source_type: Optional[str] = None):
        """Initialize data load error.
        
        Args:
            message: Error message
            source: Data source that failed (file path, URL, etc.)
            source_type: Type of data source (csv, sql, api, etc.)
        """
        details = {}
        if source:
            details['source'] = source
        if source_type:
            details['source_type'] = source_type
            
        super().__init__(message, error_code="DATA_LOAD_ERROR", details=details)


class ModelError(MLPlatformError):
    """Exception raised when model operations fail."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, stage: Optional[str] = None):
        """Initialize model error.
        
        Args:
            message: Error message
            model_type: Type of model (xgboost, lightgbm, prophet, etc.)
            stage: Stage where error occurred (training, prediction, evaluation, etc.)
        """
        details = {}
        if model_type:
            details['model_type'] = model_type
        if stage:
            details['stage'] = stage
            
        super().__init__(message, error_code="MODEL_ERROR", details=details)


class ValidationError(MLPlatformError):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, failed_checks: Optional[list] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            validation_type: Type of validation (schema, quality, etc.)
            failed_checks: List of failed validation checks
        """
        details = {}
        if validation_type:
            details['validation_type'] = validation_type
        if failed_checks:
            details['failed_checks'] = failed_checks
            
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)


class ConfigurationError(MLPlatformError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, expected_type: Optional[str] = None):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            expected_type: Expected type for the configuration value
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if expected_type:
            details['expected_type'] = expected_type
            
        super().__init__(message, error_code="CONFIG_ERROR", details=details)


class ConnectionError(MLPlatformError):
    """Exception raised when external connections fail."""
    
    def __init__(self, message: str, service: Optional[str] = None, endpoint: Optional[str] = None):
        """Initialize connection error.
        
        Args:
            message: Error message
            service: Service that failed (database, api, cloud storage, etc.)
            endpoint: Endpoint that failed to connect
        """
        details = {}
        if service:
            details['service'] = service
        if endpoint:
            details['endpoint'] = endpoint
            
        super().__init__(message, error_code="CONNECTION_ERROR", details=details)


class VisualizationError(MLPlatformError):
    """Exception raised when visualization generation fails."""
    
    def __init__(self, message: str, viz_type: Optional[str] = None, library: Optional[str] = None):
        """Initialize visualization error.
        
        Args:
            message: Error message
            viz_type: Type of visualization (scatter, line, heatmap, etc.)
            library: Visualization library (plotly, altair, matplotlib, etc.)
        """
        details = {}
        if viz_type:
            details['viz_type'] = viz_type
        if library:
            details['library'] = library
            
        super().__init__(message, error_code="VISUALIZATION_ERROR", details=details)


class CacheError(MLPlatformError):
    """Exception raised when cache operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, cache_key: Optional[str] = None):
        """Initialize cache error.
        
        Args:
            message: Error message
            operation: Cache operation that failed (get, set, delete, etc.)
            cache_key: Cache key that caused the error
        """
        details = {}
        if operation:
            details['operation'] = operation
        if cache_key:
            details['cache_key'] = cache_key
            
        super().__init__(message, error_code="CACHE_ERROR", details=details)