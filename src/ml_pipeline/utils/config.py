"""
Configuration Management for ML Pipeline

This module provides a centralized configuration system with:
- Type safety and validation using Pydantic
- Environment-specific configurations
- Model hyperparameter management
- Data processing configurations
- Production deployment settings
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from omegaconf import OmegaConf
import os


class DataConfig(BaseModel):
    """Configuration for data loading and processing"""
    
    # Data source configuration
    data_source: str = Field(default="csv", description="Data source type: csv, sql, parquet")
    file_path: Optional[str] = Field(default=None, description="Path to data file")
    sql_connection_string: Optional[str] = Field(default=None, description="SQL connection string")
    sql_query: Optional[str] = Field(default=None, description="SQL query for data extraction")
    
    # Big data processing
    use_big_data_engine: bool = Field(default=True, description="Use dask/modin for large datasets")
    big_data_engine: str = Field(default="dask", description="Big data engine: dask, modin, vaex")
    chunk_size: int = Field(default=10000, description="Chunk size for processing")
    n_workers: int = Field(default=4, description="Number of workers for parallel processing")
    
    # Data quality
    max_missing_percentage: float = Field(default=0.3, description="Max missing data percentage")
    min_unique_values: int = Field(default=2, description="Minimum unique values for categorical features")
    
    @validator('big_data_engine')
    def validate_engine(cls, v):
        if v not in ['dask', 'modin', 'vaex']:
            raise ValueError('big_data_engine must be one of: dask, modin, vaex')
        return v


class OutlierDetectionConfig(BaseModel):
    """Configuration for outlier detection"""
    
    methods: List[str] = Field(default=["isolation_forest", "z_score", "iqr"], 
                              description="Outlier detection methods to use")
    isolation_forest_contamination: float = Field(default=0.1, description="Contamination rate for Isolation Forest")
    z_score_threshold: float = Field(default=3.0, description="Z-score threshold for outlier detection")
    iqr_multiplier: float = Field(default=1.5, description="IQR multiplier for outlier detection")
    remove_outliers: bool = Field(default=True, description="Whether to remove detected outliers")
    
    @validator('methods')
    def validate_methods(cls, v):
        valid_methods = ["isolation_forest", "z_score", "iqr", "local_outlier_factor", "one_class_svm"]
        for method in v:
            if method not in valid_methods:
                raise ValueError(f'Invalid outlier detection method: {method}')
        return v


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering"""
    
    # Datetime features
    create_datetime_features: bool = Field(default=True, description="Create datetime-based features")
    datetime_features: List[str] = Field(
        default=["year", "month", "day", "hour", "dayofweek", "quarter", "is_weekend"],
        description="Datetime features to create"
    )
    
    # Categorical encoding
    categorical_encoding: str = Field(default="auto", description="Categorical encoding method")
    max_categories: int = Field(default=50, description="Max categories for one-hot encoding")
    
    # Numerical features
    scaling_method: str = Field(default="standard", description="Scaling method: standard, minmax, robust")
    polynomial_features: bool = Field(default=False, description="Create polynomial features")
    polynomial_degree: int = Field(default=2, description="Degree for polynomial features")
    
    # Feature selection
    feature_selection: bool = Field(default=True, description="Perform automatic feature selection")
    feature_selection_method: str = Field(default="mutual_info", description="Feature selection method")
    max_features: Optional[int] = Field(default=None, description="Maximum number of features to select")


class ModelConfig(BaseModel):
    """Configuration for model selection and training"""
    
    # AutoML configuration
    automl_framework: str = Field(default="pycaret", description="AutoML framework: pycaret, h2o, auto_sklearn")
    automl_time_budget: int = Field(default=3600, description="Time budget for AutoML in seconds")
    cross_validation_folds: int = Field(default=5, description="Number of CV folds")
    
    # Time series specific
    time_series_models: List[str] = Field(
        default=["prophet", "arima", "lstm", "seasonal_decompose"],
        description="Time series models to try"
    )
    forecast_horizon: int = Field(default=30, description="Forecast horizon for time series")
    seasonality_detection: bool = Field(default=True, description="Detect seasonality automatically")
    
    # Model evaluation
    primary_metric: str = Field(default="auto", description="Primary evaluation metric")
    optimize_for: str = Field(default="accuracy", description="Optimization target: accuracy, speed, interpretability")
    
    # Model selection criteria
    model_selection_strategy: str = Field(default="best_score", description="Model selection strategy")
    ensemble_methods: bool = Field(default=True, description="Use ensemble methods")


class ProductionConfig(BaseModel):
    """Configuration for production deployment"""
    
    # Model versioning
    model_registry: str = Field(default="mlflow", description="Model registry: mlflow, local")
    model_version_strategy: str = Field(default="timestamp", description="Model versioning strategy")
    
    # Monitoring
    enable_monitoring: bool = Field(default=True, description="Enable model monitoring")
    drift_detection: bool = Field(default=True, description="Enable data drift detection")
    performance_threshold: float = Field(default=0.05, description="Performance degradation threshold")
    
    # Retraining
    auto_retrain: bool = Field(default=False, description="Enable automatic retraining")
    retrain_frequency: str = Field(default="weekly", description="Retraining frequency")
    retrain_threshold: float = Field(default=0.1, description="Performance threshold for retraining")
    
    # Deployment
    deployment_target: str = Field(default="local", description="Deployment target: local, docker, kubernetes")
    api_framework: str = Field(default="fastapi", description="API framework for model serving")


class LoggingConfig(BaseModel):
    """Configuration for logging"""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format"
    )
    log_file: Optional[str] = Field(default="logs/ml_pipeline.log", description="Log file path")
    rotation: str = Field(default="1 week", description="Log rotation schedule")
    retention: str = Field(default="1 month", description="Log retention period")


class Config(BaseModel):
    """Main configuration class that combines all sub-configurations"""
    
    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    outlier_detection: OutlierDetectionConfig = Field(default_factory=OutlierDetectionConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    project_name: str = Field(default="ml_pipeline", description="Project name")
    experiment_name: str = Field(default="default", description="Experiment name")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    output_dir: str = Field(default="outputs", description="Output directory")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML using OmegaConf for advanced features
        omega_conf = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(omega_conf, resolve=True)
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config_dict = {}
        
        # Map environment variables to config structure
        env_mapping = {
            "ML_DATA_SOURCE": ("data", "data_source"),
            "ML_BIG_DATA_ENGINE": ("data", "big_data_engine"),
            "ML_AUTOML_FRAMEWORK": ("model", "automl_framework"),
            "ML_MODEL_REGISTRY": ("production", "model_registry"),
            "ML_LOG_LEVEL": ("logging", "level"),
        }
        
        for env_var, (section, key) in env_mapping.items():
            if env_var in os.environ:
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][key] = os.environ[env_var]
        
        return cls(**config_dict)
    
    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to OmegaConf and save
        omega_conf = OmegaConf.create(self.dict())
        OmegaConf.save(omega_conf, config_path)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> "Config":
        """Update configuration with dictionary values"""
        current_dict = self.dict()
        
        # Deep merge dictionaries
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(current_dict, updates)
        return Config(**current_dict)
    
    def get_model_config_for_type(self, model_type: str) -> Dict[str, Any]:
        """Get specific model configuration based on detected model type"""
        base_config = self.model.dict()
        
        if model_type == "time_series":
            return {
                **base_config,
                "models_to_try": self.model.time_series_models,
                "forecast_horizon": self.model.forecast_horizon,
                "seasonality_detection": self.model.seasonality_detection
            }
        else:
            return {
                **base_config,
                "models_to_try": ["rf", "lgb", "xgboost", "svm", "lr"],  # Cross-sectional models
            }


# Default configuration instance
default_config = Config()


def get_config(config_path: Optional[Union[str, Path]] = None, 
               from_env: bool = False,
               updates: Optional[Dict[str, Any]] = None) -> Config:
    """
    Get configuration with various loading options
    
    Args:
        config_path: Path to YAML configuration file
        from_env: Load configuration from environment variables
        updates: Dictionary of configuration updates
    
    Returns:
        Config: Loaded and updated configuration
    """
    if config_path:
        config = Config.from_yaml(config_path)
    elif from_env:
        config = Config.from_env()
    else:
        config = Config()
    
    if updates:
        config = config.update_from_dict(updates)
    
    return config