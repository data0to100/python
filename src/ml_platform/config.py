"""
Configuration management for the ML Platform.

Supports YAML configuration files and environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from loguru import logger


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="ml_platform", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class CloudConfig(BaseSettings):
    """Cloud storage configuration."""
    
    aws_access_key: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    azure_connection_string: str = Field(default="", env="AZURE_STORAGE_CONNECTION_STRING")
    gcp_credentials_path: str = Field(default="", env="GOOGLE_APPLICATION_CREDENTIALS")


class ModelConfig(BaseSettings):
    """Model configuration settings."""
    
    default_test_size: float = Field(default=0.2, env="MODEL_TEST_SIZE")
    random_state: int = Field(default=42, env="MODEL_RANDOM_STATE")
    cv_folds: int = Field(default=5, env="MODEL_CV_FOLDS")
    max_evals: int = Field(default=100, env="MODEL_MAX_EVALS")
    
    # XGBoost defaults
    xgb_n_estimators: int = Field(default=100, env="XGB_N_ESTIMATORS")
    xgb_max_depth: int = Field(default=6, env="XGB_MAX_DEPTH")
    xgb_learning_rate: float = Field(default=0.1, env="XGB_LEARNING_RATE")
    
    # LightGBM defaults
    lgb_n_estimators: int = Field(default=100, env="LGB_N_ESTIMATORS")
    lgb_max_depth: int = Field(default=-1, env="LGB_MAX_DEPTH")
    lgb_learning_rate: float = Field(default=0.1, env="LGB_LEARNING_RATE")


class MLConfig:
    """Main configuration class for the ML Platform."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self._config_data = self._load_config()
        
        # Initialize sub-configurations
        self.database = DatabaseConfig()
        self.cloud = CloudConfig()
        self.model = ModelConfig()
        
        # General settings
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        possible_paths = [
            "config/ml_platform.yaml",
            "config.yaml", 
            "ml_platform.yaml",
            os.path.expanduser("~/.ml_platform/config.yaml")
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        # Create default config if none found
        default_path = "config/ml_platform.yaml"
        self._create_default_config(default_path)
        return default_path
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not Path(self.config_path).exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
            
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}
    
    def _create_default_config(self, path: str) -> None:
        """Create default configuration file."""
        config_dir = Path(path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'ml_platform',
                'username': 'postgres',
                'password': ''
            },
            'cloud': {
                'aws_region': 'us-east-1',
                'azure_storage_account': '',
                'gcp_project_id': ''
            },
            'models': {
                'default_test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5,
                'max_evals': 100
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'cache': {
                'enabled': True,
                'redis_url': 'redis://localhost:6379/0',
                'ttl': 3600
            }
        }
        
        try:
            with open(path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created default config file: {path}")
        except IOError as e:
            logger.error(f"Failed to create default config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation like 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._config_data.update(updates)
        logger.info("Configuration updated")
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration (defaults to current config_path)
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {save_path}")
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")