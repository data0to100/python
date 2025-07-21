"""
Multi-source data ingestion module for the ML Platform.

Supports CSV files, SQL databases, REST APIs, and cloud storage providers.
"""

import io
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine, text
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import MLConfig
from .exceptions import DataLoadError, ConnectionError as MLConnectionError
from .logger import get_logger, log_performance, structured_logging


class DataLoader:
    """Enterprise-grade data loader with multi-source support."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        """Initialize data loader.
        
        Args:
            config: ML Platform configuration
        """
        self.config = config or MLConfig()
        self.logger = get_logger()
        self._engines = {}  # Cache for database engines
        
    def load_data(self, source: str, source_type: str = "auto", **kwargs) -> pd.DataFrame:
        """Load data from various sources.
        
        Args:
            source: Data source (file path, URL, connection string, etc.)
            source_type: Type of source (csv, sql, api, s3, azure, gcs, auto)
            **kwargs: Additional parameters for data loading
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataLoadError: If data loading fails
        """
        with structured_logging("data_load", source=source, source_type=source_type):
            try:
                # Auto-detect source type if needed
                if source_type == "auto":
                    source_type = self._detect_source_type(source)
                
                # Route to appropriate loader
                if source_type == "csv":
                    return self._load_csv(source, **kwargs)
                elif source_type == "sql":
                    return self._load_sql(source, **kwargs)
                elif source_type == "api":
                    return self._load_api(source, **kwargs)
                elif source_type == "s3":
                    return self._load_s3(source, **kwargs)
                elif source_type == "azure":
                    return self._load_azure(source, **kwargs)
                elif source_type == "gcs":
                    return self._load_gcs(source, **kwargs)
                else:
                    raise DataLoadError(
                        f"Unsupported source type: {source_type}",
                        source=source,
                        source_type=source_type
                    )
                    
            except Exception as e:
                if isinstance(e, DataLoadError):
                    raise
                raise DataLoadError(
                    f"Failed to load data: {str(e)}",
                    source=source,
                    source_type=source_type
                )
    
    def _detect_source_type(self, source: str) -> str:
        """Auto-detect the source type from the source string.
        
        Args:
            source: Data source string
            
        Returns:
            Detected source type
        """
        source_lower = source.lower()
        
        # URL-based detection
        if source_lower.startswith(('http://', 'https://')):
            if source_lower.endswith(('.csv', '.tsv')):
                return "csv"
            return "api"
        
        # Cloud storage detection
        if source_lower.startswith('s3://'):
            return "s3"
        if source_lower.startswith('az://') or source_lower.startswith('azure://'):
            return "azure"
        if source_lower.startswith('gs://') or source_lower.startswith('gcs://'):
            return "gcs"
        
        # Database detection
        if any(db in source_lower for db in ['postgresql://', 'mysql://', 'sqlite://', 'oracle://']):
            return "sql"
        
        # File extension detection
        if source_lower.endswith(('.csv', '.tsv')):
            return "csv"
        if source_lower.endswith(('.xlsx', '.xls')):
            return "excel"
        if source_lower.endswith(('.json', '.jsonl')):
            return "json"
        if source_lower.endswith('.parquet'):
            return "parquet"
        
        # Default to CSV for files
        return "csv"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _load_csv(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file or URL.
        
        Args:
            source: File path or URL
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            DataFrame
        """
        start_time = time.time()
        
        try:
            # Set default parameters
            csv_kwargs = {
                'encoding': 'utf-8',
                'low_memory': False,
                'parse_dates': True,
                'infer_datetime_format': True,
                **kwargs
            }
            
            if source.startswith(('http://', 'https://')):
                # Handle URLs
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text), **csv_kwargs)
            else:
                # Handle local files
                df = pd.read_csv(source, **csv_kwargs)
            
            duration = time.time() - start_time
            log_performance("csv_load", duration, {"rows": len(df), "columns": len(df.columns)})
            
            self.logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns from {source}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV: {str(e)}", source=source, source_type="csv")
    
    def _load_sql(self, query_or_table: str, connection_string: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from SQL database.
        
        Args:
            query_or_table: SQL query or table name
            connection_string: Database connection string
            **kwargs: Additional parameters
            
        Returns:
            DataFrame
        """
        start_time = time.time()
        
        try:
            # Use provided connection string or default from config
            conn_str = connection_string or self.config.database.connection_string
            
            if not conn_str:
                raise DataLoadError("No database connection string provided", source_type="sql")
            
            # Get or create engine
            if conn_str not in self._engines:
                self._engines[conn_str] = create_engine(conn_str)
            
            engine = self._engines[conn_str]
            
            # Determine if it's a query or table name
            if any(keyword in query_or_table.upper() for keyword in ['SELECT', 'WITH', 'FROM']):
                # It's a query
                df = pd.read_sql_query(text(query_or_table), engine, **kwargs)
            else:
                # It's a table name
                df = pd.read_sql_table(query_or_table, engine, **kwargs)
            
            duration = time.time() - start_time
            log_performance("sql_load", duration, {"rows": len(df), "columns": len(df.columns)})
            
            self.logger.info(f"Loaded SQL: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load from SQL: {str(e)}", source_type="sql")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _load_api(self, url: str, **kwargs) -> pd.DataFrame:
        """Load data from REST API.
        
        Args:
            url: API endpoint URL
            **kwargs: Additional parameters (headers, auth, etc.)
            
        Returns:
            DataFrame
        """
        start_time = time.time()
        
        try:
            # Extract request parameters
            headers = kwargs.pop('headers', {})
            auth = kwargs.pop('auth', None)
            params = kwargs.pop('params', {})
            method = kwargs.pop('method', 'GET').upper()
            data = kwargs.pop('data', None)
            json_data = kwargs.pop('json', None)
            timeout = kwargs.pop('timeout', 30)
            
            # Make API request
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                auth=auth,
                params=params,
                data=data,
                json=json_data,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to find the data array in the response
                    data_key = kwargs.get('data_key', 'data')
                    if data_key in data:
                        df = pd.DataFrame(data[data_key])
                    else:
                        # Flatten the dict if it's a single record
                        df = pd.DataFrame([data])
                else:
                    raise DataLoadError("Unsupported JSON format", source=url, source_type="api")
            elif 'text/csv' in content_type or url.endswith('.csv'):
                df = pd.read_csv(io.StringIO(response.text), **kwargs)
            else:
                # Try to parse as JSON first, then CSV
                try:
                    data = response.json()
                    df = pd.DataFrame(data)
                except:
                    df = pd.read_csv(io.StringIO(response.text), **kwargs)
            
            duration = time.time() - start_time
            log_performance("api_load", duration, {"rows": len(df), "columns": len(df.columns)})
            
            self.logger.info(f"Loaded API: {len(df)} rows, {len(df.columns)} columns from {url}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load from API: {str(e)}", source=url, source_type="api")
    
    def _load_s3(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Load data from Amazon S3.
        
        Args:
            s3_path: S3 path (s3://bucket/key)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame
        """
        start_time = time.time()
        
        try:
            # Parse S3 path
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.cloud.aws_access_key,
                aws_secret_access_key=self.config.cloud.aws_secret_key,
                region_name=self.config.cloud.aws_region
            )
            
            # Get object
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            
            # Determine file type and load accordingly
            if key.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            elif key.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif key.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                df = pd.DataFrame(data)
            else:
                # Default to CSV
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            
            duration = time.time() - start_time
            log_performance("s3_load", duration, {"rows": len(df), "columns": len(df.columns)})
            
            self.logger.info(f"Loaded S3: {len(df)} rows, {len(df.columns)} columns from {s3_path}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load from S3: {str(e)}", source=s3_path, source_type="s3")
    
    def _load_azure(self, azure_path: str, **kwargs) -> pd.DataFrame:
        """Load data from Azure Blob Storage.
        
        Args:
            azure_path: Azure path (azure://container/blob)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame
        """
        start_time = time.time()
        
        try:
            # Parse Azure path
            parsed = urlparse(azure_path)
            container = parsed.netloc
            blob = parsed.path.lstrip('/')
            
            # Initialize Azure client
            blob_service_client = BlobServiceClient.from_connection_string(
                self.config.cloud.azure_connection_string
            )
            
            # Get blob
            blob_client = blob_service_client.get_blob_client(
                container=container, 
                blob=blob
            )
            content = blob_client.download_blob().readall()
            
            # Determine file type and load accordingly
            if blob.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            elif blob.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif blob.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                df = pd.DataFrame(data)
            else:
                # Default to CSV
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            
            duration = time.time() - start_time
            log_performance("azure_load", duration, {"rows": len(df), "columns": len(df.columns)})
            
            self.logger.info(f"Loaded Azure: {len(df)} rows, {len(df.columns)} columns from {azure_path}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load from Azure: {str(e)}", source=azure_path, source_type="azure")
    
    def _load_gcs(self, gcs_path: str, **kwargs) -> pd.DataFrame:
        """Load data from Google Cloud Storage.
        
        Args:
            gcs_path: GCS path (gs://bucket/object)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame
        """
        start_time = time.time()
        
        try:
            # Parse GCS path
            parsed = urlparse(gcs_path)
            bucket_name = parsed.netloc
            object_name = parsed.path.lstrip('/')
            
            # Initialize GCS client
            client = gcs.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            
            # Download content
            content = blob.download_as_bytes()
            
            # Determine file type and load accordingly
            if object_name.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            elif object_name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif object_name.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                df = pd.DataFrame(data)
            else:
                # Default to CSV
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            
            duration = time.time() - start_time
            log_performance("gcs_load", duration, {"rows": len(df), "columns": len(df.columns)})
            
            self.logger.info(f"Loaded GCS: {len(df)} rows, {len(df.columns)} columns from {gcs_path}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load from GCS: {str(e)}", source=gcs_path, source_type="gcs")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        try:
            info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicates": df.duplicated().sum(),
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                "datetime_columns": list(df.select_dtypes(include=['datetime64']).columns),
            }
            
            # Add statistics for numeric columns
            if info["numeric_columns"]:
                numeric_stats = df[info["numeric_columns"]].describe().to_dict()
                info["numeric_statistics"] = numeric_stats
            
            # Add value counts for categorical columns (top 5)
            if info["categorical_columns"]:
                categorical_stats = {}
                for col in info["categorical_columns"]:
                    try:
                        value_counts = df[col].value_counts().head().to_dict()
                        categorical_stats[col] = value_counts
                    except:
                        categorical_stats[col] = {}
                info["categorical_statistics"] = categorical_stats
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get data info: {str(e)}")
            return {"error": str(e)}
    
    def validate_data(self, df: pd.DataFrame, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate data quality based on rules.
        
        Args:
            df: DataFrame to validate
            rules: Validation rules
            
        Returns:
            Validation results
        """
        results = {
            "passed": True,
            "issues": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Basic validations
            if df.empty:
                results["issues"].append("Dataset is empty")
                results["passed"] = False
            
            # Check for completely null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                results["issues"].append(f"Columns with all null values: {null_columns}")
                results["passed"] = False
            
            # Check for high missing value percentage
            missing_pct = (df.isnull().sum() / len(df) * 100)
            high_missing = missing_pct[missing_pct > 50].to_dict()
            if high_missing:
                results["warnings"].append(f"Columns with >50% missing values: {high_missing}")
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                results["warnings"].append(f"Found {duplicate_count} duplicate rows")
            
            # Custom rules validation
            if rules:
                for rule_name, rule_config in rules.items():
                    try:
                        self._validate_rule(df, rule_name, rule_config, results)
                    except Exception as e:
                        results["issues"].append(f"Failed to validate rule '{rule_name}': {str(e)}")
            
            # Summary
            results["summary"] = {
                "total_issues": len(results["issues"]),
                "total_warnings": len(results["warnings"]),
                "data_quality_score": self._calculate_quality_score(df, results)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return {
                "passed": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "summary": {}
            }
    
    def _validate_rule(self, df: pd.DataFrame, rule_name: str, rule_config: Dict[str, Any], results: Dict[str, Any]):
        """Validate a specific rule.
        
        Args:
            df: DataFrame to validate
            rule_name: Name of the rule
            rule_config: Rule configuration
            results: Results dictionary to update
        """
        rule_type = rule_config.get("type")
        
        if rule_type == "not_null":
            columns = rule_config.get("columns", [])
            for col in columns:
                if col in df.columns and df[col].isnull().any():
                    results["issues"].append(f"Rule '{rule_name}': Column '{col}' has null values")
                    results["passed"] = False
        
        elif rule_type == "unique":
            columns = rule_config.get("columns", [])
            for col in columns:
                if col in df.columns and df[col].duplicated().any():
                    results["issues"].append(f"Rule '{rule_name}': Column '{col}' has duplicate values")
                    results["passed"] = False
        
        elif rule_type == "range":
            column = rule_config.get("column")
            min_val = rule_config.get("min")
            max_val = rule_config.get("max")
            
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                if min_val is not None and (df[column] < min_val).any():
                    results["issues"].append(f"Rule '{rule_name}': Column '{column}' has values below {min_val}")
                    results["passed"] = False
                if max_val is not None and (df[column] > max_val).any():
                    results["issues"].append(f"Rule '{rule_name}': Column '{column}' has values above {max_val}")
                    results["passed"] = False
    
    def _calculate_quality_score(self, df: pd.DataFrame, results: Dict[str, Any]) -> float:
        """Calculate a data quality score.
        
        Args:
            df: DataFrame
            results: Validation results
            
        Returns:
            Quality score between 0 and 100
        """
        score = 100.0
        
        # Deduct for issues
        score -= len(results["issues"]) * 20
        
        # Deduct for warnings
        score -= len(results["warnings"]) * 5
        
        # Deduct for missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= missing_pct
        
        # Deduct for duplicates
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        score -= duplicate_pct * 0.5
        
        return max(0.0, min(100.0, score))