"""
Data Loading Utilities for Data Analysis
Supports multiple data sources and formats with robust error handling.
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import warnings
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import requests
from sqlalchemy import create_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Comprehensive data loading utility supporting multiple formats and sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader with optional configuration.
        
        Args:
            config: Configuration dictionary for database connections, APIs, etc.
        """
        self.config = config or {}
        
    def load_csv(self, 
                 filepath: Union[str, Path], 
                 **kwargs) -> pd.DataFrame:
        """
        Load CSV file with intelligent type inference and error handling.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            DataFrame with loaded data
        """
        try:
            # Default parameters for robust CSV loading
            default_params = {
                'encoding': 'utf-8',
                'low_memory': False,
                'parse_dates': True,
                'infer_datetime_format': True
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(filepath, **default_params)
            logger.info(f"Successfully loaded CSV: {filepath} - Shape: {df.shape}")
            
            return self._optimize_dtypes(df)
            
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {str(e)}")
            raise
    
    def load_excel(self, 
                   filepath: Union[str, Path], 
                   sheet_name: Optional[str] = None,
                   **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load Excel file with support for multiple sheets.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Specific sheet name or None for all sheets
            **kwargs: Additional pandas read_excel parameters
            
        Returns:
            DataFrame or dictionary of DataFrames
        """
        try:
            if sheet_name is None:
                # Load all sheets
                dfs = pd.read_excel(filepath, sheet_name=None, **kwargs)
                for name, df in dfs.items():
                    dfs[name] = self._optimize_dtypes(df)
                logger.info(f"Successfully loaded Excel: {filepath} - {len(dfs)} sheets")
                return dfs
            else:
                df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
                logger.info(f"Successfully loaded Excel sheet '{sheet_name}': {filepath} - Shape: {df.shape}")
                return self._optimize_dtypes(df)
                
        except Exception as e:
            logger.error(f"Error loading Excel {filepath}: {str(e)}")
            raise
    
    def load_json(self, 
                  filepath: Union[str, Path], 
                  orient: str = 'records') -> pd.DataFrame:
        """
        Load JSON file into DataFrame.
        
        Args:
            filepath: Path to JSON file
            orient: JSON orientation for pandas
            
        Returns:
            DataFrame with loaded data
        """
        try:
            if orient == 'records':
                df = pd.read_json(filepath, orient=orient)
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                df = pd.json_normalize(data)
            
            logger.info(f"Successfully loaded JSON: {filepath} - Shape: {df.shape}")
            return self._optimize_dtypes(df)
            
        except Exception as e:
            logger.error(f"Error loading JSON {filepath}: {str(e)}")
            raise
    
    def load_from_sql(self, 
                      query: str, 
                      connection_string: str,
                      **kwargs) -> pd.DataFrame:
        """
        Load data from SQL database.
        
        Args:
            query: SQL query string
            connection_string: Database connection string
            **kwargs: Additional parameters for pandas read_sql
            
        Returns:
            DataFrame with query results
        """
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
            logger.info(f"Successfully loaded SQL data - Shape: {df.shape}")
            return self._optimize_dtypes(df)
            
        except Exception as e:
            logger.error(f"Error loading SQL data: {str(e)}")
            raise
    
    def load_from_api(self, 
                      url: str, 
                      params: Optional[Dict] = None,
                      headers: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load data from REST API endpoint.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            DataFrame with API response data
        """
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
            
            logger.info(f"Successfully loaded API data: {url} - Shape: {df.shape}")
            return self._optimize_dtypes(df)
            
        except Exception as e:
            logger.error(f"Error loading API data from {url}: {str(e)}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized data types
        """
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if cardinality is low
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        if memory_reduction > 0:
            logger.info(f"Memory usage reduced by {memory_reduction:.1f}%")
        
        return df
    
    def load_multiple_files(self, 
                           pattern: str, 
                           file_type: str = 'csv',
                           combine: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load multiple files matching a pattern.
        
        Args:
            pattern: File pattern (glob style)
            file_type: Type of files ('csv', 'excel', 'json')
            combine: Whether to combine all files into one DataFrame
            
        Returns:
            Combined DataFrame or list of DataFrames
        """
        from glob import glob
        
        files = glob(pattern)
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        dfs = []
        load_func = getattr(self, f'load_{file_type}')
        
        for file in files:
            try:
                df = load_func(file)
                if combine:
                    df['source_file'] = Path(file).name  # Add source file column
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {str(e)}")
        
        if combine and dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(dfs)} files - Final shape: {combined_df.shape}")
            return combined_df
        
        return dfs


class DataValidator:
    """
    Data validation utilities for ensuring data quality.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 1) -> Dict[str, Any]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'summary': {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame is empty")
            return validation_results
        
        # Check minimum rows
        if len(df) < min_rows:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Missing required columns: {list(missing_cols)}")
        
        # Data quality summary
        validation_results['summary'] = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return validation_results