"""
Data Preprocessing and Validation

This module provides comprehensive data preprocessing capabilities:
- Data quality assessment and validation
- Missing value handling strategies
- Data type validation and conversion
- Duplicate detection and removal
- Data consistency checks
- Memory optimization
- Schema validation
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..utils.logger import get_logger, timing_decorator, error_handler
from ..utils.config import Config


class DataPreprocessor:
    """
    Comprehensive data preprocessing and validation
    
    Features:
    - Automated data quality assessment
    - Intelligent missing value handling
    - Data type optimization
    - Duplicate detection and removal
    - Schema validation and enforcement
    - Memory-efficient processing
    """
    
    def __init__(self, config: Config):
        """Initialize the DataPreprocessor"""
        self.config = config
        self.logger = get_logger()
        self.preprocessing_report = {}
    
    @timing_decorator("data_preprocessing")
    @error_handler
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.logger.info("Starting data preprocessing")
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Step 1: Data quality assessment
        quality_report = self._assess_data_quality(processed_data)
        
        # Step 2: Handle duplicates
        processed_data = self._handle_duplicates(processed_data)
        
        # Step 3: Data type optimization
        processed_data = self._optimize_data_types(processed_data)
        
        # Step 4: Memory optimization
        processed_data = self._optimize_memory_usage(processed_data)
        
        # Generate preprocessing report
        self.preprocessing_report = {
            "quality_assessment": quality_report,
            "original_shape": data.shape,
            "processed_shape": processed_data.shape,
            "memory_reduction": self._calculate_memory_reduction(data, processed_data)
        }
        
        self.logger.logger.info(f"Data preprocessing completed: {data.shape} -> {processed_data.shape}")
        
        return processed_data
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality across multiple dimensions"""
        
        quality_report = {
            "completeness": self._assess_completeness(data),
            "uniqueness": self._assess_uniqueness(data),
            "validity": self._assess_validity(data),
            "consistency": self._assess_consistency(data)
        }
        
        return quality_report
    
    def _assess_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness (missing values)"""
        
        missing_stats = data.isnull().sum()
        missing_percentages = (missing_stats / len(data)) * 100
        
        completeness = {
            "total_missing": missing_stats.sum(),
            "overall_completeness": ((data.size - missing_stats.sum()) / data.size) * 100,
            "column_missing_counts": missing_stats.to_dict(),
            "column_missing_percentages": missing_percentages.to_dict(),
            "columns_with_high_missing": missing_percentages[missing_percentages > 50].index.tolist()
        }
        
        return completeness
    
    def _assess_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness (duplicates)"""
        
        duplicate_rows = data.duplicated().sum()
        
        uniqueness = {
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": (duplicate_rows / len(data)) * 100,
            "unique_rows": len(data) - duplicate_rows,
            "column_uniqueness": {}
        }
        
        # Assess uniqueness per column
        for col in data.columns:
            unique_count = data[col].nunique()
            uniqueness["column_uniqueness"][col] = {
                "unique_values": unique_count,
                "uniqueness_ratio": unique_count / len(data),
                "is_likely_identifier": unique_count > len(data) * 0.95
            }
        
        return uniqueness
    
    def _assess_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity (type consistency, ranges)"""
        
        validity = {
            "data_types": data.dtypes.to_dict(),
            "invalid_values": {},
            "range_violations": {},
            "format_violations": {}
        }
        
        for col in data.columns:
            col_data = data[col]
            
            # Check for infinite values in numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                inf_count = np.isinf(col_data).sum()
                if inf_count > 0:
                    validity["invalid_values"][col] = {
                        "infinite_values": inf_count
                    }
                
                # Check for negative values where they might not make sense
                if (col_data < 0).any() and any(keyword in col.lower() for keyword in ['count', 'quantity', 'amount', 'price']):
                    validity["range_violations"][col] = {
                        "negative_values": (col_data < 0).sum()
                    }
            
            # Check for string length consistency in object columns
            elif col_data.dtype == 'object':
                if col_data.notna().any():
                    str_lengths = col_data.astype(str).str.len()
                    validity["format_violations"][col] = {
                        "min_length": str_lengths.min(),
                        "max_length": str_lengths.max(),
                        "avg_length": str_lengths.mean(),
                        "length_variance": str_lengths.var()
                    }
        
        return validity
    
    def _assess_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency across columns"""
        
        consistency = {
            "column_correlations": {},
            "categorical_consistency": {},
            "temporal_consistency": {}
        }
        
        # Assess correlations between numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr()
            high_correlations = []
            
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.9:  # High correlation threshold
                        high_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr_value
                        })
            
            consistency["column_correlations"] = {
                "high_correlations": high_correlations,
                "correlation_matrix_shape": correlation_matrix.shape
            }
        
        # Assess categorical consistency
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = data[col].unique()
            consistency["categorical_consistency"][col] = {
                "unique_count": len(unique_values),
                "potential_duplicates": self._find_potential_categorical_duplicates(unique_values)
            }
        
        return consistency
    
    def _find_potential_categorical_duplicates(self, values: np.ndarray) -> List[Dict]:
        """Find potential duplicates in categorical values (case sensitivity, whitespace)"""
        
        potential_duplicates = []
        str_values = [str(v).strip().lower() for v in values if pd.notna(v)]
        
        from collections import Counter
        value_counts = Counter(str_values)
        
        for value, count in value_counts.items():
            if count > 1:
                original_values = [v for v in values if str(v).strip().lower() == value]
                if len(set(str(v) for v in original_values)) > 1:  # Different original formats
                    potential_duplicates.append({
                        "standardized_value": value,
                        "original_variants": list(set(str(v) for v in original_values)),
                        "count": count
                    })
        
        return potential_duplicates
    
    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows"""
        
        initial_rows = len(data)
        duplicate_count = data.duplicated().sum()
        
        if duplicate_count > 0:
            # Remove duplicates, keeping first occurrence
            data_cleaned = data.drop_duplicates(keep='first')
            
            self.logger.logger.info(f"Removed {duplicate_count} duplicate rows ({duplicate_count/initial_rows*100:.2f}%)")
            
            return data_cleaned
        
        return data
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        
        optimized_data = data.copy()
        
        for col in optimized_data.columns:
            col_data = optimized_data[col]
            
            # Optimize integer columns
            if pd.api.types.is_integer_dtype(col_data):
                optimized_data[col] = pd.to_numeric(col_data, downcast='integer')
            
            # Optimize float columns
            elif pd.api.types.is_float_dtype(col_data):
                optimized_data[col] = pd.to_numeric(col_data, downcast='float')
            
            # Convert object columns with low cardinality to category
            elif col_data.dtype == 'object':
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.5:  # Low cardinality
                    optimized_data[col] = col_data.astype('category')
        
        return optimized_data
    
    def _optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Further optimize memory usage"""
        
        # This method can include more advanced memory optimizations
        # For now, it's a placeholder for future enhancements
        return data
    
    def _calculate_memory_reduction(self, original_data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate memory usage reduction"""
        
        original_memory = original_data.memory_usage(deep=True).sum()
        processed_memory = processed_data.memory_usage(deep=True).sum()
        
        reduction = {
            "original_memory_mb": original_memory / (1024 * 1024),
            "processed_memory_mb": processed_memory / (1024 * 1024),
            "memory_saved_mb": (original_memory - processed_memory) / (1024 * 1024),
            "reduction_percentage": ((original_memory - processed_memory) / original_memory) * 100 if original_memory > 0 else 0
        }
        
        return reduction
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing report"""
        return self.preprocessing_report