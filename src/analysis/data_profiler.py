"""
Data Profiling and Exploratory Data Analysis Utilities
Comprehensive analysis tools for understanding datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
from scipy import stats
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Comprehensive data profiling utility for exploratory data analysis.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataProfiler with a DataFrame.
        
        Args:
            df: DataFrame to profile
        """
        self.df = df.copy()
        self._profile_cache = {}
    
    def generate_profile_report(self, 
                               include_correlations: bool = True,
                               include_distributions: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive profile report of the dataset.
        
        Args:
            include_correlations: Whether to include correlation analysis
            include_distributions: Whether to include distribution analysis
            
        Returns:
            Dictionary containing comprehensive profile information
        """
        profile = {
            'dataset_info': self._get_dataset_info(),
            'column_profiles': self._get_column_profiles(),
            'missing_data_analysis': self._analyze_missing_data(),
            'data_quality_issues': self._identify_data_quality_issues(),
            'statistical_summary': self._get_statistical_summary()
        }
        
        if include_correlations:
            profile['correlation_analysis'] = self._analyze_correlations()
        
        if include_distributions:
            profile['distribution_analysis'] = self._analyze_distributions()
        
        return profile
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'column_count': len(self.df.columns),
            'row_count': len(self.df),
            'duplicate_rows': self.df.duplicated().sum(),
            'total_missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        }
    
    def _get_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Generate detailed profiles for each column."""
        column_profiles = {}
        
        for column in self.df.columns:
            col_data = self.df[column]
            
            profile = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / col_data.count()) * 100 if col_data.count() > 0 else 0
            }
            
            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(col_data):
                profile.update(self._analyze_numeric_column(col_data))
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                profile.update(self._analyze_datetime_column(col_data))
            else:
                profile.update(self._analyze_categorical_column(col_data))
            
            column_profiles[column] = profile
        
        return column_profiles
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'analysis_type': 'numeric', 'error': 'No non-null values'}
        
        analysis = {
            'analysis_type': 'numeric',
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'mean': float(clean_series.mean()),
            'median': float(clean_series.median()),
            'std': float(clean_series.std()),
            'variance': float(clean_series.var()),
            'skewness': float(clean_series.skew()),
            'kurtosis': float(clean_series.kurtosis()),
            'q1': float(clean_series.quantile(0.25)),
            'q3': float(clean_series.quantile(0.75)),
            'iqr': float(clean_series.quantile(0.75) - clean_series.quantile(0.25))
        }
        
        # Outlier detection using IQR method
        q1, q3 = analysis['q1'], analysis['q3']
        iqr = analysis['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
        
        analysis.update({
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(clean_series)) * 100,
            'zeros_count': (clean_series == 0).sum(),
            'negative_count': (clean_series < 0).sum(),
            'positive_count': (clean_series > 0).sum()
        })
        
        # Distribution analysis
        try:
            # Test for normality (Shapiro-Wilk test for small samples, D'Agostino-Pearson for larger)
            if len(clean_series) <= 5000:
                _, p_value = stats.shapiro(clean_series.sample(min(5000, len(clean_series))))
            else:
                _, p_value = stats.normaltest(clean_series.sample(5000))
            analysis['normality_p_value'] = float(p_value)
            analysis['is_normal'] = p_value > 0.05
        except:
            analysis['normality_p_value'] = None
            analysis['is_normal'] = None
        
        return analysis
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'analysis_type': 'datetime', 'error': 'No non-null values'}
        
        analysis = {
            'analysis_type': 'datetime',
            'min_date': clean_series.min().isoformat() if hasattr(clean_series.min(), 'isoformat') else str(clean_series.min()),
            'max_date': clean_series.max().isoformat() if hasattr(clean_series.max(), 'isoformat') else str(clean_series.max()),
            'date_range_days': (clean_series.max() - clean_series.min()).days if hasattr((clean_series.max() - clean_series.min()), 'days') else None
        }
        
        # Frequency analysis
        if hasattr(clean_series.dt, 'year'):
            analysis.update({
                'year_range': f"{clean_series.dt.year.min()}-{clean_series.dt.year.max()}",
                'unique_years': clean_series.dt.year.nunique(),
                'unique_months': clean_series.dt.month.nunique(),
                'unique_days': clean_series.dt.day.nunique()
            })
        
        return analysis
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical/text column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'analysis_type': 'categorical', 'error': 'No non-null values'}
        
        value_counts = clean_series.value_counts()
        
        analysis = {
            'analysis_type': 'categorical',
            'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'least_frequent_value': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
            'top_5_values': value_counts.head().to_dict()
        }
        
        # Text analysis for string columns
        if series.dtype == 'object':
            str_series = clean_series.astype(str)
            analysis.update({
                'avg_length': float(str_series.str.len().mean()),
                'min_length': int(str_series.str.len().min()),
                'max_length': int(str_series.str.len().max()),
                'empty_strings': (str_series == '').sum()
            })
        
        return analysis
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        analysis = {
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentages': missing_percentage[missing_percentage > 0].to_dict(),
            'rows_with_any_missing': self.df.isnull().any(axis=1).sum(),
            'rows_with_all_missing': self.df.isnull().all(axis=1).sum(),
            'complete_rows': len(self.df) - self.df.isnull().any(axis=1).sum()
        }
        
        # Missing data pattern analysis
        if analysis['columns_with_missing']:
            missing_patterns = self.df.isnull().value_counts()
            analysis['missing_patterns'] = {str(k): v for k, v in missing_patterns.head(10).items()}
        
        return analysis
    
    def _identify_data_quality_issues(self) -> Dict[str, List[str]]:
        """Identify potential data quality issues."""
        issues = {
            'high_cardinality': [],
            'low_variance': [],
            'potential_duplicates': [],
            'suspicious_values': [],
            'data_type_issues': []
        }
        
        for column in self.df.columns:
            col_data = self.df[column]
            
            # High cardinality check (for non-numeric columns)
            if not pd.api.types.is_numeric_dtype(col_data):
                if col_data.nunique() / len(col_data) > 0.95:
                    issues['high_cardinality'].append(column)
            
            # Low variance check (for numeric columns)
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.var() < 1e-10:
                    issues['low_variance'].append(column)
                
                # Check for suspicious values (e.g., all zeros, all same value)
                if col_data.nunique() == 1:
                    issues['suspicious_values'].append(f"{column}: All values are identical")
            
            # Check for potential data type issues
            if col_data.dtype == 'object':
                # Check if column should be numeric
                try:
                    pd.to_numeric(col_data.dropna(), errors='raise')
                    issues['data_type_issues'].append(f"{column}: Could be converted to numeric")
                except:
                    pass
                
                # Check if column should be datetime
                if col_data.dropna().str.match(r'\d{4}-\d{2}-\d{2}').any():
                    issues['data_type_issues'].append(f"{column}: Could be converted to datetime")
        
        # Check for potential duplicate columns
        for i, col1 in enumerate(self.df.columns):
            for col2 in self.df.columns[i+1:]:
                if self.df[col1].equals(self.df[col2]):
                    issues['potential_duplicates'].append(f"{col1} and {col2}")
        
        return issues
    
    def _get_statistical_summary(self) -> pd.DataFrame:
        """Get comprehensive statistical summary."""
        return self.df.describe(include='all')
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numeric variables."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns found'}
        
        correlation_matrix = numeric_df.corr()
        
        # Find high correlations (excluding self-correlations)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'mean_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
        }
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions of numeric variables."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns found'}
        
        distributions = {}
        
        for column in numeric_df.columns:
            col_data = numeric_df[column].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Test for various distributions
            distributions[column] = {
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'is_symmetric': abs(col_data.skew()) < 0.5,
                'distribution_shape': self._classify_distribution(col_data)
            }
        
        return distributions
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the shape of a distribution."""
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'Normal-like'
        elif skewness > 1:
            return 'Right-skewed'
        elif skewness < -1:
            return 'Left-skewed'
        elif kurtosis > 3:
            return 'Heavy-tailed'
        elif kurtosis < -1:
            return 'Light-tailed'
        else:
            return 'Other'
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        profile = self.generate_profile_report()
        
        report = f"""
DATA PROFILE SUMMARY REPORT
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
================
• Shape: {profile['dataset_info']['shape'][0]:,} rows × {profile['dataset_info']['shape'][1]} columns
• Memory Usage: {profile['dataset_info']['memory_usage_mb']:.2f} MB
• Missing Values: {profile['dataset_info']['total_missing_values']:,} ({profile['dataset_info']['missing_percentage']:.1f}%)
• Duplicate Rows: {profile['dataset_info']['duplicate_rows']:,}

COLUMN SUMMARY
==============
"""
        
        for col_name, col_profile in profile['column_profiles'].items():
            report += f"\n{col_name} ({col_profile['dtype']}):\n"
            report += f"  • Missing: {col_profile['null_percentage']:.1f}%\n"
            report += f"  • Unique: {col_profile['unique_count']:,} ({col_profile['unique_percentage']:.1f}%)\n"
            
            if col_profile['analysis_type'] == 'numeric':
                report += f"  • Range: {col_profile.get('min', 'N/A')} to {col_profile.get('max', 'N/A')}\n"
                report += f"  • Mean: {col_profile.get('mean', 'N/A'):.2f}, Std: {col_profile.get('std', 'N/A'):.2f}\n"
                if col_profile.get('outlier_count', 0) > 0:
                    report += f"  • Outliers: {col_profile['outlier_count']} ({col_profile['outlier_percentage']:.1f}%)\n"
        
        # Data Quality Issues
        issues = profile['data_quality_issues']
        if any(issues.values()):
            report += "\nDATA QUALITY ISSUES\n"
            report += "===================\n"
            for issue_type, issue_list in issues.items():
                if issue_list:
                    report += f"• {issue_type.replace('_', ' ').title()}: {len(issue_list)} columns\n"
        
        return report


class DataComparer:
    """
    Utility for comparing two datasets and identifying differences.
    """
    
    @staticmethod
    def compare_dataframes(df1: pd.DataFrame, 
                          df2: pd.DataFrame, 
                          name1: str = "Dataset 1", 
                          name2: str = "Dataset 2") -> Dict[str, Any]:
        """
        Compare two DataFrames and identify differences.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            name1: Name for first dataset
            name2: Name for second dataset
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'basic_comparison': {
                f'{name1}_shape': df1.shape,
                f'{name2}_shape': df2.shape,
                'shape_match': df1.shape == df2.shape
            },
            'column_comparison': {
                f'{name1}_columns': list(df1.columns),
                f'{name2}_columns': list(df2.columns),
                'common_columns': list(set(df1.columns) & set(df2.columns)),
                'unique_to_df1': list(set(df1.columns) - set(df2.columns)),
                'unique_to_df2': list(set(df2.columns) - set(df1.columns))
            },
            'data_type_comparison': {},
            'statistical_comparison': {}
        }
        
        # Compare data types for common columns
        common_cols = comparison['column_comparison']['common_columns']
        for col in common_cols:
            comparison['data_type_comparison'][col] = {
                f'{name1}_dtype': str(df1[col].dtype),
                f'{name2}_dtype': str(df2[col].dtype),
                'dtypes_match': str(df1[col].dtype) == str(df2[col].dtype)
            }
        
        # Statistical comparison for numeric columns
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                comparison['statistical_comparison'][col] = {
                    f'{name1}_mean': float(df1[col].mean()) if not df1[col].isnull().all() else None,
                    f'{name2}_mean': float(df2[col].mean()) if not df2[col].isnull().all() else None,
                    f'{name1}_std': float(df1[col].std()) if not df1[col].isnull().all() else None,
                    f'{name2}_std': float(df2[col].std()) if not df2[col].isnull().all() else None
                }
        
        return comparison