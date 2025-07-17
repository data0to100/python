"""
Data Quality Monitor
Comprehensive data quality monitoring and alerting system.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetric:
    """Data class for quality metrics."""
    name: str
    value: float
    threshold: Optional[float] = None
    status: str = "OK"  # OK, WARNING, CRITICAL
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class QualityAlert:
    """Data class for quality alerts."""
    metric_name: str
    alert_type: str  # DRIFT, THRESHOLD, ANOMALY
    severity: str    # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    current_value: float
    expected_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system that tracks
    data quality metrics over time and detects issues.
    """
    
    def __init__(self, 
                 baseline_path: Optional[str] = None,
                 history_path: Optional[str] = None,
                 drift_threshold: float = 0.1,
                 anomaly_threshold: float = 3.0):
        """
        Initialize the data quality monitor.
        
        Args:
            baseline_path: Path to save/load baseline metrics
            history_path: Path to save quality history
            drift_threshold: Threshold for detecting drift (0-1)
            anomaly_threshold: Standard deviations for anomaly detection
        """
        self.baseline_path = baseline_path or "data_quality_baseline.json"
        self.history_path = history_path or "data_quality_history.json"
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold
        
        self.baseline_metrics = {}
        self.quality_history = []
        self.alerts = []
        
        # Load existing baseline and history
        self._load_baseline()
        self._load_history()
    
    def create_baseline(self, df: pd.DataFrame, save: bool = True) -> Dict[str, QualityMetric]:
        """
        Create baseline quality metrics from a reference dataset.
        
        Args:
            df: Reference dataframe
            save: Whether to save baseline to file
            
        Returns:
            Dictionary of baseline metrics
        """
        logger.info("Creating data quality baseline...")
        
        metrics = {}
        
        # Basic dataset metrics
        metrics['row_count'] = QualityMetric('row_count', len(df))
        metrics['column_count'] = QualityMetric('column_count', len(df.columns))
        metrics['memory_usage'] = QualityMetric('memory_usage', df.memory_usage(deep=True).sum())
        
        # Missing data metrics
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        metrics['missing_ratio'] = QualityMetric('missing_ratio', total_missing / total_cells)
        
        # Column-specific metrics
        for col in df.columns:
            col_prefix = f"col_{col}"
            
            # Missing values
            missing_pct = df[col].isnull().sum() / len(df)
            metrics[f"{col_prefix}_missing_pct"] = QualityMetric(f"{col_prefix}_missing_pct", missing_pct)
            
            # Unique values
            unique_pct = df[col].nunique() / len(df)
            metrics[f"{col_prefix}_unique_pct"] = QualityMetric(f"{col_prefix}_unique_pct", unique_pct)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column metrics
                values = df[col].dropna()
                if len(values) > 0:
                    metrics[f"{col_prefix}_mean"] = QualityMetric(f"{col_prefix}_mean", values.mean())
                    metrics[f"{col_prefix}_std"] = QualityMetric(f"{col_prefix}_std", values.std())
                    metrics[f"{col_prefix}_min"] = QualityMetric(f"{col_prefix}_min", values.min())
                    metrics[f"{col_prefix}_max"] = QualityMetric(f"{col_prefix}_max", values.max())
                    metrics[f"{col_prefix}_skewness"] = QualityMetric(f"{col_prefix}_skewness", stats.skew(values))
                    
                    # Outlier percentage (using IQR method)
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
                    outlier_pct = len(outliers) / len(values)
                    metrics[f"{col_prefix}_outlier_pct"] = QualityMetric(f"{col_prefix}_outlier_pct", outlier_pct)
            
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                # Categorical column metrics
                values = df[col].dropna()
                if len(values) > 0:
                    value_counts = values.value_counts()
                    # Most frequent value percentage
                    most_frequent_pct = value_counts.iloc[0] / len(values) if len(value_counts) > 0 else 0
                    metrics[f"{col_prefix}_most_frequent_pct"] = QualityMetric(f"{col_prefix}_most_frequent_pct", most_frequent_pct)
                    
                    # Entropy (measure of diversity)
                    probabilities = value_counts / len(values)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    metrics[f"{col_prefix}_entropy"] = QualityMetric(f"{col_prefix}_entropy", entropy)
        
        # Data type consistency
        for col in df.columns:
            dtype_name = str(df[col].dtype)
            metrics[f"col_{col}_dtype"] = QualityMetric(f"col_{col}_dtype", hash(dtype_name))
        
        self.baseline_metrics = metrics
        
        if save:
            self._save_baseline()
        
        logger.info(f"Baseline created with {len(metrics)} metrics")
        return metrics
    
    def check_quality(self, df: pd.DataFrame, save_history: bool = True) -> Tuple[Dict[str, QualityMetric], List[QualityAlert]]:
        """
        Check data quality against baseline and detect issues.
        
        Args:
            df: DataFrame to check
            save_history: Whether to save results to history
            
        Returns:
            Tuple of (current metrics, alerts)
        """
        logger.info("Checking data quality...")
        
        # Calculate current metrics (same logic as baseline)
        current_metrics = self._calculate_current_metrics(df)
        
        # Compare with baseline and generate alerts
        alerts = self._compare_with_baseline(current_metrics)
        
        # Detect anomalies based on history
        history_alerts = self._detect_anomalies(current_metrics)
        alerts.extend(history_alerts)
        
        # Store results
        self.alerts = alerts
        
        if save_history:
            self._save_to_history(current_metrics, alerts)
        
        logger.info(f"Quality check completed. Found {len(alerts)} alerts")
        return current_metrics, alerts