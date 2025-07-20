"""
Advanced Outlier Detection for Large-Scale Data

This module provides scalable outlier detection using multiple algorithms:
- Isolation Forest for anomaly detection in high-dimensional data
- Statistical methods (Z-score, Modified Z-score, IQR)
- Local Outlier Factor for density-based detection
- One-Class SVM for complex decision boundaries
- Ensemble methods combining multiple detectors
- Memory-efficient processing for large datasets
- Automatic threshold optimization and validation
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# Outlier detection algorithms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Statistical analysis
from scipy import stats
from scipy.stats import chi2, zscore

# Big data support
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.combination import majority_vote, average, maximization
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

from ..utils.logger import get_logger, timing_decorator, error_handler
from ..utils.config import OutlierDetectionConfig


class OutlierDetector:
    """
    Enterprise-grade outlier detection with multiple algorithms and ensemble methods
    
    Features:
    - Multiple detection algorithms with automatic parameter tuning
    - Scalable processing for large datasets using chunking
    - Statistical validation of outlier detection results
    - Ensemble methods for robust detection
    - Memory-efficient processing with Dask support
    - Comprehensive outlier analysis and reporting
    - Automated threshold optimization
    """
    
    def __init__(self, config: OutlierDetectionConfig):
        """
        Initialize the OutlierDetector
        
        Args:
            config: OutlierDetectionConfig object with detection parameters
        """
        self.config = config
        self.logger = get_logger()
        
        # Detection results storage
        self.detection_results = {}
        self.outlier_indices = set()
        self.outlier_scores = {}
        self.method_performance = {}
        
        # Scalers for preprocessing
        self.scaler = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate outlier detection configuration"""
        if not self.config.methods:
            raise ValueError("At least one outlier detection method must be specified")
        
        # Warn about performance implications
        if len(self.config.methods) > 3:
            self.logger.logger.warning(
                "Using more than 3 outlier detection methods may impact performance on large datasets"
            )
    
    @timing_decorator("outlier_detection")
    @error_handler
    def detect_outliers(self, 
                       data: Union[pd.DataFrame, dd.DataFrame],
                       target_column: Optional[str] = None,
                       sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect outliers using configured methods
        
        Args:
            data: Input DataFrame
            target_column: Target column for supervised outlier detection
            sample_size: Sample size for large datasets (None = full data)
        
        Returns:
            Dictionary containing outlier detection results
        """
        self.logger.logger.info(f"Starting outlier detection with methods: {self.config.methods}")
        
        # Handle large datasets with sampling
        if sample_size and len(data) > sample_size:
            self.logger.logger.info(f"Sampling {sample_size} rows from {len(data)} for outlier detection")
            if hasattr(data, 'sample'):
                working_data = data.sample(n=sample_size, random_state=42)
            else:
                working_data = data.head(sample_size)
        else:
            working_data = data
        
        # Prepare numerical data for outlier detection
        numerical_data = self._prepare_numerical_data(working_data)
        
        if numerical_data.empty:
            self.logger.logger.warning("No numerical columns found for outlier detection")
            return {"outliers": [], "scores": {}, "summary": {"total_outliers": 0}}
        
        # Apply each detection method
        method_results = {}
        
        for method in self.config.methods:
            try:
                self.logger.logger.info(f"Running {method} outlier detection")
                method_results[method] = self._apply_detection_method(
                    numerical_data, method, target_column
                )
            except Exception as e:
                self.logger.logger.error(f"Failed to run {method}: {e}")
                continue
        
        # Combine results from multiple methods
        combined_results = self._combine_detection_results(method_results, numerical_data)
        
        # Generate comprehensive analysis
        analysis = self._analyze_outliers(combined_results, working_data)
        
        # Store results
        self.detection_results = {
            "method_results": method_results,
            "combined_results": combined_results,
            "analysis": analysis,
            "data_shape": working_data.shape,
            "methods_used": list(method_results.keys())
        }
        
        self.logger.logger.info(
            f"Outlier detection completed. Found {len(combined_results['outlier_indices'])} outliers "
            f"({len(combined_results['outlier_indices'])/len(working_data)*100:.2f}%)"
        )
        
        return self.detection_results
    
    def _prepare_numerical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare numerical data for outlier detection"""
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            return pd.DataFrame()
        
        numerical_data = data[numerical_cols].copy()
        
        # Handle missing values
        if numerical_data.isnull().any().any():
            self.logger.logger.info("Handling missing values in numerical data")
            # Use median imputation for robustness
            numerical_data = numerical_data.fillna(numerical_data.median())
        
        # Remove infinite values
        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
        numerical_data = numerical_data.dropna()
        
        # Scale data for algorithms that require it
        if len(numerical_data) > 0:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            scaled_data = self.scaler.fit_transform(numerical_data)
            numerical_data_scaled = pd.DataFrame(
                scaled_data, 
                columns=numerical_data.columns, 
                index=numerical_data.index
            )
            return numerical_data_scaled
        
        return numerical_data
    
    def _apply_detection_method(self, 
                               data: pd.DataFrame, 
                               method: str,
                               target_column: Optional[str] = None) -> Dict[str, Any]:
        """Apply a specific outlier detection method"""
        
        if method == "isolation_forest":
            return self._isolation_forest_detection(data)
        elif method == "z_score":
            return self._z_score_detection(data)
        elif method == "modified_z_score":
            return self._modified_z_score_detection(data)
        elif method == "iqr":
            return self._iqr_detection(data)
        elif method == "local_outlier_factor":
            return self._lof_detection(data)
        elif method == "one_class_svm":
            return self._one_class_svm_detection(data)
        elif method == "statistical_ensemble":
            return self._statistical_ensemble_detection(data)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _isolation_forest_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Isolation Forest outlier detection with parameter optimization"""
        
        # Parameter grid for optimization
        param_grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'max_samples': ['auto', 0.5, 0.8],
            'max_features': [1.0, 0.8, 0.5]
        }
        
        best_contamination = self.config.isolation_forest_contamination
        best_score = -np.inf
        
        # Try different contamination rates if dataset is small enough
        if len(data) < 10000:
            for contamination in param_grid['contamination']:
                try:
                    model = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_jobs=-1
                    )
                    outlier_labels = model.fit_predict(data)
                    scores = model.decision_function(data)
                    
                    # Use silhouette-like score for evaluation
                    score = self._evaluate_outlier_detection(data, outlier_labels)
                    if score > best_score:
                        best_score = score
                        best_contamination = contamination
                        
                except Exception as e:
                    self.logger.logger.warning(f"Failed with contamination {contamination}: {e}")
                    continue
        
        # Final model with best parameters
        model = IsolationForest(
            contamination=best_contamination,
            random_state=42,
            n_jobs=-1,
            max_samples='auto'
        )
        
        outlier_labels = model.fit_predict(data)
        outlier_scores = model.decision_function(data)
        
        # Convert to boolean mask
        outlier_mask = outlier_labels == -1
        outlier_indices = data.index[outlier_mask].tolist()
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_scores": dict(zip(data.index, outlier_scores)),
            "model": model,
            "parameters": {"contamination": best_contamination},
            "method": "isolation_forest"
        }
    
    def _z_score_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Z-score based outlier detection"""
        
        # Calculate Z-scores for each column
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
        
        # Identify outliers (any column with |z-score| > threshold)
        threshold = self.config.z_score_threshold
        outlier_mask = (z_scores > threshold).any(axis=1)
        outlier_indices = data.index[outlier_mask].tolist()
        
        # Calculate composite outlier score (max z-score across columns)
        composite_scores = np.max(z_scores, axis=1)
        outlier_scores = dict(zip(data.index, composite_scores))
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_scores": outlier_scores,
            "z_scores": z_scores,
            "parameters": {"threshold": threshold},
            "method": "z_score"
        }
    
    def _modified_z_score_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Modified Z-score using median absolute deviation (more robust)"""
        
        def modified_z_score(x):
            median = np.median(x)
            mad = np.median(np.abs(x - median))
            if mad == 0:
                return np.zeros_like(x)
            return 0.6745 * (x - median) / mad
        
        # Calculate modified Z-scores
        modified_z_scores = data.apply(modified_z_score, axis=0)
        
        # Identify outliers
        threshold = self.config.z_score_threshold
        outlier_mask = (np.abs(modified_z_scores) > threshold).any(axis=1)
        outlier_indices = data.index[outlier_mask].tolist()
        
        # Composite score
        composite_scores = np.max(np.abs(modified_z_scores), axis=1)
        outlier_scores = dict(zip(data.index, composite_scores))
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_scores": outlier_scores,
            "modified_z_scores": modified_z_scores,
            "parameters": {"threshold": threshold},
            "method": "modified_z_score"
        }
    
    def _iqr_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Interquartile Range (IQR) based outlier detection"""
        
        outlier_indices_all = set()
        column_outliers = {}
        
        multiplier = self.config.iqr_multiplier
        
        for column in data.columns:
            col_data = data[column].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Identify outliers
            column_outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            column_outlier_indices = col_data.index[column_outlier_mask].tolist()
            
            column_outliers[column] = {
                "indices": column_outlier_indices,
                "bounds": (lower_bound, upper_bound),
                "count": len(column_outlier_indices)
            }
            
            outlier_indices_all.update(column_outlier_indices)
        
        # Calculate outlier scores based on distance from IQR bounds
        outlier_scores = {}
        for idx in data.index:
            max_distance = 0
            for column in data.columns:
                value = data.loc[idx, column]
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    if value < lower_bound:
                        distance = (lower_bound - value) / IQR
                    elif value > upper_bound:
                        distance = (value - upper_bound) / IQR
                    else:
                        distance = 0
                    
                    max_distance = max(max_distance, distance)
            
            outlier_scores[idx] = max_distance
        
        return {
            "outlier_indices": list(outlier_indices_all),
            "outlier_scores": outlier_scores,
            "column_details": column_outliers,
            "parameters": {"multiplier": multiplier},
            "method": "iqr"
        }
    
    def _lof_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Local Outlier Factor detection"""
        
        # Limit data size for LOF to avoid memory issues
        if len(data) > 5000:
            self.logger.logger.warning("LOF detection: sampling 5000 points for performance")
            data_sample = data.sample(n=5000, random_state=42)
        else:
            data_sample = data
        
        # Optimize n_neighbors based on data size
        n_neighbors = min(20, max(5, len(data_sample) // 10))
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.config.isolation_forest_contamination,
            n_jobs=-1
        )
        
        outlier_labels = model.fit_predict(data_sample)
        outlier_scores_raw = model.negative_outlier_factor_
        
        # Convert negative LOF scores to positive outlier scores
        outlier_scores = -outlier_scores_raw
        
        outlier_mask = outlier_labels == -1
        outlier_indices = data_sample.index[outlier_mask].tolist()
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_scores": dict(zip(data_sample.index, outlier_scores)),
            "model": model,
            "parameters": {"n_neighbors": n_neighbors},
            "method": "local_outlier_factor"
        }
    
    def _one_class_svm_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """One-Class SVM outlier detection"""
        
        # Limit data size for SVM performance
        if len(data) > 3000:
            self.logger.logger.warning("One-Class SVM: sampling 3000 points for performance")
            data_sample = data.sample(n=3000, random_state=42)
        else:
            data_sample = data
        
        model = OneClassSVM(
            nu=self.config.isolation_forest_contamination,
            kernel='rbf',
            gamma='scale'
        )
        
        outlier_labels = model.fit_predict(data_sample)
        outlier_scores = model.decision_function(data_sample)
        
        outlier_mask = outlier_labels == -1
        outlier_indices = data_sample.index[outlier_mask].tolist()
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_scores": dict(zip(data_sample.index, outlier_scores)),
            "model": model,
            "parameters": {"nu": self.config.isolation_forest_contamination},
            "method": "one_class_svm"
        }
    
    def _statistical_ensemble_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Ensemble of statistical methods"""
        
        # Run multiple statistical methods
        z_result = self._z_score_detection(data)
        modified_z_result = self._modified_z_score_detection(data)
        iqr_result = self._iqr_detection(data)
        
        # Combine results using voting
        all_indices = set()
        all_indices.update(z_result["outlier_indices"])
        all_indices.update(modified_z_result["outlier_indices"])
        all_indices.update(iqr_result["outlier_indices"])
        
        # Voting: require at least 2 out of 3 methods to agree
        final_outliers = []
        for idx in all_indices:
            vote_count = 0
            if idx in z_result["outlier_indices"]:
                vote_count += 1
            if idx in modified_z_result["outlier_indices"]:
                vote_count += 1
            if idx in iqr_result["outlier_indices"]:
                vote_count += 1
            
            if vote_count >= 2:
                final_outliers.append(idx)
        
        # Combine scores
        combined_scores = {}
        for idx in data.index:
            scores = []
            if idx in z_result["outlier_scores"]:
                scores.append(z_result["outlier_scores"][idx])
            if idx in modified_z_result["outlier_scores"]:
                scores.append(modified_z_result["outlier_scores"][idx])
            if idx in iqr_result["outlier_scores"]:
                scores.append(iqr_result["outlier_scores"][idx])
            
            combined_scores[idx] = np.mean(scores) if scores else 0.0
        
        return {
            "outlier_indices": final_outliers,
            "outlier_scores": combined_scores,
            "component_results": {
                "z_score": z_result,
                "modified_z_score": modified_z_result,
                "iqr": iqr_result
            },
            "method": "statistical_ensemble"
        }
    
    def _combine_detection_results(self, method_results: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Combine results from multiple detection methods using ensemble voting"""
        
        if not method_results:
            return {"outlier_indices": [], "outlier_scores": {}, "method_votes": {}}
        
        # Collect all potential outlier indices
        all_outlier_indices = set()
        for result in method_results.values():
            all_outlier_indices.update(result["outlier_indices"])
        
        # Voting mechanism: count how many methods detected each point as outlier
        method_votes = {}
        final_outlier_scores = {}
        
        for idx in data.index:
            votes = 0
            scores = []
            
            for method, result in method_results.items():
                if idx in result["outlier_indices"]:
                    votes += 1
                
                if idx in result["outlier_scores"]:
                    scores.append(result["outlier_scores"][idx])
            
            method_votes[idx] = votes
            final_outlier_scores[idx] = np.mean(scores) if scores else 0.0
        
        # Determine final outliers based on voting threshold
        min_votes = max(1, len(method_results) // 2)  # Majority voting
        final_outliers = [idx for idx, votes in method_votes.items() if votes >= min_votes]
        
        return {
            "outlier_indices": final_outliers,
            "outlier_scores": final_outlier_scores,
            "method_votes": method_votes,
            "voting_threshold": min_votes,
            "total_methods": len(method_results)
        }
    
    def _analyze_outliers(self, combined_results: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive outlier analysis"""
        
        outlier_indices = combined_results["outlier_indices"]
        
        if not outlier_indices:
            return {"summary": "No outliers detected"}
        
        analysis = {
            "total_outliers": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(data)) * 100,
            "outlier_statistics": {},
            "column_impact": {},
            "severity_distribution": {}
        }
        
        # Analyze outlier characteristics
        if outlier_indices:
            outlier_data = data.loc[outlier_indices]
            normal_data = data.drop(outlier_indices)
            
            # Statistical comparison
            for column in data.select_dtypes(include=[np.number]).columns:
                if column in outlier_data.columns:
                    analysis["column_impact"][column] = {
                        "outlier_mean": outlier_data[column].mean(),
                        "normal_mean": normal_data[column].mean(),
                        "outlier_std": outlier_data[column].std(),
                        "normal_std": normal_data[column].std(),
                        "difference_ratio": abs(outlier_data[column].mean() - normal_data[column].mean()) / normal_data[column].std() if normal_data[column].std() > 0 else 0
                    }
        
        # Severity distribution
        outlier_scores = combined_results["outlier_scores"]
        if outlier_scores:
            outlier_score_values = [outlier_scores[idx] for idx in outlier_indices]
            analysis["severity_distribution"] = {
                "min_score": min(outlier_score_values),
                "max_score": max(outlier_score_values),
                "mean_score": np.mean(outlier_score_values),
                "std_score": np.std(outlier_score_values)
            }
        
        return analysis
    
    def _evaluate_outlier_detection(self, data: pd.DataFrame, outlier_labels: np.ndarray) -> float:
        """Evaluate outlier detection quality using internal metrics"""
        
        try:
            # Calculate silhouette-like score for outlier detection
            from sklearn.metrics import silhouette_score
            
            # Convert outlier labels to binary (1 for outlier, 0 for normal)
            binary_labels = (outlier_labels == -1).astype(int)
            
            if len(np.unique(binary_labels)) > 1:
                score = silhouette_score(data, binary_labels)
                return score
            else:
                return 0.0
                
        except Exception:
            # Fallback: use proportion of outliers as simple metric
            outlier_proportion = np.sum(outlier_labels == -1) / len(outlier_labels)
            # Prefer moderate outlier proportions (5-15%)
            optimal_proportion = 0.1
            score = 1.0 - abs(outlier_proportion - optimal_proportion) / optimal_proportion
            return max(0.0, score)
    
    def remove_outliers(self, 
                       data: Union[pd.DataFrame, dd.DataFrame],
                       outlier_indices: Optional[List] = None) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Remove outliers from the dataset
        
        Args:
            data: Input DataFrame
            outlier_indices: Specific outlier indices to remove (optional)
        
        Returns:
            DataFrame with outliers removed
        """
        
        if outlier_indices is None:
            if not self.detection_results:
                raise ValueError("No outlier detection results available. Run detect_outliers() first.")
            outlier_indices = self.detection_results["combined_results"]["outlier_indices"]
        
        if not outlier_indices:
            self.logger.logger.info("No outliers to remove")
            return data
        
        # Remove outliers
        if hasattr(data, 'drop'):  # Pandas DataFrame
            cleaned_data = data.drop(outlier_indices)
        else:  # Dask DataFrame
            mask = ~data.index.isin(outlier_indices)
            cleaned_data = data.loc[mask]
        
        removed_count = len(outlier_indices)
        remaining_count = len(cleaned_data)
        
        self.logger.logger.info(
            f"Removed {removed_count} outliers. "
            f"Dataset size: {len(data)} -> {remaining_count} "
            f"({removed_count/len(data)*100:.2f}% removed)"
        )
        
        return cleaned_data
    
    def get_outlier_report(self) -> Dict[str, Any]:
        """Generate comprehensive outlier detection report"""
        
        if not self.detection_results:
            return {"error": "No outlier detection results available"}
        
        report = {
            "detection_summary": {
                "methods_used": self.detection_results["methods_used"],
                "total_outliers": len(self.detection_results["combined_results"]["outlier_indices"]),
                "outlier_percentage": len(self.detection_results["combined_results"]["outlier_indices"]) / self.detection_results["data_shape"][0] * 100,
                "data_shape": self.detection_results["data_shape"]
            },
            "method_comparison": {},
            "outlier_analysis": self.detection_results["analysis"],
            "recommendations": self._generate_recommendations()
        }
        
        # Compare method performance
        for method, result in self.detection_results["method_results"].items():
            report["method_comparison"][method] = {
                "outliers_detected": len(result["outlier_indices"]),
                "outlier_percentage": len(result["outlier_indices"]) / self.detection_results["data_shape"][0] * 100,
                "parameters": result.get("parameters", {})
            }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on outlier detection results"""
        
        recommendations = []
        
        if not self.detection_results:
            return ["Run outlier detection first"]
        
        outlier_percentage = len(self.detection_results["combined_results"]["outlier_indices"]) / self.detection_results["data_shape"][0] * 100
        
        if outlier_percentage > 20:
            recommendations.append(
                "High outlier percentage (>20%) detected. Consider reviewing data collection process or adjusting detection parameters."
            )
        elif outlier_percentage < 1:
            recommendations.append(
                "Very low outlier percentage (<1%) detected. Consider using more sensitive detection parameters."
            )
        
        if len(self.detection_results["methods_used"]) == 1:
            recommendations.append(
                "Consider using multiple outlier detection methods for more robust results."
            )
        
        # Method-specific recommendations
        method_results = self.detection_results["method_results"]
        if "isolation_forest" in method_results:
            contamination = method_results["isolation_forest"].get("parameters", {}).get("contamination", 0.1)
            if contamination != self.config.isolation_forest_contamination:
                recommendations.append(
                    f"Isolation Forest automatically adjusted contamination to {contamination} for better results."
                )
        
        return recommendations