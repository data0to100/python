"""
Advanced Feature Engineering for Large-Scale ML Pipelines

This module provides comprehensive feature engineering capabilities:
- Intelligent datetime feature extraction (time-based, cyclical, holiday features)
- Advanced categorical encoding (target encoding, entity embeddings, hash encoding)
- Numerical feature transformations (scaling, binning, polynomial features)
- Automatic feature interaction detection and creation
- Text feature engineering for unstructured data
- Domain-specific feature engineering patterns
- Feature selection using multiple algorithms
- Memory-efficient processing for large datasets
"""

import re
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, SelectFromModel,
    mutual_info_regression, mutual_info_classif, f_regression, f_classif
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Advanced feature engineering
try:
    from category_encoders import (
        TargetEncoder, BinaryEncoder, HashingEncoder, 
        BackwardDifferenceEncoder, HelmertEncoder
    )
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    TEXT_VECTORIZERS_AVAILABLE = True
except ImportError:
    TEXT_VECTORIZERS_AVAILABLE = False

from ..utils.logger import get_logger, timing_decorator, error_handler
from ..utils.config import FeatureEngineeringConfig


class FeatureEngineer:
    """
    Enterprise-grade feature engineering with automatic feature generation and selection
    
    Features:
    - Intelligent datetime feature extraction with domain awareness
    - Advanced categorical encoding with automatic method selection
    - Numerical transformations and interaction detection
    - Text processing and NLP feature extraction
    - Automatic feature selection using multiple criteria
    - Memory-efficient processing for large datasets
    - Feature importance analysis and interpretability
    - Robust handling of missing values and outliers
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        """
        Initialize the FeatureEngineer
        
        Args:
            config: FeatureEngineeringConfig object with engineering parameters
        """
        self.config = config
        self.logger = get_logger()
        
        # Feature engineering artifacts
        self.feature_transformers = {}
        self.feature_importance = {}
        self.selected_features = []
        self.feature_metadata = {}
        
        # Scalers and encoders
        self.numerical_scaler = None
        self.categorical_encoders = {}
        self.feature_selector = None
        
        # Analysis results
        self.engineering_report = {}
    
    @timing_decorator("feature_engineering")
    @error_handler
    def engineer_features(self, 
                         data: pd.DataFrame,
                         target_column: Optional[str] = None,
                         datetime_columns: Optional[List[str]] = None,
                         categorical_columns: Optional[List[str]] = None,
                         numerical_columns: Optional[List[str]] = None,
                         text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Comprehensive feature engineering pipeline
        
        Args:
            data: Input DataFrame
            target_column: Target variable name
            datetime_columns: List of datetime columns
            categorical_columns: List of categorical columns  
            numerical_columns: List of numerical columns
            text_columns: List of text columns
        
        Returns:
            DataFrame with engineered features
        """
        self.logger.logger.info("Starting comprehensive feature engineering")
        
        # Make a copy to avoid modifying original data
        engineered_data = data.copy()
        
        # Auto-detect column types if not provided
        column_types = self._detect_column_types(
            engineered_data, datetime_columns, categorical_columns, 
            numerical_columns, text_columns
        )
        
        # Step 1: Handle missing values
        engineered_data = self._handle_missing_values(engineered_data, column_types)
        
        # Step 2: Engineer datetime features
        if column_types["datetime"]:
            engineered_data = self._engineer_datetime_features(
                engineered_data, column_types["datetime"]
            )
        
        # Step 3: Engineer categorical features
        if column_types["categorical"]:
            engineered_data = self._engineer_categorical_features(
                engineered_data, column_types["categorical"], target_column
            )
        
        # Step 4: Engineer numerical features
        if column_types["numerical"]:
            engineered_data = self._engineer_numerical_features(
                engineered_data, column_types["numerical"]
            )
        
        # Step 5: Engineer text features
        if column_types["text"]:
            engineered_data = self._engineer_text_features(
                engineered_data, column_types["text"]
            )
        
        # Step 6: Create interaction features
        engineered_data = self._create_interaction_features(
            engineered_data, column_types
        )
        
        # Step 7: Feature selection
        if self.config.feature_selection and target_column:
            engineered_data = self._perform_feature_selection(
                engineered_data, target_column
            )
        
        # Generate engineering report
        self.engineering_report = self._generate_engineering_report(
            data, engineered_data, column_types
        )
        
        self.logger.logger.info(
            f"Feature engineering completed. Features: {data.shape[1]} -> {engineered_data.shape[1]}"
        )
        
        return engineered_data
    
    def _detect_column_types(self, 
                           data: pd.DataFrame,
                           datetime_columns: Optional[List[str]] = None,
                           categorical_columns: Optional[List[str]] = None,
                           numerical_columns: Optional[List[str]] = None,
                           text_columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Automatically detect column types"""
        
        detected_types = {
            "datetime": datetime_columns or [],
            "categorical": categorical_columns or [],
            "numerical": numerical_columns or [],
            "text": text_columns or []
        }
        
        # Auto-detect if not provided
        for col in data.columns:
            if col in sum(detected_types.values(), []):
                continue
            
            # Detect datetime columns
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                detected_types["datetime"].append(col)
            
            # Detect numerical columns
            elif pd.api.types.is_numeric_dtype(data[col]):
                detected_types["numerical"].append(col)
            
            # Detect categorical vs text columns
            elif data[col].dtype == 'object':
                unique_ratio = data[col].nunique() / len(data)
                avg_length = data[col].astype(str).str.len().mean()
                
                # Text column heuristics
                if avg_length > 50 or unique_ratio > 0.8:
                    detected_types["text"].append(col)
                else:
                    detected_types["categorical"].append(col)
        
        # Log detected types
        for col_type, columns in detected_types.items():
            if columns:
                self.logger.logger.info(f"Detected {col_type} columns: {columns}")
        
        return detected_types
    
    def _handle_missing_values(self, 
                             data: pd.DataFrame, 
                             column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Handle missing values with type-specific strategies"""
        
        result_data = data.copy()
        
        for col_type, columns in column_types.items():
            for col in columns:
                if col not in result_data.columns:
                    continue
                
                missing_pct = result_data[col].isnull().mean()
                
                if missing_pct > self.config.feature_engineering.get("max_missing_threshold", 0.5):
                    self.logger.logger.warning(f"Dropping column {col} due to high missing percentage: {missing_pct:.2%}")
                    result_data = result_data.drop(columns=[col])
                    continue
                
                if missing_pct > 0:
                    if col_type == "numerical":
                        # Use median for numerical columns
                        result_data[col] = result_data[col].fillna(result_data[col].median())
                    elif col_type == "categorical":
                        # Use mode for categorical columns
                        mode_value = result_data[col].mode().iloc[0] if len(result_data[col].mode()) > 0 else "Unknown"
                        result_data[col] = result_data[col].fillna(mode_value)
                    elif col_type == "datetime":
                        # Forward fill for datetime columns
                        result_data[col] = result_data[col].fillna(method='ffill')
                    elif col_type == "text":
                        # Empty string for text columns
                        result_data[col] = result_data[col].fillna("")
        
        return result_data
    
    def _engineer_datetime_features(self, 
                                  data: pd.DataFrame, 
                                  datetime_columns: List[str]) -> pd.DataFrame:
        """Extract comprehensive datetime features"""
        
        result_data = data.copy()
        
        for col in datetime_columns:
            if col not in result_data.columns:
                continue
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(result_data[col]):
                result_data[col] = pd.to_datetime(result_data[col], errors='coerce')
            
            dt_series = result_data[col]
            
            # Basic datetime features
            if "year" in self.config.datetime_features:
                result_data[f"{col}_year"] = dt_series.dt.year
            if "month" in self.config.datetime_features:
                result_data[f"{col}_month"] = dt_series.dt.month
            if "day" in self.config.datetime_features:
                result_data[f"{col}_day"] = dt_series.dt.day
            if "hour" in self.config.datetime_features:
                result_data[f"{col}_hour"] = dt_series.dt.hour
            if "dayofweek" in self.config.datetime_features:
                result_data[f"{col}_dayofweek"] = dt_series.dt.dayofweek
            if "quarter" in self.config.datetime_features:
                result_data[f"{col}_quarter"] = dt_series.dt.quarter
            if "is_weekend" in self.config.datetime_features:
                result_data[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
            
            # Advanced datetime features
            if "week_of_year" in self.config.datetime_features:
                result_data[f"{col}_week_of_year"] = dt_series.dt.isocalendar().week
            if "day_of_year" in self.config.datetime_features:
                result_data[f"{col}_day_of_year"] = dt_series.dt.dayofyear
            if "is_month_start" in self.config.datetime_features:
                result_data[f"{col}_is_month_start"] = dt_series.dt.is_month_start.astype(int)
            if "is_month_end" in self.config.datetime_features:
                result_data[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(int)
            
            # Cyclical features (sine/cosine encoding)
            if "cyclical_month" in self.config.datetime_features:
                result_data[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                result_data[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
            if "cyclical_hour" in self.config.datetime_features:
                result_data[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                result_data[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
            if "cyclical_dayofweek" in self.config.datetime_features:
                result_data[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
                result_data[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
            
            # Holiday features
            if HOLIDAYS_AVAILABLE and "holidays" in self.config.datetime_features:
                # Assume US holidays, can be configured
                us_holidays = holidays.US()
                result_data[f"{col}_is_holiday"] = dt_series.dt.date.apply(lambda x: x in us_holidays).astype(int)
            
            # Time since epoch (for trend modeling)
            if "timestamp" in self.config.datetime_features:
                result_data[f"{col}_timestamp"] = dt_series.astype(np.int64) // 10**9
            
            # Relative time features (if multiple datetime columns)
            if len(datetime_columns) > 1 and "time_differences" in self.config.datetime_features:
                base_col = datetime_columns[0]
                if col != base_col and base_col in result_data.columns:
                    time_diff = (dt_series - result_data[base_col]).dt.total_seconds()
                    result_data[f"{col}_vs_{base_col}_seconds"] = time_diff
        
        return result_data
    
    def _engineer_categorical_features(self, 
                                     data: pd.DataFrame, 
                                     categorical_columns: List[str],
                                     target_column: Optional[str] = None) -> pd.DataFrame:
        """Engineer categorical features with advanced encoding techniques"""
        
        result_data = data.copy()
        
        for col in categorical_columns:
            if col not in result_data.columns or col == target_column:
                continue
            
            # Convert to string to handle mixed types
            result_data[col] = result_data[col].astype(str)
            
            unique_count = result_data[col].nunique()
            
            # Choose encoding strategy based on cardinality
            if unique_count <= 2:
                # Binary encoding for binary categories
                encoder = LabelEncoder()
                result_data[f"{col}_encoded"] = encoder.fit_transform(result_data[col])
                self.categorical_encoders[col] = encoder
                
            elif unique_count <= self.config.max_categories:
                # One-hot encoding for low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(result_data[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Add encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=result_data.index)
                result_data = pd.concat([result_data, encoded_df], axis=1)
                self.categorical_encoders[col] = encoder
                
            else:
                # High cardinality encoding strategies
                if CATEGORY_ENCODERS_AVAILABLE and target_column and target_column in result_data.columns:
                    # Target encoding for high cardinality with target
                    encoder = TargetEncoder()
                    result_data[f"{col}_target_encoded"] = encoder.fit_transform(
                        result_data[col], result_data[target_column]
                    )
                    self.categorical_encoders[col] = encoder
                    
                    # Binary encoder as alternative
                    binary_encoder = BinaryEncoder()
                    binary_encoded = binary_encoder.fit_transform(result_data[[col]])
                    binary_encoded.columns = [f"{col}_binary_{i}" for i in range(len(binary_encoded.columns))]
                    result_data = pd.concat([result_data, binary_encoded], axis=1)
                    self.categorical_encoders[f"{col}_binary"] = binary_encoder
                    
                else:
                    # Hash encoding for high cardinality without target
                    if CATEGORY_ENCODERS_AVAILABLE:
                        hash_encoder = HashingEncoder(n_components=min(32, unique_count//2))
                        hash_encoded = hash_encoder.fit_transform(result_data[[col]])
                        hash_encoded.columns = [f"{col}_hash_{i}" for i in range(len(hash_encoded.columns))]
                        result_data = pd.concat([result_data, hash_encoded], axis=1)
                        self.categorical_encoders[f"{col}_hash"] = hash_encoder
                    else:
                        # Fallback to label encoding
                        encoder = LabelEncoder()
                        result_data[f"{col}_label_encoded"] = encoder.fit_transform(result_data[col])
                        self.categorical_encoders[col] = encoder
            
            # Frequency encoding
            freq_encoding = result_data[col].value_counts().to_dict()
            result_data[f"{col}_frequency"] = result_data[col].map(freq_encoding)
            
            # Length features for string categories
            result_data[f"{col}_length"] = result_data[col].str.len()
        
        return result_data
    
    def _engineer_numerical_features(self, 
                                   data: pd.DataFrame, 
                                   numerical_columns: List[str]) -> pd.DataFrame:
        """Engineer numerical features with transformations and scaling"""
        
        result_data = data.copy()
        
        # Scaling
        if self.config.scaling_method != "none":
            if self.config.scaling_method == "standard":
                self.numerical_scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                self.numerical_scaler = MinMaxScaler()
            elif self.config.scaling_method == "robust":
                self.numerical_scaler = RobustScaler()
            
            # Apply scaling
            scaled_features = self.numerical_scaler.fit_transform(result_data[numerical_columns])
            scaled_df = pd.DataFrame(
                scaled_features, 
                columns=[f"{col}_scaled" for col in numerical_columns],
                index=result_data.index
            )
            result_data = pd.concat([result_data, scaled_df], axis=1)
        
        # Mathematical transformations
        for col in numerical_columns:
            if col not in result_data.columns:
                continue
            
            col_data = result_data[col]
            
            # Log transformation (for positive values)
            if (col_data > 0).all():
                result_data[f"{col}_log"] = np.log1p(col_data)
                result_data[f"{col}_sqrt"] = np.sqrt(col_data)
            
            # Square transformation
            result_data[f"{col}_squared"] = col_data ** 2
            
            # Binning
            try:
                result_data[f"{col}_binned"] = pd.cut(col_data, bins=5, labels=False)
            except:
                pass
            
            # Statistical features
            rolling_window = min(10, len(result_data) // 10)
            if rolling_window > 1:
                result_data[f"{col}_rolling_mean"] = col_data.rolling(window=rolling_window).mean()
                result_data[f"{col}_rolling_std"] = col_data.rolling(window=rolling_window).std()
        
        # Polynomial features
        if self.config.polynomial_features and len(numerical_columns) <= 10:  # Limit to avoid explosion
            from sklearn.preprocessing import PolynomialFeatures
            
            poly = PolynomialFeatures(
                degree=self.config.polynomial_degree, 
                include_bias=False,
                interaction_only=True
            )
            poly_features = poly.fit_transform(result_data[numerical_columns])
            poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
            
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=result_data.index)
            result_data = pd.concat([result_data, poly_df], axis=1)
            
            self.feature_transformers["polynomial"] = poly
        
        return result_data
    
    def _engineer_text_features(self, 
                              data: pd.DataFrame, 
                              text_columns: List[str]) -> pd.DataFrame:
        """Engineer text features using NLP techniques"""
        
        if not TEXT_VECTORIZERS_AVAILABLE:
            self.logger.logger.warning("Text vectorizers not available, skipping text feature engineering")
            return data
        
        result_data = data.copy()
        
        for col in text_columns:
            if col not in result_data.columns:
                continue
            
            text_data = result_data[col].fillna("").astype(str)
            
            # Basic text statistics
            result_data[f"{col}_length"] = text_data.str.len()
            result_data[f"{col}_word_count"] = text_data.str.split().str.len()
            result_data[f"{col}_char_count"] = text_data.str.len()
            result_data[f"{col}_sentence_count"] = text_data.str.count(r'[.!?]') + 1
            
            # Special character counts
            result_data[f"{col}_uppercase_count"] = text_data.str.count(r'[A-Z]')
            result_data[f"{col}_digit_count"] = text_data.str.count(r'\d')
            result_data[f"{col}_special_char_count"] = text_data.str.count(r'[!@#$%^&*(),.?":{}|<>]')
            
            # TF-IDF features (limited to top features for memory efficiency)
            if len(text_data.unique()) > 10:  # Only if sufficient variety
                tfidf = TfidfVectorizer(
                    max_features=min(100, len(text_data.unique())),
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                try:
                    tfidf_features = tfidf.fit_transform(text_data)
                    tfidf_feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])]
                    
                    tfidf_df = pd.DataFrame(
                        tfidf_features.toarray(), 
                        columns=tfidf_feature_names,
                        index=result_data.index
                    )
                    result_data = pd.concat([result_data, tfidf_df], axis=1)
                    
                    self.feature_transformers[f"{col}_tfidf"] = tfidf
                    
                except Exception as e:
                    self.logger.logger.warning(f"TF-IDF failed for column {col}: {e}")
        
        return result_data
    
    def _create_interaction_features(self, 
                                   data: pd.DataFrame, 
                                   column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Create interaction features between important variables"""
        
        result_data = data.copy()
        numerical_cols = column_types["numerical"]
        
        # Limit interactions to avoid feature explosion
        if len(numerical_cols) > 20:
            # Select top correlated features for interactions
            correlation_matrix = result_data[numerical_cols].corr().abs()
            high_corr_pairs = []
            
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                    if correlation_matrix.loc[col1, col2] > 0.3:  # Threshold for interaction
                        high_corr_pairs.append((col1, col2))
            
            # Create interactions for top correlated pairs (limit to 20)
            for col1, col2 in high_corr_pairs[:20]:
                if col1 in result_data.columns and col2 in result_data.columns:
                    # Multiplicative interaction
                    result_data[f"{col1}_x_{col2}"] = result_data[col1] * result_data[col2]
                    
                    # Ratio interaction (avoid division by zero)
                    denominator = result_data[col2].replace(0, np.nan)
                    result_data[f"{col1}_div_{col2}"] = result_data[col1] / denominator
        
        # Cross-categorical interactions (limited)
        categorical_cols = column_types["categorical"]
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols[:3]):  # Limit to first 3
                for col2 in categorical_cols[i+1:3]:
                    if col1 in result_data.columns and col2 in result_data.columns:
                        result_data[f"{col1}_x_{col2}"] = (
                            result_data[col1].astype(str) + "_" + result_data[col2].astype(str)
                        )
        
        return result_data
    
    def _perform_feature_selection(self, 
                                 data: pd.DataFrame, 
                                 target_column: str) -> pd.DataFrame:
        """Perform intelligent feature selection"""
        
        if target_column not in data.columns:
            self.logger.logger.warning(f"Target column {target_column} not found, skipping feature selection")
            return data
        
        feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns]
        y = data[target_column]
        
        # Handle categorical target
        is_classification = not pd.api.types.is_numeric_dtype(y)
        
        if is_classification:
            score_func = mutual_info_classif
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            score_func = mutual_info_regression  
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Method 1: Statistical feature selection
        if self.config.feature_selection_method == "mutual_info":
            selector = SelectKBest(score_func=score_func, k=min(len(feature_columns), self.config.max_features or len(feature_columns)))
        elif self.config.feature_selection_method == "f_test":
            score_func = f_classif if is_classification else f_regression
            selector = SelectKBest(score_func=score_func, k=min(len(feature_columns), self.config.max_features or len(feature_columns)))
        else:
            # Model-based selection
            selector = SelectFromModel(estimator, threshold='median')
        
        try:
            # Handle missing values for feature selection
            X_clean = X.fillna(X.median() if not is_classification else X.mode().iloc[0])
            
            # Fit selector
            X_selected = selector.fit_transform(X_clean, y)
            
            # Get selected feature names
            if hasattr(selector, 'get_support'):
                selected_mask = selector.get_support()
                selected_features = [col for col, selected in zip(feature_columns, selected_mask) if selected]
            else:
                # For SelectFromModel, get feature importances
                selected_features = [col for col, importance in zip(feature_columns, selector.estimator_.feature_importances_) 
                                   if importance > selector.threshold_]
            
            self.selected_features = selected_features
            self.feature_selector = selector
            
            # Create feature importance dictionary
            if hasattr(selector, 'scores_'):
                self.feature_importance = dict(zip(feature_columns, selector.scores_))
            elif hasattr(selector.estimator_, 'feature_importances_'):
                self.feature_importance = dict(zip(feature_columns, selector.estimator_.feature_importances_))
            
            # Return data with selected features + target
            result_data = data[selected_features + [target_column]].copy()
            
            self.logger.logger.info(f"Feature selection: {len(feature_columns)} -> {len(selected_features)} features")
            
            return result_data
            
        except Exception as e:
            self.logger.logger.error(f"Feature selection failed: {e}")
            return data
    
    def _generate_engineering_report(self, 
                                   original_data: pd.DataFrame,
                                   engineered_data: pd.DataFrame,
                                   column_types: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate comprehensive feature engineering report"""
        
        report = {
            "summary": {
                "original_features": original_data.shape[1],
                "engineered_features": engineered_data.shape[1],
                "features_added": engineered_data.shape[1] - original_data.shape[1],
                "data_size_change": {
                    "original_mb": original_data.memory_usage(deep=True).sum() / (1024*1024),
                    "engineered_mb": engineered_data.memory_usage(deep=True).sum() / (1024*1024)
                }
            },
            "column_types": column_types,
            "transformations_applied": [],
            "feature_importance": self.feature_importance,
            "selected_features": self.selected_features
        }
        
        # Document transformations applied
        if self.categorical_encoders:
            report["transformations_applied"].append("categorical_encoding")
        if self.numerical_scaler:
            report["transformations_applied"].append("numerical_scaling")
        if self.feature_transformers:
            report["transformations_applied"].extend(list(self.feature_transformers.keys()))
        if self.feature_selector:
            report["transformations_applied"].append("feature_selection")
        
        # Feature creation summary
        new_features = set(engineered_data.columns) - set(original_data.columns)
        report["new_features"] = {
            "datetime_features": [f for f in new_features if any(dt_col in f for dt_col in column_types["datetime"])],
            "categorical_features": [f for f in new_features if any(cat_col in f for cat_col in column_types["categorical"])],
            "numerical_features": [f for f in new_features if any(num_col in f for num_col in column_types["numerical"])],
            "text_features": [f for f in new_features if any(txt_col in f for txt_col in column_types["text"])],
            "interaction_features": [f for f in new_features if "_x_" in f or "_div_" in f]
        }
        
        return report
    
    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers"""
        
        if not self.feature_transformers and not self.categorical_encoders and not self.numerical_scaler:
            self.logger.logger.warning("No transformers fitted. Run engineer_features first.")
            return new_data
        
        result_data = new_data.copy()
        
        # Apply categorical encoders
        for col, encoder in self.categorical_encoders.items():
            if col in result_data.columns:
                try:
                    if hasattr(encoder, 'transform'):
                        result_data[f"{col}_encoded"] = encoder.transform(result_data[[col]])
                    else:
                        result_data[f"{col}_encoded"] = encoder.transform(result_data[col])
                except Exception as e:
                    self.logger.logger.warning(f"Failed to transform {col}: {e}")
        
        # Apply numerical scaler
        if self.numerical_scaler:
            numerical_cols = [col for col in result_data.columns if pd.api.types.is_numeric_dtype(result_data[col])]
            if numerical_cols:
                try:
                    scaled_features = self.numerical_scaler.transform(result_data[numerical_cols])
                    scaled_df = pd.DataFrame(
                        scaled_features,
                        columns=[f"{col}_scaled" for col in numerical_cols],
                        index=result_data.index
                    )
                    result_data = pd.concat([result_data, scaled_df], axis=1)
                except Exception as e:
                    self.logger.logger.warning(f"Failed to apply numerical scaling: {e}")
        
        # Apply feature selection
        if self.feature_selector and self.selected_features:
            # Select only the features that were selected during training
            available_features = [col for col in self.selected_features if col in result_data.columns]
            result_data = result_data[available_features]
        
        return result_data
    
    def get_feature_importance_report(self, top_n: int = 20) -> Dict[str, Any]:
        """Get feature importance analysis"""
        
        if not self.feature_importance:
            return {"error": "No feature importance available. Run feature selection first."}
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_features = sorted_features[:top_n]
        
        report = {
            "top_features": top_features,
            "total_features_analyzed": len(self.feature_importance),
            "selected_features_count": len(self.selected_features),
            "feature_categories": {
                "datetime": [f for f, _ in top_features if any(keyword in f for keyword in ['year', 'month', 'day', 'hour', 'weekend'])],
                "categorical": [f for f, _ in top_features if any(keyword in f for keyword in ['encoded', 'frequency', 'target'])],
                "numerical": [f for f, _ in top_features if any(keyword in f for keyword in ['scaled', 'log', 'sqrt', 'squared'])],
                "interaction": [f for f, _ in top_features if '_x_' in f or '_div_' in f],
                "text": [f for f, _ in top_features if 'tfidf' in f or 'length' in f or 'count' in f]
            }
        }
        
        return report