"""
Machine learning models module for the ML Platform.

Provides implementations for XGBoost, LightGBM, Prophet, and other ML algorithms
with automated hyperparameter tuning and model evaluation.
"""

import time
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import joblib

from .config import MLConfig
from .exceptions import ModelError, ValidationError
from .logger import get_logger, log_model_metrics, structured_logging


class BaseModel:
    """Base class for all ML models."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        """Initialize base model.
        
        Args:
            config: ML Platform configuration
        """
        self.config = config or MLConfig()
        self.logger = get_logger()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_name = None
        self.model_type = None
        self.scaler = None
        self.label_encoder = None
        
    def save_model(self, filepath: str):
        """Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before saving", model_type=self.model_type)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.scaler = model_data.get('scaler')
        self.label_encoder = model_data.get('label_encoder')
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Model loaded from {filepath}")


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, config: Optional[MLConfig] = None, task_type: str = "classification"):
        """Initialize XGBoost model.
        
        Args:
            config: ML Platform configuration
            task_type: Type of task (classification or regression)
        """
        super().__init__(config)
        self.task_type = task_type
        self.model_type = "xgboost"
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics
        """
        with structured_logging("xgboost_training", task_type=self.task_type):
            try:
                # Store feature and target names
                self.feature_names = list(X.columns)
                self.target_name = y.name if hasattr(y, 'name') else 'target'
                
                # Split data
                test_size = kwargs.get('test_size', self.config.model.default_test_size)
                random_state = kwargs.get('random_state', self.config.model.random_state)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if self.task_type == "classification" else None
                )
                
                # Prepare parameters
                params = self._get_default_params()
                params.update(kwargs.get('model_params', {}))
                
                # Handle categorical features for classification
                if self.task_type == "classification" and y.dtype == 'object':
                    self.label_encoder = LabelEncoder()
                    y_train = pd.Series(self.label_encoder.fit_transform(y_train), index=y_train.index)
                    y_test = pd.Series(self.label_encoder.transform(y_test), index=y_test.index)
                
                # Train model
                if self.task_type == "classification":
                    self.model = xgb.XGBClassifier(**params)
                else:
                    self.model = xgb.XGBRegressor(**params)
                
                # Fit with early stopping
                eval_set = [(X_test, y_test)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
                    verbose=kwargs.get('verbose', False)
                )
                
                self.is_fitted = True
                
                # Calculate metrics
                metrics = self._calculate_metrics(X_train, X_test, y_train, y_test)
                
                # Log metrics
                log_model_metrics(
                    model_name="XGBoost",
                    model_type=self.model_type,
                    metrics=metrics,
                    stage="training"
                )
                
                return metrics
                
            except Exception as e:
                raise ModelError(f"XGBoost training failed: {str(e)}", model_type=self.model_type, stage="training")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", model_type=self.model_type, stage="prediction")
        
        try:
            predictions = self.model.predict(X)
            
            # Decode labels if using label encoder
            if self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))
            
            return predictions
            
        except Exception as e:
            raise ModelError(f"XGBoost prediction failed: {str(e)}", model_type=self.model_type, stage="prediction")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only).
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != "classification":
            raise ModelError("predict_proba only available for classification", model_type=self.model_type)
        
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", model_type=self.model_type, stage="prediction")
        
        return self.model.predict_proba(X)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        return {
            'n_estimators': self.config.model.xgb_n_estimators,
            'max_depth': self.config.model.xgb_max_depth,
            'learning_rate': self.config.model.xgb_learning_rate,
            'random_state': self.config.model.random_state,
            'n_jobs': -1
        }
    
    def _calculate_metrics(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {}
        
        if self.task_type == "classification":
            metrics.update({
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'train_precision': precision_score(y_train, train_pred, average='weighted', zero_division=0),
                'test_precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train, train_pred, average='weighted'),
                'test_recall': recall_score(y_test, test_pred, average='weighted'),
                'train_f1': f1_score(y_train, train_pred, average='weighted'),
                'test_f1': f1_score(y_test, test_pred, average='weighted')
            })
            
            # Add AUC for binary classification
            if len(np.unique(y_train)) == 2:
                train_proba = self.model.predict_proba(X_train)[:, 1]
                test_proba = self.model.predict_proba(X_test)[:, 1]
                metrics.update({
                    'train_auc': roc_auc_score(y_train, train_proba),
                    'test_auc': roc_auc_score(y_test, test_proba)
                })
        else:
            metrics.update({
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            })
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance", model_type=self.model_type)
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""
    
    def __init__(self, config: Optional[MLConfig] = None, task_type: str = "classification"):
        """Initialize LightGBM model.
        
        Args:
            config: ML Platform configuration
            task_type: Type of task (classification or regression)
        """
        super().__init__(config)
        self.task_type = task_type
        self.model_type = "lightgbm"
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the LightGBM model.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics
        """
        with structured_logging("lightgbm_training", task_type=self.task_type):
            try:
                # Store feature and target names
                self.feature_names = list(X.columns)
                self.target_name = y.name if hasattr(y, 'name') else 'target'
                
                # Split data
                test_size = kwargs.get('test_size', self.config.model.default_test_size)
                random_state = kwargs.get('random_state', self.config.model.random_state)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if self.task_type == "classification" else None
                )
                
                # Prepare parameters
                params = self._get_default_params()
                params.update(kwargs.get('model_params', {}))
                
                # Handle categorical features for classification
                if self.task_type == "classification" and y.dtype == 'object':
                    self.label_encoder = LabelEncoder()
                    y_train = pd.Series(self.label_encoder.fit_transform(y_train), index=y_train.index)
                    y_test = pd.Series(self.label_encoder.transform(y_test), index=y_test.index)
                
                # Train model
                if self.task_type == "classification":
                    self.model = lgb.LGBMClassifier(**params)
                else:
                    self.model = lgb.LGBMRegressor(**params)
                
                # Fit with early stopping
                eval_set = [(X_test, y_test)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(kwargs.get('early_stopping_rounds', 50))],
                    eval_names=['test']
                )
                
                self.is_fitted = True
                
                # Calculate metrics
                metrics = self._calculate_metrics(X_train, X_test, y_train, y_test)
                
                # Log metrics
                log_model_metrics(
                    model_name="LightGBM",
                    model_type=self.model_type,
                    metrics=metrics,
                    stage="training"
                )
                
                return metrics
                
            except Exception as e:
                raise ModelError(f"LightGBM training failed: {str(e)}", model_type=self.model_type, stage="training")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", model_type=self.model_type, stage="prediction")
        
        try:
            predictions = self.model.predict(X)
            
            # Decode labels if using label encoder
            if self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))
            
            return predictions
            
        except Exception as e:
            raise ModelError(f"LightGBM prediction failed: {str(e)}", model_type=self.model_type, stage="prediction")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type != "classification":
            raise ModelError("predict_proba only available for classification", model_type=self.model_type)
        
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", model_type=self.model_type, stage="prediction")
        
        return self.model.predict_proba(X)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters."""
        return {
            'n_estimators': self.config.model.lgb_n_estimators,
            'max_depth': self.config.model.lgb_max_depth,
            'learning_rate': self.config.model.lgb_learning_rate,
            'random_state': self.config.model.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def _calculate_metrics(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {}
        
        if self.task_type == "classification":
            metrics.update({
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'train_precision': precision_score(y_train, train_pred, average='weighted', zero_division=0),
                'test_precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train, train_pred, average='weighted'),
                'test_recall': recall_score(y_test, test_pred, average='weighted'),
                'train_f1': f1_score(y_train, train_pred, average='weighted'),
                'test_f1': f1_score(y_test, test_pred, average='weighted')
            })
            
            # Add AUC for binary classification
            if len(np.unique(y_train)) == 2:
                train_proba = self.model.predict_proba(X_train)[:, 1]
                test_proba = self.model.predict_proba(X_test)[:, 1]
                metrics.update({
                    'train_auc': roc_auc_score(y_train, train_proba),
                    'test_auc': roc_auc_score(y_test, test_proba)
                })
        else:
            metrics.update({
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            })
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted to get feature importance", model_type=self.model_type)
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class ProphetModel(BaseModel):
    """Prophet time series forecasting model."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        """Initialize Prophet model.
        
        Args:
            config: ML Platform configuration
        """
        super().__init__(config)
        self.model_type = "prophet"
        
    def fit(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the Prophet model.
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (target) columns
            **kwargs: Additional Prophet parameters
            
        Returns:
            Training metrics
        """
        with structured_logging("prophet_training"):
            try:
                # Validate input data
                required_cols = ['ds', 'y']
                if not all(col in df.columns for col in required_cols):
                    raise ValidationError(
                        f"Prophet requires columns {required_cols}. Got: {list(df.columns)}",
                        validation_type="schema"
                    )
                
                # Prepare data
                prophet_df = df[['ds', 'y']].copy()
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                
                # Initialize Prophet with parameters
                prophet_params = {
                    'yearly_seasonality': kwargs.get('yearly_seasonality', True),
                    'weekly_seasonality': kwargs.get('weekly_seasonality', True),
                    'daily_seasonality': kwargs.get('daily_seasonality', False),
                    'seasonality_mode': kwargs.get('seasonality_mode', 'additive'),
                    'changepoint_prior_scale': kwargs.get('changepoint_prior_scale', 0.05),
                    'seasonality_prior_scale': kwargs.get('seasonality_prior_scale', 10.0),
                    'holidays_prior_scale': kwargs.get('holidays_prior_scale', 10.0)
                }
                
                self.model = Prophet(**prophet_params)
                
                # Add regressors if provided
                regressors = kwargs.get('regressors', [])
                for regressor in regressors:
                    if regressor in df.columns:
                        self.model.add_regressor(regressor)
                        prophet_df[regressor] = df[regressor]
                
                # Fit the model
                self.model.fit(prophet_df)
                self.is_fitted = True
                
                # Calculate metrics using cross-validation
                metrics = self._calculate_cv_metrics(prophet_df, kwargs)
                
                # Log metrics
                log_model_metrics(
                    model_name="Prophet",
                    model_type=self.model_type,
                    metrics=metrics,
                    stage="training"
                )
                
                return metrics
                
            except Exception as e:
                raise ModelError(f"Prophet training failed: {str(e)}", model_type=self.model_type, stage="training")
    
    def predict(self, periods: int = None, future_df: pd.DataFrame = None) -> pd.DataFrame:
        """Make forecasts with the trained model.
        
        Args:
            periods: Number of periods to forecast
            future_df: Future dataframe with regressors
            
        Returns:
            Forecast dataframe
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", model_type=self.model_type, stage="prediction")
        
        try:
            if future_df is not None:
                future = future_df.copy()
            else:
                if periods is None:
                    periods = 30  # Default to 30 periods
                future = self.model.make_future_dataframe(periods=periods)
            
            forecast = self.model.predict(future)
            return forecast
            
        except Exception as e:
            raise ModelError(f"Prophet prediction failed: {str(e)}", model_type=self.model_type, stage="prediction")
    
    def _calculate_cv_metrics(self, df: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cross-validation metrics for Prophet."""
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Perform cross-validation
            initial = kwargs.get('cv_initial', '730 days')
            period = kwargs.get('cv_period', '180 days')
            horizon = kwargs.get('cv_horizon', '365 days')
            
            df_cv = cross_validation(
                self.model, 
                initial=initial, 
                period=period, 
                horizon=horizon
            )
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Return aggregated metrics
            return {
                'mse': df_p['mse'].mean(),
                'rmse': df_p['rmse'].mean(),
                'mae': df_p['mae'].mean(),
                'mape': df_p['mape'].mean(),
                'coverage': df_p['coverage'].mean() if 'coverage' in df_p.columns else None
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
            return {"error": "Cross-validation metrics unavailable"}
    
    def plot_forecast(self, forecast: pd.DataFrame = None):
        """Plot the forecast (returns plotly figure for Streamlit)."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before plotting", model_type=self.model_type)
        
        if forecast is None:
            future = self.model.make_future_dataframe(periods=365)
            forecast = self.model.predict(future)
        
        # Convert Prophet's matplotlib figure to plotly
        from prophet.plot import plot_plotly
        return plot_plotly(self.model, forecast)


class AutoML:
    """Automated machine learning pipeline."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        """Initialize AutoML.
        
        Args:
            config: ML Platform configuration
        """
        self.config = config or MLConfig()
        self.logger = get_logger()
        self.best_model = None
        self.results = {}
        
    def auto_train(self, X: pd.DataFrame, y: pd.Series, task_type: str = "auto") -> Dict[str, Any]:
        """Automatically train and compare multiple models.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: Type of task (auto, classification, regression)
            
        Returns:
            Results from all models
        """
        with structured_logging("automl_training", task_type=task_type):
            try:
                # Auto-detect task type
                if task_type == "auto":
                    task_type = self._detect_task_type(y)
                
                models_to_try = {
                    'xgboost': XGBoostModel(self.config, task_type),
                    'lightgbm': LightGBMModel(self.config, task_type)
                }
                
                results = {}
                best_score = -np.inf if task_type == "classification" else np.inf
                best_model_name = None
                
                for model_name, model in models_to_try.items():
                    try:
                        self.logger.info(f"Training {model_name}...")
                        metrics = model.fit(X, y)
                        results[model_name] = {
                            'model': model,
                            'metrics': metrics
                        }
                        
                        # Determine best model based on primary metric
                        if task_type == "classification":
                            score = metrics.get('test_accuracy', 0)
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name
                        else:
                            score = metrics.get('test_r2', -np.inf)
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name
                        
                    except Exception as e:
                        self.logger.error(f"Failed to train {model_name}: {str(e)}")
                        results[model_name] = {'error': str(e)}
                
                # Store best model
                if best_model_name:
                    self.best_model = results[best_model_name]['model']
                    self.logger.info(f"Best model: {best_model_name} (score: {best_score:.4f})")
                
                self.results = results
                return results
                
            except Exception as e:
                raise ModelError(f"AutoML training failed: {str(e)}", model_type="automl", stage="training")
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect whether the task is classification or regression.
        
        Args:
            y: Target vector
            
        Returns:
            Task type ('classification' or 'regression')
        """
        # Check if target is numeric and has many unique values (likely regression)
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = len(y.unique()) / len(y)
            if unique_ratio > 0.05:  # More than 5% unique values
                return "regression"
        
        # Check for binary or categorical (classification)
        if y.dtype == 'object' or len(y.unique()) <= 20:
            return "classification"
        
        # Default to regression for numeric data
        return "regression"
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get a comparison of all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, result in self.results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                row = {'model': model_name}
                row.update(metrics)
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def hyperparameter_tuning(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                            search_type: str = "random", n_iter: int = 50) -> Dict[str, Any]:
        """Perform hyperparameter tuning for a specific model.
        
        Args:
            model_type: Type of model (xgboost, lightgbm)
            X: Feature matrix
            y: Target vector
            search_type: Type of search (random, grid)
            n_iter: Number of iterations for random search
            
        Returns:
            Tuning results
        """
        with structured_logging("hyperparameter_tuning", model_type=model_type, search_type=search_type):
            try:
                # Get parameter grids
                param_grids = self._get_param_grids()
                
                if model_type not in param_grids:
                    raise ModelError(f"No parameter grid defined for {model_type}", model_type=model_type)
                
                # Initialize base model
                task_type = self._detect_task_type(y)
                if model_type == "xgboost":
                    base_model = XGBoostModel(self.config, task_type).model or (
                        xgb.XGBClassifier() if task_type == "classification" else xgb.XGBRegressor()
                    )
                elif model_type == "lightgbm":
                    base_model = LightGBMModel(self.config, task_type).model or (
                        lgb.LGBMClassifier() if task_type == "classification" else lgb.LGBMRegressor()
                    )
                else:
                    raise ModelError(f"Unsupported model type for tuning: {model_type}", model_type=model_type)
                
                # Setup search
                param_grid = param_grids[model_type]
                scoring = 'accuracy' if task_type == "classification" else 'r2'
                
                if search_type == "random":
                    search = RandomizedSearchCV(
                        base_model, param_grid, n_iter=n_iter, 
                        cv=self.config.model.cv_folds, scoring=scoring, 
                        random_state=self.config.model.random_state, n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        base_model, param_grid, cv=self.config.model.cv_folds, 
                        scoring=scoring, n_jobs=-1
                    )
                
                # Perform search
                search.fit(X, y)
                
                results = {
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'cv_results': search.cv_results_
                }
                
                self.logger.info(f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}")
                return results
                
            except Exception as e:
                raise ModelError(f"Hyperparameter tuning failed: {str(e)}", model_type=model_type, stage="tuning")
    
    def _get_param_grids(self) -> Dict[str, Dict[str, List]]:
        """Get parameter grids for hyperparameter tuning."""
        return {
            'xgboost': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [-1, 3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'num_leaves': [31, 50, 100, 200]
            }
        }