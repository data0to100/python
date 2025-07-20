"""
Intelligent Model Selection and Training

This module provides automated model selection and training:
- Data type aware model selection (time series vs cross-sectional)
- AutoML frameworks integration (PyCaret, H2O, Auto-sklearn)
- Time series specific models (Prophet, ARIMA, LSTM)
- Cross-validation and model evaluation
- Hyperparameter optimization
- Ensemble methods
- Model interpretability and explainability
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib

# AutoML frameworks
try:
    import pycaret.classification as pyc_clf
    import pycaret.regression as pyc_reg
    import pycaret.time_series as pyc_ts
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

# Time series models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ..utils.logger import get_logger, timing_decorator, error_handler


class ModelSelector:
    """
    Intelligent model selection and training system
    
    Features:
    - Automatic model selection based on data characteristics
    - AutoML framework integration for rapid prototyping
    - Time series specialized models
    - Comprehensive model evaluation and comparison
    - Hyperparameter optimization
    - Model interpretability analysis
    - Production-ready model artifacts
    """
    
    def __init__(self, config: Dict[str, Any], data_type: str = "cross_sectional"):
        """
        Initialize the ModelSelector
        
        Args:
            config: Model configuration dictionary
            data_type: Type of data ('time_series' or 'cross_sectional')
        """
        self.config = config
        self.data_type = data_type
        self.logger = get_logger()
        
        # Model artifacts
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_score = -np.inf
        self.model_metadata = {}
        
        # Evaluation results
        self.evaluation_results = {}
        self.cross_validation_results = {}
        
        # Initialize model candidates based on data type
        self.model_candidates = self._get_model_candidates()
    
    def _get_model_candidates(self) -> Dict[str, Any]:
        """Get model candidates based on data type and configuration"""
        
        if self.data_type == "time_series":
            return self._get_time_series_models()
        else:
            return self._get_cross_sectional_models()
    
    def _get_time_series_models(self) -> Dict[str, Any]:
        """Get time series model candidates"""
        
        models = {}
        
        # Prophet
        if PROPHET_AVAILABLE and "prophet" in self.config.get("time_series_models", []):
            models["prophet"] = {
                "class": Prophet,
                "params": {
                    "daily_seasonality": True,
                    "weekly_seasonality": True,
                    "yearly_seasonality": True,
                    "changepoint_prior_scale": 0.05
                },
                "type": "prophet"
            }
        
        # ARIMA
        if STATSMODELS_AVAILABLE and "arima" in self.config.get("time_series_models", []):
            models["arima"] = {
                "class": ARIMA,
                "params": {"order": (1, 1, 1)},
                "type": "arima"
            }
        
        # SARIMA
        if STATSMODELS_AVAILABLE and "sarima" in self.config.get("time_series_models", []):
            models["sarima"] = {
                "class": SARIMAX,
                "params": {
                    "order": (1, 1, 1),
                    "seasonal_order": (1, 1, 1, 12)
                },
                "type": "sarima"
            }
        
        # LSTM
        if TENSORFLOW_AVAILABLE and "lstm" in self.config.get("time_series_models", []):
            models["lstm"] = {
                "class": "custom_lstm",
                "params": {
                    "units": 50,
                    "dropout": 0.2,
                    "epochs": 100,
                    "batch_size": 32
                },
                "type": "lstm"
            }
        
        return models
    
    def _get_cross_sectional_models(self) -> Dict[str, Any]:
        """Get cross-sectional model candidates"""
        
        models = {
            "random_forest": {
                "classifier": RandomForestClassifier,
                "regressor": RandomForestRegressor,
                "params": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "random_state": 42
                },
                "type": "ensemble"
            },
            "logistic_regression": {
                "classifier": LogisticRegression,
                "regressor": LinearRegression,
                "params": {
                    "random_state": 42,
                    "max_iter": 1000
                },
                "type": "linear"
            },
            "svm": {
                "classifier": SVC,
                "regressor": SVR,
                "params": {
                    "kernel": "rbf",
                    "random_state": 42
                },
                "type": "kernel"
            },
            "knn": {
                "classifier": KNeighborsClassifier,
                "regressor": KNeighborsRegressor,
                "params": {
                    "n_neighbors": 5
                },
                "type": "distance"
            },
            "decision_tree": {
                "classifier": DecisionTreeClassifier,
                "regressor": DecisionTreeRegressor,
                "params": {
                    "max_depth": 10,
                    "random_state": 42
                },
                "type": "tree"
            }
        }
        
        return models
    
    @timing_decorator("model_training")
    @error_handler
    def train_and_evaluate_models(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                test_size: float = 0.2,
                                validation_strategy: str = "holdout") -> Dict[str, Any]:
        """
        Train and evaluate all model candidates
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            validation_strategy: Validation strategy ('holdout', 'cv', 'timeseries')
        
        Returns:
            Dictionary containing model evaluation results
        """
        self.logger.logger.info(f"Training models for {self.data_type} data")
        
        # Determine if it's a classification or regression problem
        is_classification = self._is_classification_problem(y)
        problem_type = "classification" if is_classification else "regression"
        
        self.logger.logger.info(f"Detected {problem_type} problem")
        
        # Split data based on strategy
        if validation_strategy == "timeseries" and self.data_type == "time_series":
            train_test_splits = self._time_series_split(X, y, test_size)
        else:
            train_test_splits = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_splits
        
        # Train models
        if self.data_type == "time_series":
            self._train_time_series_models(X_train, y_train, X_test, y_test)
        else:
            self._train_cross_sectional_models(X_train, y_train, X_test, y_test, is_classification)
        
        # Perform cross-validation
        if validation_strategy in ["cv", "cross_validation"]:
            self._perform_cross_validation(X, y, is_classification)
        
        # Use AutoML if configured
        if self.config.get("automl_framework") and self.data_type == "cross_sectional":
            self._run_automl(X_train, y_train, X_test, y_test, is_classification)
        
        # Select best model
        self._select_best_model()
        
        # Generate evaluation results
        results = self._generate_evaluation_results(X_test, y_test, is_classification)
        
        self.logger.logger.info(f"Model training completed. Best model: {self.best_model.__class__.__name__}")
        
        return results
    
    def _is_classification_problem(self, y: pd.Series) -> bool:
        """Determine if the problem is classification or regression"""
        
        # Check if target is categorical or has limited unique values
        if y.dtype == 'object' or y.dtype.name == 'category':
            return True
        
        # For numeric targets, use heuristic based on unique values
        unique_ratio = y.nunique() / len(y)
        
        # If less than 5% unique values or less than 20 unique values, likely classification
        if unique_ratio < 0.05 or y.nunique() < 20:
            return True
        
        return False
    
    def _time_series_split(self, X: pd.DataFrame, y: pd.Series, test_size: float) -> Tuple:
        """Split time series data maintaining temporal order"""
        
        # For time series, we maintain temporal order
        split_index = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        return X_train, X_test, y_train, y_test
    
    def _train_time_series_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_test: pd.DataFrame, y_test: pd.Series):
        """Train time series specific models"""
        
        for model_name, model_config in self.model_candidates.items():
            try:
                self.logger.logger.info(f"Training {model_name}")
                
                if model_config["type"] == "prophet":
                    model = self._train_prophet_model(y_train, model_config)
                    predictions = self._predict_prophet(model, len(y_test))
                    
                elif model_config["type"] == "arima":
                    model = self._train_arima_model(y_train, model_config)
                    predictions = self._predict_arima(model, len(y_test))
                    
                elif model_config["type"] == "lstm":
                    model = self._train_lstm_model(X_train, y_train, model_config)
                    predictions = self._predict_lstm(model, X_test)
                    
                else:
                    continue
                
                # Evaluate model
                score = self._evaluate_time_series_model(y_test, predictions)
                
                self.models[model_name] = model
                self.model_scores[model_name] = score
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                
                self.logger.logger.info(f"{model_name} score: {score:.4f}")
                
            except Exception as e:
                self.logger.logger.error(f"Failed to train {model_name}: {e}")
                continue
    
    def _train_cross_sectional_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    is_classification: bool):
        """Train cross-sectional models"""
        
        for model_name, model_config in self.model_candidates.items():
            try:
                self.logger.logger.info(f"Training {model_name}")
                
                # Select appropriate model class
                if is_classification:
                    model_class = model_config["classifier"]
                else:
                    model_class = model_config["regressor"]
                
                # Initialize and train model
                model = model_class(**model_config["params"])
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Evaluate model
                if is_classification:
                    score = accuracy_score(y_test, predictions)
                else:
                    score = r2_score(y_test, predictions)
                
                self.models[model_name] = model
                self.model_scores[model_name] = score
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                
                self.logger.logger.info(f"{model_name} score: {score:.4f}")
                
            except Exception as e:
                self.logger.logger.error(f"Failed to train {model_name}: {e}")
                continue
    
    def _train_prophet_model(self, y_train: pd.Series, model_config: Dict) -> Any:
        """Train Prophet model"""
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(y_train), freq='D'),
            'y': y_train.values
        })
        
        model = Prophet(**model_config["params"])
        model.fit(df)
        
        return model
    
    def _predict_prophet(self, model: Any, periods: int) -> np.ndarray:
        """Make predictions using Prophet model"""
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast['yhat'].tail(periods).values
    
    def _train_arima_model(self, y_train: pd.Series, model_config: Dict) -> Any:
        """Train ARIMA model"""
        
        model = ARIMA(y_train, **model_config["params"])
        fitted_model = model.fit()
        
        return fitted_model
    
    def _predict_arima(self, model: Any, periods: int) -> np.ndarray:
        """Make predictions using ARIMA model"""
        
        forecast = model.forecast(steps=periods)
        
        return forecast
    
    def _train_lstm_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        model_config: Dict) -> Any:
        """Train LSTM model"""
        
        # Prepare data for LSTM
        X_scaled, y_scaled, scaler_X, scaler_y = self._prepare_lstm_data(X_train, y_train)
        
        # Build LSTM model
        model = Sequential([
            LSTM(model_config["params"]["units"], return_sequences=True, 
                 input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
            Dropout(model_config["params"]["dropout"]),
            LSTM(model_config["params"]["units"]),
            Dropout(model_config["params"]["dropout"]),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        model.fit(
            X_scaled, y_scaled,
            epochs=model_config["params"]["epochs"],
            batch_size=model_config["params"]["batch_size"],
            validation_split=0.2,
            verbose=0
        )
        
        return {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y
        }
    
    def _predict_lstm(self, model_dict: Dict, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions using LSTM model"""
        
        model = model_dict["model"]
        scaler_X = model_dict["scaler_X"]
        scaler_y = model_dict["scaler_y"]
        
        # Scale test data
        X_test_scaled = scaler_X.transform(X_test)
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        # Make predictions
        predictions_scaled = model.predict(X_test_reshaped)
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def _prepare_lstm_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Prepare data for LSTM training"""
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale features
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Scale target
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Reshape for LSTM (samples, time steps, features)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        return X_reshaped, y_scaled, scaler_X, scaler_y
    
    def _evaluate_time_series_model(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Evaluate time series model using appropriate metrics"""
        
        # Use negative RMSE as score (higher is better)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return -rmse
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, is_classification: bool):
        """Perform cross-validation on all models"""
        
        cv_folds = self.config.get("cross_validation_folds", 5)
        
        if self.data_type == "time_series":
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = cv_folds
        
        for model_name, model in self.models.items():
            try:
                if is_classification:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                
                self.cross_validation_results[model_name] = {
                    "mean_score": scores.mean(),
                    "std_score": scores.std(),
                    "scores": scores.tolist()
                }
                
                self.logger.logger.info(f"{model_name} CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.logger.warning(f"Cross-validation failed for {model_name}: {e}")
                continue
    
    def _run_automl(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series, is_classification: bool):
        """Run AutoML framework"""
        
        framework = self.config.get("automl_framework", "").lower()
        
        if framework == "pycaret" and PYCARET_AVAILABLE:
            self._run_pycaret_automl(X_train, y_train, X_test, y_test, is_classification)
        elif framework == "h2o" and H2O_AVAILABLE:
            self._run_h2o_automl(X_train, y_train, X_test, y_test, is_classification)
        else:
            self.logger.logger.warning(f"AutoML framework '{framework}' not available")
    
    def _run_pycaret_automl(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series, is_classification: bool):
        """Run PyCaret AutoML"""
        
        try:
            # Combine features and target for PyCaret
            train_data = X_train.copy()
            train_data['target'] = y_train
            
            # Setup PyCaret environment
            if is_classification:
                exp = pyc_clf.setup(
                    train_data, 
                    target='target',
                    session_id=42,
                    train_size=0.8,
                    silent=True
                )
                
                # Compare models
                best_models = pyc_clf.compare_models(
                    include=['rf', 'lr', 'svm', 'dt', 'nb'],
                    sort='Accuracy',
                    n_select=3
                )
                
                # Finalize best model
                best_model = pyc_clf.finalize_model(best_models[0])
                
                # Evaluate
                test_data = X_test.copy()
                test_data['target'] = y_test
                predictions = pyc_clf.predict_model(best_model, data=test_data)
                score = accuracy_score(y_test, predictions['Label'])
                
            else:
                exp = pyc_reg.setup(
                    train_data,
                    target='target',
                    session_id=42,
                    train_size=0.8,
                    silent=True
                )
                
                # Compare models
                best_models = pyc_reg.compare_models(
                    include=['rf', 'lr', 'svm', 'dt'],
                    sort='R2',
                    n_select=3
                )
                
                # Finalize best model
                best_model = pyc_reg.finalize_model(best_models[0])
                
                # Evaluate
                test_data = X_test.copy()
                test_data['target'] = y_test
                predictions = pyc_reg.predict_model(best_model, data=test_data)
                score = r2_score(y_test, predictions['Label'])
            
            # Store results
            self.models["pycaret_automl"] = best_model
            self.model_scores["pycaret_automl"] = score
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = best_model
            
            self.logger.logger.info(f"PyCaret AutoML score: {score:.4f}")
            
        except Exception as e:
            self.logger.logger.error(f"PyCaret AutoML failed: {e}")
    
    def _run_h2o_automl(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series, is_classification: bool):
        """Run H2O AutoML"""
        
        try:
            # Initialize H2O
            h2o.init()
            
            # Prepare data
            train_data = X_train.copy()
            train_data['target'] = y_train
            
            h2o_train = h2o.H2OFrame(train_data)
            
            # Setup AutoML
            x = h2o_train.columns[:-1]  # All columns except target
            y = "target"
            
            if is_classification:
                h2o_train[y] = h2o_train[y].asfactor()
            
            # Run AutoML
            aml = H2OAutoML(
                max_models=20,
                max_runtime_secs=self.config.get("automl_time_budget", 3600),
                seed=42
            )
            
            aml.train(x=x, y=y, training_frame=h2o_train)
            
            # Get best model
            best_model = aml.leader
            
            # Evaluate on test set
            test_data = X_test.copy()
            test_data['target'] = y_test
            h2o_test = h2o.H2OFrame(test_data)
            
            if is_classification:
                h2o_test[y] = h2o_test[y].asfactor()
            
            perf = best_model.model_performance(h2o_test)
            
            if is_classification:
                score = perf.auc()[0][1]  # AUC score
            else:
                score = perf.rmse()  # RMSE (convert to R² equivalent)
                score = 1 - (score / y_test.std())  # Rough R² approximation
            
            # Store results
            self.models["h2o_automl"] = best_model
            self.model_scores["h2o_automl"] = score
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = best_model
            
            self.logger.logger.info(f"H2O AutoML score: {score:.4f}")
            
        except Exception as e:
            self.logger.logger.error(f"H2O AutoML failed: {e}")
        finally:
            # Shutdown H2O
            try:
                h2o.shutdown(prompt=False)
            except:
                pass
    
    def _select_best_model(self):
        """Select the best performing model"""
        
        if not self.model_scores:
            self.logger.logger.warning("No models trained successfully")
            return
        
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.models[best_model_name]
        self.best_score = self.model_scores[best_model_name]
        
        self.logger.logger.info(f"Best model: {best_model_name} (score: {self.best_score:.4f})")
    
    def _generate_evaluation_results(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                   is_classification: bool) -> Dict[str, Any]:
        """Generate comprehensive evaluation results"""
        
        results = {
            "best_model": {
                "name": self._get_model_name(self.best_model),
                "score": self.best_score,
                "algorithm_type": self._get_algorithm_type(self.best_model)
            },
            "model_comparison": {},
            "cross_validation_scores": self.cross_validation_results,
            "detailed_metrics": {}
        }
        
        # Model comparison
        for model_name, score in self.model_scores.items():
            results["model_comparison"][model_name] = {
                "score": score,
                "algorithm_type": self._get_algorithm_type(self.models[model_name])
            }
        
        # Detailed metrics for best model
        if self.best_model is not None:
            predictions = self.best_model.predict(X_test)
            
            if is_classification:
                results["detailed_metrics"] = {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions, average='weighted'),
                    "recall": recall_score(y_test, predictions, average='weighted'),
                    "f1_score": f1_score(y_test, predictions, average='weighted')
                }
                
                # Add AUC if binary classification
                if len(y_test.unique()) == 2:
                    if hasattr(self.best_model, 'predict_proba'):
                        y_proba = self.best_model.predict_proba(X_test)[:, 1]
                        results["detailed_metrics"]["auc_score"] = roc_auc_score(y_test, y_proba)
            else:
                results["detailed_metrics"] = {
                    "r2_score": r2_score(y_test, predictions),
                    "mean_squared_error": mean_squared_error(y_test, predictions),
                    "mean_absolute_error": mean_absolute_error(y_test, predictions),
                    "rmse": np.sqrt(mean_squared_error(y_test, predictions))
                }
        
        return results
    
    def _get_model_name(self, model: Any) -> str:
        """Get model name from model object"""
        
        if hasattr(model, '__class__'):
            return model.__class__.__name__
        elif isinstance(model, dict) and 'model' in model:
            return model['model'].__class__.__name__
        else:
            return str(type(model))
    
    def _get_algorithm_type(self, model: Any) -> str:
        """Get algorithm type from model object"""
        
        model_name = self._get_model_name(model).lower()
        
        if 'forest' in model_name or 'tree' in model_name:
            return "tree_based"
        elif 'linear' in model_name or 'logistic' in model_name:
            return "linear"
        elif 'svm' in model_name or 'svc' in model_name:
            return "kernel"
        elif 'neighbors' in model_name or 'knn' in model_name:
            return "distance_based"
        elif 'neural' in model_name or 'lstm' in model_name:
            return "neural_network"
        elif 'prophet' in model_name:
            return "time_series"
        elif 'arima' in model_name:
            return "statistical"
        else:
            return "other"
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata about the trained models"""
        
        metadata = {
            "total_models_trained": len(self.models),
            "best_model_name": self._get_model_name(self.best_model) if self.best_model else None,
            "best_score": self.best_score,
            "data_type": self.data_type,
            "model_types": {},
            "training_config": self.config
        }
        
        # Count models by algorithm type
        for model_name, model in self.models.items():
            algorithm_type = self._get_algorithm_type(model)
            if algorithm_type not in metadata["model_types"]:
                metadata["model_types"][algorithm_type] = 0
            metadata["model_types"][algorithm_type] += 1
        
        return metadata
    
    def save_best_model(self, filepath: str):
        """Save the best model to file"""
        
        if self.best_model is None:
            raise ValueError("No trained model available to save")
        
        joblib.dump(self.best_model, filepath)
        self.logger.logger.info(f"Best model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        
        self.best_model = joblib.load(filepath)
        self.logger.logger.info(f"Model loaded from: {filepath}")