"""
Comprehensive Machine Learning Pipeline for Data Analysis
Automated model selection, training, evaluation, and interpretation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime
import joblib
import json

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

# SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AutoMLPipeline:
    """
    Comprehensive automated machine learning pipeline.
    """
    
    def __init__(self, 
                 problem_type: str = 'auto',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize the AutoML pipeline.
        
        Args:
            problem_type: 'classification', 'regression', or 'auto'
            cv_folds: Number of cross-validation folds
            test_size: Test set size ratio
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize containers
        self.models = {}
        self.results = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_name = None
        
        # Get model configurations
        self._setup_models()
    
    def _setup_models(self):
        """Setup model configurations for different algorithms."""
        
        # Classification models
        self.classification_models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        # Regression models
        self.regression_models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'lasso': {
                'model': Lasso(random_state=self.random_state),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10]
                }
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.classification_models['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
            self.regression_models['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.classification_models['lightgbm'] = {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
            self.regression_models['lightgbm'] = {
                'model': lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """Automatically detect if problem is classification or regression."""
        if self.problem_type != 'auto':
            return self.problem_type
        
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            # Check number of unique values
            unique_values = y.nunique()
            if unique_values <= 20 and unique_values < len(y) * 0.05:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing steps
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        return preprocessor
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            feature_selection: bool = True,
            hyperparameter_tuning: bool = True,
            scoring: Optional[str] = None) -> Dict[str, Any]:
        """
        Fit the AutoML pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_selection: Whether to perform feature selection
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            scoring: Scoring metric to use
            
        Returns:
            Dictionary with fitting results
        """
        
        logger.info("Starting AutoML pipeline...")
        start_time = datetime.now()
        
        # Store feature and target names
        self.feature_names = X.columns.tolist()
        self.target_name = y.name if y.name else 'target'
        
        # Detect problem type
        detected_type = self._detect_problem_type(y)
        logger.info(f"Detected problem type: {detected_type}")
        
        # Get appropriate models
        if detected_type == 'classification':
            models_config = self.classification_models
            if scoring is None:
                scoring = 'accuracy'
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            models_config = self.regression_models
            if scoring is None:
                scoring = 'neg_mean_squared_error'
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if detected_type == 'classification' else None
        )
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X_train)
        
        # Store results
        results = {
            'problem_type': detected_type,
            'models': {},
            'best_model_name': None,
            'best_score': float('-inf') if 'neg_' not in scoring else float('inf'),
            'feature_importance': {},
            'training_time': None
        }
        
        # Train and evaluate models
        for model_name, model_config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Create pipeline
                if hyperparameter_tuning and model_config['params']:
                    # Use RandomizedSearchCV for hyperparameter tuning
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('model', model_config['model'])
                    ])
                    
                    # Create parameter grid with pipeline prefixes
                    param_grid = {f'model__{k}': v for k, v in model_config['params'].items()}
                    
                    search = RandomizedSearchCV(
                        pipeline, 
                        param_grid,
                        n_iter=20,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state
                    )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    cv_score = search.best_score_
                    
                else:
                    # No hyperparameter tuning
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('model', model_config['model'])
                    ])
                    
                    # Cross-validation
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
                    cv_score = cv_scores.mean()
                    
                    # Fit on full training data
                    pipeline.fit(X_train, y_train)
                    best_model = pipeline
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                
                if detected_type == 'classification':
                    test_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1': f1_score(y_test, y_pred, average='weighted')
                    }
                    
                    # Add AUC for binary classification
                    if len(np.unique(y)) == 2 and hasattr(best_model, 'predict_proba'):
                        y_proba = best_model.predict_proba(X_test)[:, 1]
                        test_metrics['auc'] = roc_auc_score(y_test, y_proba)
                
                else:
                    test_metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                        'explained_variance': explained_variance_score(y_test, y_pred)
                    }
                
                # Store model results
                results['models'][model_name] = {
                    'cv_score': cv_score,
                    'test_metrics': test_metrics,
                    'model': best_model
                }
                
                # Update best model
                is_better = (cv_score > results['best_score'] if 'neg_' not in scoring 
                           else cv_score > results['best_score'])
                
                if is_better:
                    results['best_score'] = cv_score
                    results['best_model_name'] = model_name
                    self.best_model = best_model
                
                # Get feature importance if available
                if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                    # Get feature names after preprocessing
                    feature_names = self._get_feature_names_after_preprocessing(X_train)
                    importance = best_model.named_steps['model'].feature_importances_
                    results['feature_importance'][model_name] = dict(zip(feature_names, importance))
                
                logger.info(f"{model_name} - CV Score: {cv_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Calculate total training time
        end_time = datetime.now()
        results['training_time'] = (end_time - start_time).total_seconds()
        
        self.results = results
        
        logger.info(f"Best model: {results['best_model_name']} (Score: {results['best_score']:.4f})")
        logger.info(f"Total training time: {results['training_time']:.2f} seconds")
        
        return results
    
    def _get_feature_names_after_preprocessing(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after preprocessing transformations."""
        
        # Fit preprocessor to get feature names
        self.preprocessor.fit(X)
        
        feature_names = []
        
        # Get numeric feature names
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_names.extend(numeric_features)
        
        # Get categorical feature names (after one-hot encoding)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            # Get the one-hot encoder
            cat_transformer = self.preprocessor.named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            
            # Get categories for each categorical feature
            for i, feature in enumerate(categorical_features):
                categories = onehot_encoder.categories_[i]
                # Skip first category (dropped)
                for cat in categories[1:]:
                    feature_names.append(f"{feature}_{cat}")
        
        return feature_names
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification only)."""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError("Best model does not support probability predictions.")
        
        return self.best_model.predict_proba(X)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models."""
        if not self.results:
            raise ValueError("No models have been trained yet.")
        
        comparison_data = []
        
        for model_name, model_results in self.results['models'].items():
            row = {
                'Model': model_name,
                'CV_Score': model_results['cv_score']
            }
            
            # Add test metrics
            for metric, value in model_results['test_metrics'].items():
                row[f'Test_{metric.upper()}'] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('CV_Score', ascending=False)
    
    def get_feature_importance(self, model_name: Optional[str] = None, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for specified model or best model."""
        if not self.results:
            raise ValueError("No models have been trained yet.")
        
        if model_name is None:
            model_name = self.results['best_model_name']
        
        if model_name not in self.results['feature_importance']:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        importance_dict = self.results['feature_importance'][model_name]
        importance_df = pd.DataFrame([
            {'Feature': feature, 'Importance': importance}
            for feature, importance in importance_dict.items()
        ])
        
        return importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    def save_model(self, filepath: str, include_results: bool = True):
        """Save the trained model and results."""
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'problem_type': self.results.get('problem_type'),
            'best_model_name': self.results.get('best_model_name')
        }
        
        if include_results:
            # Remove model objects from results to avoid serialization issues
            results_copy = self.results.copy()
            for model_name in results_copy['models']:
                if 'model' in results_copy['models'][model_name]:
                    del results_copy['models'][model_name]['model']
            save_data['results'] = results_copy
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and results."""
        save_data = joblib.load(filepath)
        
        self.best_model = save_data['model']
        self.preprocessor = save_data['preprocessor']
        self.feature_names = save_data['feature_names']
        self.target_name = save_data['target_name']
        
        if 'results' in save_data:
            self.results = save_data['results']
        
        logger.info(f"Model loaded from {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive model performance report."""
        if not self.results:
            raise ValueError("No results to report. Train models first.")
        
        report = f"""
AUTOML PIPELINE REPORT
======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROBLEM OVERVIEW
================
• Problem Type: {self.results['problem_type'].title()}
• Target Variable: {self.target_name}
• Number of Features: {len(self.feature_names)}
• Training Time: {self.results['training_time']:.2f} seconds

BEST MODEL
==========
• Model: {self.results['best_model_name']}
• Cross-Validation Score: {self.results['best_score']:.4f}

MODEL COMPARISON
================
"""
        
        comparison_df = self.get_model_comparison()
        report += comparison_df.to_string(index=False)
        
        # Add feature importance for best model
        if self.results['best_model_name'] in self.results['feature_importance']:
            report += f"""

TOP FEATURES (Best Model: {self.results['best_model_name']})
===========================================================
"""
            importance_df = self.get_feature_importance(top_n=10)
            report += importance_df.to_string(index=False)
        
        return report


class ModelInterpreter:
    """
    Model interpretation and explanation utilities.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str]):
        """
        Initialize model interpreter.
        
        Args:
            model: Trained model
            X_train: Training data for SHAP background
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        
        if SHAP_AVAILABLE:
            self._setup_shap_explainer()
    
    def _setup_shap_explainer(self):
        """Setup SHAP explainer based on model type."""
        try:
            # Try TreeExplainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
        except:
            try:
                # Fall back to general explainer
                self.explainer = shap.Explainer(self.model, self.X_train.sample(min(100, len(self.X_train))))
            except Exception as e:
                logger.warning(f"Could not setup SHAP explainer: {e}")
                self.explainer = None
    
    def explain_prediction(self, X_instance: pd.DataFrame, plot: bool = True) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X_instance: Single instance to explain
            plot: Whether to create SHAP plots
            
        Returns:
            Dictionary with explanation results
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP not available or explainer not setup"}
        
        try:
            # Get SHAP values
            shap_values = self.explainer(X_instance)
            
            explanation = {
                'prediction': self.model.predict(X_instance)[0],
                'shap_values': shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0],
                'base_value': shap_values.base_values[0] if hasattr(shap_values, 'base_values') else self.explainer.expected_value,
                'feature_contributions': dict(zip(self.feature_names, shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]))
            }
            
            if plot:
                # Create SHAP waterfall plot
                import matplotlib.pyplot as plt
                shap.plots.waterfall(shap_values[0], show=False)
                plt.title("SHAP Waterfall Plot - Feature Contributions")
                plt.tight_layout()
                explanation['plot'] = plt.gcf()
            
            return explanation
            
        except Exception as e:
            return {"error": f"Error explaining prediction: {e}"}
    
    def global_feature_importance(self, X_sample: pd.DataFrame, plot: bool = True) -> Dict[str, Any]:
        """
        Get global feature importance using SHAP.
        
        Args:
            X_sample: Sample of data to analyze
            plot: Whether to create plots
            
        Returns:
            Dictionary with global importance results
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP not available or explainer not setup"}
        
        try:
            # Get SHAP values for sample
            shap_values = self.explainer(X_sample)
            
            # Calculate mean absolute SHAP values
            if hasattr(shap_values, 'values'):
                mean_shap = np.abs(shap_values.values).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            
            importance_dict = dict(zip(self.feature_names, mean_shap))
            
            result = {
                'feature_importance': importance_dict,
                'sorted_importance': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            }
            
            if plot:
                # Create summary plot
                import matplotlib.pyplot as plt
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
                plt.title("SHAP Summary Plot - Global Feature Importance")
                result['summary_plot'] = plt.gcf()
                
                # Create bar plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
                plt.title("SHAP Bar Plot - Average Feature Importance")
                result['bar_plot'] = plt.gcf()
            
            return result
            
        except Exception as e:
            return {"error": f"Error calculating global importance: {e}"}