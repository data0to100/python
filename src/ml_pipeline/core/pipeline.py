"""
Main ML Pipeline Orchestrator

This module provides the main pipeline that orchestrates all components:
- Data loading and preprocessing
- Outlier detection and removal
- Time series vs cross-sectional data detection
- Feature engineering and selection
- Automated model selection and training
- Model evaluation and validation
- Results generation and insights
- Production deployment preparation
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import numpy as np
import pandas as pd

# Core components
from .data_loader import DataLoader
from .outlier_detector import OutlierDetector
from .time_series_detector import TimeSeriesDetector
from .feature_engineer import FeatureEngineer
from .model_selector import ModelSelector
from .preprocessor import DataPreprocessor

# Utilities
from ..utils.logger import get_logger, timing_decorator, error_handler
from ..utils.config import Config, get_config
from ..utils.visualizer import AdvancedVisualizer


class MLPipeline:
    """
    Enterprise-grade ML Pipeline for automated machine learning
    
    This pipeline provides:
    - End-to-end automation from raw data to production-ready models
    - Intelligent data type detection and processing strategies
    - Scalable processing for large datasets
    - Comprehensive model evaluation and selection
    - Production deployment utilities
    - Extensive logging and monitoring
    - Reproducible experiments with versioning
    """
    
    def __init__(self, config: Optional[Union[Config, str, Dict]] = None):
        """
        Initialize the ML Pipeline
        
        Args:
            config: Configuration object, file path, or dictionary
        """
        # Load configuration
        if isinstance(config, str):
            self.config = Config.from_yaml(config)
        elif isinstance(config, dict):
            self.config = Config(**config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = Config()
        
        # Initialize logger
        self.logger = get_logger(self.config.logging.dict())
        
        # Initialize components
        self.data_loader = DataLoader(self.config.data)
        self.outlier_detector = OutlierDetector(self.config.outlier_detection)
        self.ts_detector = TimeSeriesDetector()
        self.feature_engineer = FeatureEngineer(self.config.feature_engineering)
        self.preprocessor = DataPreprocessor(self.config)
        self.model_selector = None  # Will be initialized based on data type
        
        # Initialize visualizer
        self.visualizer = AdvancedVisualizer(
            config=self.config,
            output_dir=self.config.output_dir if hasattr(self.config, 'output_dir') else "visualizations"
        )
        
        # Pipeline state
        self.pipeline_state = {
            "initialized": True,
            "data_loaded": False,
            "outliers_detected": False,
            "data_type_detected": False,
            "features_engineered": False,
            "model_trained": False,
            "evaluation_completed": False
        }
        
        # Results storage
        self.results = {
            "data_profile": {},
            "outlier_analysis": {},
            "temporal_analysis": {},
            "feature_engineering": {},
            "model_results": {},
            "insights": {},
            "recommendations": []
        }
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.final_data = None
        self.target_column = None
        self.datetime_columns = []
        
        # Model artifacts
        self.best_model = None
        self.model_metadata = {}
        
        self.logger.log_experiment_start(
            f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
            self.config.dict()
        )
    
    @timing_decorator("complete_pipeline")
    @error_handler
    def run_complete_pipeline(self, 
                            data_source: str,
                            target_column: str,
                            test_size: float = 0.2,
                            validation_strategy: str = "holdout") -> Dict[str, Any]:
        """
        Run the complete ML pipeline from data loading to model training
        
        Args:
            data_source: Path to data file or SQL connection
            target_column: Name of target variable
            test_size: Proportion of data for testing
            validation_strategy: Validation strategy ('holdout', 'cv', 'timeseries')
        
        Returns:
            Dictionary containing complete pipeline results
        """
        self.logger.logger.info(f"Starting complete ML pipeline for target: {target_column}")
        
        try:
            # Step 1: Load and profile data
            self.load_and_profile_data(data_source)
            
            # Step 2: Detect and handle outliers
            self.detect_and_handle_outliers()
            
            # Step 3: Detect data type (time series vs cross-sectional)
            self.detect_data_type(target_column)
            
            # Step 4: Feature engineering
            self.engineer_features(target_column)
            
            # Step 5: Model selection and training
            self.select_and_train_models(target_column, test_size, validation_strategy)
            
            # Step 6: Generate insights and recommendations
            self.generate_insights_and_recommendations()
            
            # Step 7: Generate comprehensive visualizations
            self.generate_comprehensive_visualizations()
            
            # Step 8: Save pipeline artifacts
            self.save_pipeline_artifacts()
            
            self.pipeline_state["evaluation_completed"] = True
            
            self.logger.log_experiment_end("FINISHED")
            self.logger.logger.info("Complete ML pipeline finished successfully")
            
            return self.get_pipeline_results()
            
        except Exception as e:
            self.logger.log_experiment_end("FAILED")
            self.logger.logger.error(f"Pipeline failed: {e}")
            raise
    
    @timing_decorator("data_loading")
    def load_and_profile_data(self, data_source: str) -> pd.DataFrame:
        """Load data and generate comprehensive profile"""
        
        self.logger.logger.info(f"Loading data from: {data_source}")
        
        # Load data
        self.raw_data = self.data_loader.load_data(data_source)
        
        # Generate data profile
        data_profile = self.data_loader.get_data_profile(self.raw_data)
        self.results["data_profile"] = data_profile
        
        # Log data profile metrics
        self.logger.log_data_profile(data_profile)
        
        self.pipeline_state["data_loaded"] = True
        self.logger.logger.info(f"Data loaded successfully: {self.raw_data.shape}")
        
        return self.raw_data
    
    @timing_decorator("outlier_detection")
    def detect_and_handle_outliers(self, 
                                 remove_outliers: Optional[bool] = None) -> pd.DataFrame:
        """Detect and optionally remove outliers"""
        
        if not self.pipeline_state["data_loaded"]:
            raise ValueError("Data must be loaded first")
        
        remove_outliers = remove_outliers if remove_outliers is not None else self.config.outlier_detection.remove_outliers
        
        self.logger.logger.info("Starting outlier detection and analysis")
        
        # Detect outliers
        outlier_results = self.outlier_detector.detect_outliers(self.raw_data)
        self.results["outlier_analysis"] = outlier_results
        
        # Log outlier metrics
        outlier_count = len(outlier_results["combined_results"]["outlier_indices"])
        outlier_percentage = (outlier_count / len(self.raw_data)) * 100
        self.logger.log_metric("outliers_detected", outlier_count)
        self.logger.log_metric("outlier_percentage", outlier_percentage)
        
        # Remove outliers if configured
        if remove_outliers and outlier_count > 0:
            self.processed_data = self.outlier_detector.remove_outliers(self.raw_data)
            self.logger.logger.info(f"Removed {outlier_count} outliers ({outlier_percentage:.2f}%)")
        else:
            self.processed_data = self.raw_data.copy()
            self.logger.logger.info(f"Detected {outlier_count} outliers but keeping all data")
        
        self.pipeline_state["outliers_detected"] = True
        return self.processed_data
    
    @timing_decorator("data_type_detection")
    def detect_data_type(self, target_column: str) -> Dict[str, Any]:
        """Detect whether data is time series or cross-sectional"""
        
        if not self.pipeline_state["outliers_detected"]:
            raise ValueError("Outlier detection must be completed first")
        
        self.target_column = target_column
        
        self.logger.logger.info("Analyzing temporal structure and data type")
        
        # Analyze temporal structure
        temporal_analysis = self.ts_detector.analyze_temporal_structure(
            self.processed_data, target_column
        )
        
        self.results["temporal_analysis"] = temporal_analysis
        self.datetime_columns = temporal_analysis["datetime_detection"]["detected_columns"]
        
        # Log temporal analysis results
        data_type = temporal_analysis["classification"]["data_type"]
        confidence = temporal_analysis["classification"]["confidence"]
        
        self.logger.log_metric("data_type_confidence", confidence)
        self.logger.logger.info(f"Data type detected: {data_type} (confidence: {confidence:.2f})")
        
        # Store recommendations
        self.results["recommendations"].extend(temporal_analysis["recommendations"])
        
        self.pipeline_state["data_type_detected"] = True
        return temporal_analysis
    
    @timing_decorator("feature_engineering")
    def engineer_features(self, target_column: str) -> pd.DataFrame:
        """Perform comprehensive feature engineering"""
        
        if not self.pipeline_state["data_type_detected"]:
            raise ValueError("Data type detection must be completed first")
        
        self.logger.logger.info("Starting feature engineering")
        
        # Engineer features
        self.final_data = self.feature_engineer.engineer_features(
            data=self.processed_data,
            target_column=target_column,
            datetime_columns=self.datetime_columns
        )
        
        # Store feature engineering results
        self.results["feature_engineering"] = self.feature_engineer.engineering_report
        
        # Log feature engineering metrics
        original_features = self.processed_data.shape[1]
        final_features = self.final_data.shape[1]
        features_created = final_features - original_features
        
        self.logger.log_metric("original_features", original_features)
        self.logger.log_metric("final_features", final_features)
        self.logger.log_metric("features_created", features_created)
        
        self.logger.logger.info(f"Feature engineering completed: {original_features} -> {final_features} features")
        
        self.pipeline_state["features_engineered"] = True
        return self.final_data
    
    @timing_decorator("model_training")
    def select_and_train_models(self, 
                              target_column: str,
                              test_size: float = 0.2,
                              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Select and train appropriate models"""
        
        if not self.pipeline_state["features_engineered"]:
            raise ValueError("Feature engineering must be completed first")
        
        # Initialize model selector based on data type
        data_type = self.results["temporal_analysis"]["classification"]["data_type"]
        model_config = self.config.get_model_config_for_type(data_type)
        
        # Import here to avoid circular imports
        from .model_selector import ModelSelector
        self.model_selector = ModelSelector(model_config, data_type)
        
        self.logger.logger.info(f"Starting model selection for {data_type} data")
        
        # Prepare data for modeling
        X = self.final_data.drop(columns=[target_column])
        y = self.final_data[target_column]
        
        # Train and evaluate models
        model_results = self.model_selector.train_and_evaluate_models(
            X, y, test_size=test_size, validation_strategy=validation_strategy
        )
        
        self.results["model_results"] = model_results
        self.best_model = self.model_selector.best_model
        self.model_metadata = self.model_selector.get_model_metadata()
        
        # Log model results
        best_score = model_results["best_model"]["score"]
        best_model_name = model_results["best_model"]["name"]
        
        self.logger.log_metric("best_model_score", best_score)
        self.logger.log_metric("models_evaluated", len(model_results["model_comparison"]))
        
        # Log model to MLflow
        if self.best_model:
            self.logger.log_model(self.best_model, best_model_name)
        
        self.logger.logger.info(f"Model training completed. Best model: {best_model_name} (score: {best_score:.4f})")
        
        self.pipeline_state["model_trained"] = True
        return model_results
    
    def generate_insights_and_recommendations(self) -> Dict[str, Any]:
        """Generate actionable insights and recommendations"""
        
        self.logger.logger.info("Generating insights and recommendations")
        
        insights = {
            "data_insights": self._generate_data_insights(),
            "model_insights": self._generate_model_insights(),
            "business_insights": self._generate_business_insights(),
            "performance_insights": self._generate_performance_insights()
        }
        
        self.results["insights"] = insights
        
        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations()
        self.results["recommendations"].extend(recommendations)
        
        return insights
    
    def _generate_data_insights(self) -> Dict[str, Any]:
        """Generate insights about the data"""
        
        data_insights = {
            "data_quality": {
                "total_rows": len(self.raw_data),
                "total_features": self.raw_data.shape[1],
                "missing_data_percentage": self.raw_data.isnull().sum().sum() / (self.raw_data.shape[0] * self.raw_data.shape[1]) * 100,
                "duplicate_rows": self.raw_data.duplicated().sum()
            },
            "outlier_impact": {},
            "temporal_characteristics": {}
        }
        
        # Outlier insights
        if "outlier_analysis" in self.results:
            outlier_results = self.results["outlier_analysis"]
            if "combined_results" in outlier_results:
                outlier_count = len(outlier_results["combined_results"]["outlier_indices"])
                data_insights["outlier_impact"] = {
                    "outliers_detected": outlier_count,
                    "outlier_percentage": (outlier_count / len(self.raw_data)) * 100,
                    "methods_agreed": outlier_results["combined_results"].get("voting_threshold", 1),
                    "severity": "high" if outlier_count > len(self.raw_data) * 0.1 else "low"
                }
        
        # Temporal insights
        if "temporal_analysis" in self.results:
            temporal_analysis = self.results["temporal_analysis"]
            data_insights["temporal_characteristics"] = {
                "data_type": temporal_analysis["classification"]["data_type"],
                "confidence": temporal_analysis["classification"]["confidence"],
                "datetime_columns": len(temporal_analysis["datetime_detection"]["detected_columns"]),
                "has_seasonality": temporal_analysis.get("seasonality_analysis", {}).get("has_seasonality", False),
                "has_trend": temporal_analysis.get("seasonality_analysis", {}).get("has_trend", False)
            }
        
        return data_insights
    
    def _generate_model_insights(self) -> Dict[str, Any]:
        """Generate insights about model performance"""
        
        if not self.results.get("model_results"):
            return {"status": "No model results available"}
        
        model_results = self.results["model_results"]
        
        model_insights = {
            "best_model": {
                "name": model_results["best_model"]["name"],
                "score": model_results["best_model"]["score"],
                "algorithm_type": model_results["best_model"].get("algorithm_type", "unknown")
            },
            "model_comparison": {
                "models_evaluated": len(model_results["model_comparison"]),
                "score_range": self._calculate_score_range(model_results["model_comparison"]),
                "algorithm_performance": self._analyze_algorithm_performance(model_results["model_comparison"])
            },
            "feature_importance": self._get_top_features(),
            "cross_validation": model_results.get("cross_validation_scores", {})
        }
        
        return model_insights
    
    def _generate_business_insights(self) -> Dict[str, Any]:
        """Generate business-relevant insights"""
        
        business_insights = {
            "predictive_power": self._assess_predictive_power(),
            "data_requirements": self._assess_data_requirements(),
            "deployment_readiness": self._assess_deployment_readiness(),
            "risk_factors": self._identify_risk_factors()
        }
        
        return business_insights
    
    def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate performance and scalability insights"""
        
        performance_insights = {
            "pipeline_performance": {
                "total_execution_time": sum([
                    self.data_loader.load_metrics.get("load_time_seconds", 0),
                    # Add other component timings
                ]),
                "data_size_mb": self.results["data_profile"].get("memory_usage_mb", 0),
                "scalability_assessment": self._assess_scalability()
            },
            "resource_usage": {
                "memory_efficiency": self._calculate_memory_efficiency(),
                "processing_speed": self._calculate_processing_speed()
            }
        }
        
        return performance_insights
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations based on all analysis"""
        
        recommendations = []
        
        # Data quality recommendations
        data_quality = self.results["insights"]["data_insights"]["data_quality"]
        if data_quality["missing_data_percentage"] > 10:
            recommendations.append("High missing data detected (>10%). Consider data quality improvement initiatives.")
        
        # Model performance recommendations
        if "model_insights" in self.results["insights"]:
            best_score = self.results["insights"]["model_insights"]["best_model"]["score"]
            if best_score < 0.7:
                recommendations.append("Model performance is below 70%. Consider collecting more data or additional features.")
        
        # Temporal data recommendations
        temporal_chars = self.results["insights"]["data_insights"]["temporal_characteristics"]
        if temporal_chars["data_type"] == "time_series":
            if not temporal_chars["has_seasonality"] and not temporal_chars["has_trend"]:
                recommendations.append("Time series data without clear patterns detected. Verify data ordering and consider external factors.")
        
        # Feature engineering recommendations
        if "feature_engineering" in self.results:
            features_created = self.results["feature_engineering"]["summary"]["features_added"]
            if features_created > 100:
                recommendations.append("Large number of features created. Consider more aggressive feature selection to reduce overfitting.")
        
        # Outlier recommendations
        outlier_impact = self.results["insights"]["data_insights"]["outlier_impact"]
        if outlier_impact.get("severity") == "high":
            recommendations.append("High outlier percentage detected. Investigate data collection process and consider domain expertise.")
        
        return recommendations
    
    def _calculate_score_range(self, model_comparison: Dict) -> Dict[str, float]:
        """Calculate score range across models"""
        scores = [model["score"] for model in model_comparison.values()]
        return {
            "min_score": min(scores),
            "max_score": max(scores),
            "score_spread": max(scores) - min(scores)
        }
    
    def _analyze_algorithm_performance(self, model_comparison: Dict) -> Dict[str, Any]:
        """Analyze performance by algorithm type"""
        algorithm_performance = {}
        
        for model_name, results in model_comparison.items():
            algorithm = results.get("algorithm_type", "unknown")
            if algorithm not in algorithm_performance:
                algorithm_performance[algorithm] = {
                    "scores": [],
                    "count": 0
                }
            algorithm_performance[algorithm]["scores"].append(results["score"])
            algorithm_performance[algorithm]["count"] += 1
        
        # Calculate average scores
        for algorithm in algorithm_performance:
            scores = algorithm_performance[algorithm]["scores"]
            algorithm_performance[algorithm]["average_score"] = np.mean(scores)
            algorithm_performance[algorithm]["std_score"] = np.std(scores)
        
        return algorithm_performance
    
    def _get_top_features(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top important features"""
        if hasattr(self.feature_engineer, 'feature_importance') and self.feature_engineer.feature_importance:
            sorted_features = sorted(
                self.feature_engineer.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_features[:top_n]
        return []
    
    def _assess_predictive_power(self) -> Dict[str, Any]:
        """Assess the predictive power of the model"""
        if not self.results.get("model_results"):
            return {"status": "No model results available"}
        
        best_score = self.results["model_results"]["best_model"]["score"]
        
        if best_score >= 0.9:
            power_level = "excellent"
        elif best_score >= 0.8:
            power_level = "good"
        elif best_score >= 0.7:
            power_level = "moderate"
        else:
            power_level = "poor"
        
        return {
            "power_level": power_level,
            "score": best_score,
            "business_value": "high" if best_score >= 0.8 else "medium" if best_score >= 0.7 else "low"
        }
    
    def _assess_data_requirements(self) -> Dict[str, Any]:
        """Assess data requirements and quality"""
        return {
            "data_sufficiency": "sufficient" if len(self.raw_data) > 1000 else "insufficient",
            "feature_richness": "rich" if self.raw_data.shape[1] > 10 else "limited",
            "data_quality_score": 100 - self.results["insights"]["data_insights"]["data_quality"]["missing_data_percentage"]
        }
    
    def _assess_deployment_readiness(self) -> Dict[str, str]:
        """Assess readiness for deployment"""
        readiness_score = 0
        
        if self.best_model is not None:
            readiness_score += 25
        
        if self.results["insights"]["data_insights"]["data_quality"]["missing_data_percentage"] < 10:
            readiness_score += 25
        
        if self.results.get("model_results", {}).get("best_model", {}).get("score", 0) > 0.7:
            readiness_score += 25
        
        if len(self.results["recommendations"]) < 5:
            readiness_score += 25
        
        if readiness_score >= 75:
            readiness = "ready"
        elif readiness_score >= 50:
            readiness = "needs_improvement"
        else:
            readiness = "not_ready"
        
        return {
            "readiness": readiness,
            "score": readiness_score,
            "blocking_issues": [rec for rec in self.results["recommendations"] if "consider" in rec.lower()]
        }
    
    def _identify_risk_factors(self) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Data quality risks
        if self.results["insights"]["data_insights"]["data_quality"]["missing_data_percentage"] > 20:
            risks.append("High missing data may impact model reliability")
        
        # Outlier risks
        if self.results["insights"]["data_insights"]["outlier_impact"].get("severity") == "high":
            risks.append("High outlier percentage may indicate data quality issues")
        
        # Model performance risks
        if self.results.get("model_results", {}).get("best_model", {}).get("score", 1) < 0.6:
            risks.append("Low model performance may lead to poor business outcomes")
        
        # Overfitting risks
        if "feature_engineering" in self.results and self.results["feature_engineering"]["summary"]["features_added"] > self.raw_data.shape[0] / 10:
            risks.append("High feature-to-sample ratio may lead to overfitting")
        
        return risks
    
    def _assess_scalability(self) -> str:
        """Assess pipeline scalability"""
        data_size_mb = self.results["data_profile"].get("memory_usage_mb", 0)
        
        if data_size_mb < 100:
            return "high_scalability"
        elif data_size_mb < 1000:
            return "medium_scalability"
        else:
            return "low_scalability"
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        original_size = self.results["data_profile"].get("memory_usage_mb", 1)
        if "feature_engineering" in self.results:
            final_size = self.results["feature_engineering"]["summary"]["data_size_change"]["engineered_mb"]
            return original_size / final_size if final_size > 0 else 1.0
        return 1.0
    
    def _calculate_processing_speed(self) -> float:
        """Calculate processing speed (rows per second)"""
        total_time = self.data_loader.load_metrics.get("load_time_seconds", 1)
        total_rows = len(self.raw_data) if self.raw_data is not None else 1
        return total_rows / total_time
    
    def save_pipeline_artifacts(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save all pipeline artifacts"""
        
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        artifacts = {}
        
        try:
            # Save configuration
            config_path = output_path / f"config_{timestamp}.yaml"
            self.config.save_yaml(config_path)
            artifacts["config"] = str(config_path)
            
            # Save pipeline results
            results_path = output_path / f"results_{timestamp}.json"
            with open(results_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = self._convert_to_json_serializable(self.results)
                json.dump(json_results, f, indent=2, default=str)
            artifacts["results"] = str(results_path)
            
            # Save best model
            if self.best_model:
                model_path = output_path / f"best_model_{timestamp}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.best_model, f)
                artifacts["model"] = str(model_path)
            
            # Save feature transformers
            if self.feature_engineer.feature_transformers:
                transformers_path = output_path / f"feature_transformers_{timestamp}.pkl"
                with open(transformers_path, 'wb') as f:
                    pickle.dump(self.feature_engineer.feature_transformers, f)
                artifacts["transformers"] = str(transformers_path)
            
            # Save data artifacts
            if self.final_data is not None:
                data_path = output_path / f"processed_data_{timestamp}.parquet"
                self.final_data.to_parquet(data_path)
                artifacts["processed_data"] = str(data_path)
            
            self.logger.logger.info(f"Pipeline artifacts saved to: {output_path}")
            self.logger.log_artifact(str(output_path))
            
            return artifacts
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save pipeline artifacts: {e}")
            return {}
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj
    
    def load_pipeline_artifacts(self, artifacts_path: str) -> Dict[str, Any]:
        """Load saved pipeline artifacts"""
        
        artifacts_path = Path(artifacts_path)
        
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Artifacts path not found: {artifacts_path}")
        
        loaded_artifacts = {}
        
        # Load results
        results_files = list(artifacts_path.glob("results_*.json"))
        if results_files:
            with open(results_files[-1], 'r') as f:  # Load most recent
                loaded_artifacts["results"] = json.load(f)
        
        # Load model
        model_files = list(artifacts_path.glob("best_model_*.pkl"))
        if model_files:
            with open(model_files[-1], 'rb') as f:
                loaded_artifacts["model"] = pickle.load(f)
        
        # Load transformers
        transformer_files = list(artifacts_path.glob("feature_transformers_*.pkl"))
        if transformer_files:
            with open(transformer_files[-1], 'rb') as f:
                loaded_artifacts["transformers"] = pickle.load(f)
        
        return loaded_artifacts
    
    def predict_new_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data using the trained pipeline"""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Run the pipeline first.")
        
        # Apply preprocessing pipeline
        processed_data = new_data.copy()
        
        # Apply outlier detection (flagging only, not removal)
        if hasattr(self.outlier_detector, 'detection_results'):
            outlier_flags = self.outlier_detector.detect_outliers(processed_data)
            self.logger.logger.info(f"Detected {len(outlier_flags['combined_results']['outlier_indices'])} outliers in new data")
        
        # Apply feature engineering
        processed_data = self.feature_engineer.transform_new_data(processed_data)
        
        # Make predictions
        if self.target_column in processed_data.columns:
            X_new = processed_data.drop(columns=[self.target_column])
        else:
            X_new = processed_data
        
        predictions = self.best_model.predict(X_new)
        
        return predictions
    
    def get_pipeline_results(self) -> Dict[str, Any]:
        """Get comprehensive pipeline results"""
        
        return {
            "pipeline_state": self.pipeline_state,
            "results": self.results,
            "model_metadata": self.model_metadata,
            "configuration": self.config.dict(),
            "summary": {
                "data_shape": self.raw_data.shape if self.raw_data is not None else None,
                "final_features": self.final_data.shape[1] if self.final_data is not None else None,
                "best_model_score": self.results.get("model_results", {}).get("best_model", {}).get("score"),
                "data_type": self.results.get("temporal_analysis", {}).get("classification", {}).get("data_type"),
                "recommendations_count": len(self.results.get("recommendations", []))
            }
        }
    
    @timing_decorator
    @error_handler
    def generate_comprehensive_visualizations(self, 
                                             include_forecast: bool = True,
                                             include_residuals: bool = True,
                                             include_feature_importance: bool = True,
                                             include_model_comparison: bool = True,
                                             include_data_distribution: bool = True,
                                             include_outliers: bool = True,
                                             create_dashboard: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for the ML pipeline results
        
        Args:
            include_forecast: Generate time series forecast plots
            include_residuals: Generate residual analysis plots
            include_feature_importance: Generate feature importance plots
            include_model_comparison: Generate model comparison plots
            include_data_distribution: Generate data distribution plots
            include_outliers: Generate outlier visualization
            create_dashboard: Create interactive HTML dashboard
        
        Returns:
            Dict containing all generated visualizations and metadata
        """
        
        self.logger.logger.info("Generating comprehensive visualizations")
        
        visualizations = {}
        
        # 1. Time Series Forecast Plots
        if include_forecast and self.results.get("temporal_analysis", {}).get("classification", {}).get("data_type") == "time_series":
            visualizations['forecast'] = self._generate_forecast_plots()
        
        # 2. Residual Analysis
        if include_residuals and self.results.get("model_results"):
            visualizations['residuals'] = self._generate_residual_plots()
        
        # 3. Feature Importance
        if include_feature_importance and self.results.get("model_results"):
            visualizations['feature_importance'] = self._generate_feature_importance_plots()
        
        # 4. Model Comparison
        if include_model_comparison and self.results.get("model_results"):
            visualizations['model_comparison'] = self._generate_model_comparison_plots()
        
        # 5. Data Distribution
        if include_data_distribution and hasattr(self, 'processed_data'):
            visualizations['data_distribution'] = self._generate_distribution_plots()
        
        # 6. Outlier Visualization
        if include_outliers and self.results.get("outlier_analysis"):
            visualizations['outliers'] = self._generate_outlier_plots()
        
        # 7. Comprehensive Dashboard
        if create_dashboard:
            dashboard_path = self.visualizer.create_comprehensive_dashboard(self.results)
            visualizations['dashboard_path'] = dashboard_path
        
        # Store visualizations in results
        self.results['visualizations'] = visualizations
        
        self.logger.logger.info(f"Generated {len(visualizations)} visualization categories")
        return visualizations
    
    def _generate_forecast_plots(self) -> Dict[str, Any]:
        """Generate time series forecast visualizations"""
        
        try:
            # Get the best model for time series
            if self.model_selector and hasattr(self.model_selector, 'best_model'):
                model = self.model_selector.best_model
                
                # Get historical data and predictions
                if hasattr(model, 'predict'):
                    # Create forecast data (simplified example)
                    if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
                        historical_data = pd.Series(self.y_train, name='actual')
                        predictions = pd.Series(model.predict(self.X_train), name='predictions', index=historical_data.index)
                        
                        # Generate future forecast (example with simple extension)
                        forecast_steps = min(30, len(historical_data) // 4)  # Forecast 30 steps or 25% of history
                        forecast_horizon = None
                        confidence_intervals = None
                        
                        # Try to get confidence intervals if model supports it
                        if hasattr(model, 'predict_interval'):
                            try:
                                forecast_values, lower_ci, upper_ci = model.predict_interval(forecast_steps)
                                forecast_horizon = pd.Series(forecast_values, name='forecast')
                                confidence_intervals = {
                                    'lower': pd.Series(lower_ci, name='lower_ci'),
                                    'upper': pd.Series(upper_ci, name='upper_ci')
                                }
                            except:
                                pass
                        
                        return self.visualizer.create_time_series_forecast_plot(
                            actual_data=historical_data,
                            predictions=predictions,
                            forecast_horizon=forecast_horizon,
                            confidence_intervals=confidence_intervals,
                            title=f"Time Series Forecast - {model.__class__.__name__}"
                        )
        
        except Exception as e:
            self.logger.logger.warning(f"Could not generate forecast plots: {str(e)}")
        
        return {}
    
    def _generate_residual_plots(self) -> Dict[str, Any]:
        """Generate residual analysis visualizations"""
        
        try:
            if hasattr(self, 'y_test') and hasattr(self, 'X_test') and self.model_selector:
                model = self.model_selector.best_model
                predictions = model.predict(self.X_test)
                
                return self.visualizer.create_residual_analysis_plot(
                    actual=self.y_test,
                    predicted=predictions,
                    model_name=model.__class__.__name__
                )
        
        except Exception as e:
            self.logger.logger.warning(f"Could not generate residual plots: {str(e)}")
        
        return {}
    
    def _generate_feature_importance_plots(self) -> Dict[str, Any]:
        """Generate feature importance visualizations"""
        
        try:
            if self.model_selector and hasattr(self.model_selector, 'best_model'):
                model = self.model_selector.best_model
                
                # Get feature importance
                feature_importance = None
                feature_names = []
                
                if hasattr(self, 'feature_names'):
                    feature_names = self.feature_names
                elif hasattr(self, 'X_train'):
                    feature_names = list(self.X_train.columns)
                
                # Try different methods to get feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    feature_importance = np.abs(model.coef_).flatten()
                elif hasattr(model, 'feature_importance_'):
                    feature_importance = model.feature_importance_
                
                if feature_importance is not None and len(feature_names) == len(feature_importance):
                    return self.visualizer.create_feature_importance_plot(
                        feature_names=feature_names,
                        importance_scores=feature_importance,
                        model_name=model.__class__.__name__
                    )
        
        except Exception as e:
            self.logger.logger.warning(f"Could not generate feature importance plots: {str(e)}")
        
        return {}
    
    def _generate_model_comparison_plots(self) -> Dict[str, Any]:
        """Generate model comparison visualizations"""
        
        try:
            model_results = self.results.get("model_results", {})
            if "model_comparison" in model_results:
                return self.visualizer.create_model_comparison_plot(
                    model_results=model_results["model_comparison"],
                    metric_name="score"
                )
        
        except Exception as e:
            self.logger.logger.warning(f"Could not generate model comparison plots: {str(e)}")
        
        return {}
    
    def _generate_distribution_plots(self) -> Dict[str, Any]:
        """Generate data distribution visualizations"""
        
        try:
            if hasattr(self, 'processed_data'):
                target_col = getattr(self, 'target_column', None)
                return self.visualizer.create_data_distribution_plots(
                    data=self.processed_data,
                    target_column=target_col,
                    max_features=20
                )
        
        except Exception as e:
            self.logger.logger.warning(f"Could not generate distribution plots: {str(e)}")
        
        return {}
    
    def _generate_outlier_plots(self) -> Dict[str, Any]:
        """Generate outlier detection visualizations"""
        
        try:
            outlier_results = self.results.get("outlier_analysis", {})
            if "combined_results" in outlier_results and hasattr(self, 'raw_data'):
                outlier_indices = outlier_results["combined_results"].get("outlier_indices", [])
                return self.visualizer.create_outlier_visualization(
                    data=self.raw_data,
                    outlier_indices=outlier_indices,
                    feature_columns=None  # Will auto-select numeric columns
                )
        
        except Exception as e:
            self.logger.logger.warning(f"Could not generate outlier plots: {str(e)}")
        
        return {}
    
    def save_visualizations(self, output_dir: Optional[str] = None) -> str:
        """
        Save all generated visualizations to specified directory
        
        Args:
            output_dir: Output directory for visualizations
        
        Returns:
            Path to the output directory
        """
        
        if output_dir:
            self.visualizer.output_dir = Path(output_dir)
            self.visualizer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations if not already done
        if 'visualizations' not in self.results:
            self.generate_comprehensive_visualizations()
        
        # Get summary of generated plots
        summary = self.visualizer.get_generated_plots_summary()
        
        self.logger.logger.info(f"Visualizations saved to: {self.visualizer.output_dir}")
        return str(self.visualizer.output_dir)
    
    def cleanup(self):
        """Cleanup resources and connections"""
        if self.data_loader:
            self.data_loader.cleanup()
        
        # Clear large data objects to free memory
        self.raw_data = None
        self.processed_data = None
        self.final_data = None
        
        # Clear visualizations from memory
        if hasattr(self, 'visualizer'):
            self.visualizer.clear_plots()
        
        self.logger.logger.info("Pipeline cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass