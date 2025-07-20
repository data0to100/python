# Advanced ML Pipeline for Large-Scale Data Processing

A production-ready, enterprise-grade machine learning pipeline that automatically handles large datasets (millions of rows), performs intelligent outlier detection, automatic time-series vs cross-sectional data detection, comprehensive feature engineering, and AutoML model selection.

## 🚀 Key Features

### **Automated Data Processing**
- **Large Dataset Support**: Efficiently handles millions of rows using Dask, Modin, and Vaex
- **Multiple Data Sources**: CSV, SQL, Parquet, JSON with automatic optimization
- **Memory Management**: Intelligent memory usage optimization and chunked processing
- **Data Quality Assessment**: Comprehensive data profiling and validation

### **Intelligent Outlier Detection**
- **Multiple Algorithms**: Isolation Forest, Z-score, IQR, LOF, One-Class SVM
- **Ensemble Methods**: Voting-based outlier detection for robust results
- **Scalable Processing**: Memory-efficient outlier detection for large datasets
- **Automatic Parameter Tuning**: Optimal threshold selection based on data characteristics

### **Time Series Intelligence**
- **Automatic Detection**: Intelligent identification of time-series vs cross-sectional data
- **Temporal Analysis**: Advanced datetime pattern recognition and feature extraction
- **Seasonality Detection**: Automatic identification of seasonal patterns and trends
- **Statistical Testing**: Comprehensive stationarity and autocorrelation analysis

### **Advanced Feature Engineering**
- **Datetime Features**: Cyclical encoding, holiday detection, time-based features
- **Categorical Encoding**: Target encoding, binary encoding, hash encoding for high cardinality
- **Numerical Transformations**: Scaling, polynomial features, binning, statistical features
- **Text Processing**: TF-IDF, N-grams, text statistics and NLP features
- **Interaction Features**: Automatic detection and creation of feature interactions

### **Automated Model Selection**
- **Data Type Aware**: Different model sets for time-series vs cross-sectional data
- **AutoML Integration**: PyCaret, H2O.ai, Auto-sklearn support
- **Time Series Models**: Prophet, ARIMA, SARIMA, LSTM with automatic parameter tuning
- **Cross-sectional Models**: Random Forest, XGBoost, SVM, Neural Networks
- **Model Evaluation**: Comprehensive cross-validation and performance metrics

### **Advanced Visualization Suite**
- **Time Series Plots**: Interactive forecast plots with confidence intervals
- **Model Diagnostics**: Comprehensive residual analysis and diagnostic plots
- **Feature Analysis**: Feature importance visualization and distribution analysis
- **Performance Comparison**: Model performance comparison and evaluation charts
- **Outlier Visualization**: Interactive outlier detection and analysis plots
- **Interactive Dashboards**: Comprehensive HTML dashboards with all visualizations
- **Export Options**: Both interactive (HTML) and static (PNG) plot generation

### **Production Ready**
- **MLflow Integration**: Experiment tracking, model versioning, and artifact management
- **Comprehensive Logging**: Structured logging with performance monitoring
- **Configuration Management**: Type-safe configuration with Pydantic validation
- **Error Handling**: Robust error handling with automatic retries
- **Monitoring**: Data drift detection and model performance monitoring

## 📋 Requirements

### Core Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
```

### Big Data Processing
```
dask[complete]>=2023.5.0
modin[all]>=0.21.0
vaex>=4.16.0
```

### Time Series & Forecasting
```
prophet>=1.1.4
statsmodels>=0.14.0
pmdarima>=2.0.3
sktime>=0.24.0
```

### AutoML Frameworks
```
pycaret>=3.2.0
h2o>=3.42.0
auto-sklearn>=0.15.0
```

### Production & Monitoring
```
mlflow>=2.6.0
loguru>=0.7.0
pydantic>=2.0.0
hydra-core>=1.3.0
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from ml_pipeline import MLPipeline, Config

# Initialize pipeline with default configuration
pipeline = MLPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    data_source="your_data.csv",
    target_column="target_variable"
)

# Get insights and recommendations
insights = results["results"]["insights"]
recommendations = results["results"]["recommendations"]
```

### 3. Custom Configuration
```python
# Load configuration from YAML
config = Config.from_yaml("config/custom_config.yaml")

# Or create programmatically
config = Config(
    data=DataConfig(
        use_big_data_engine=True,
        big_data_engine="dask"
    ),
    feature_engineering=FeatureEngineeringConfig(
        scaling_method="robust",
        feature_selection=True
    )
)

pipeline = MLPipeline(config)
```

## 🏗️ Architecture

### Pipeline Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loader   │───►│   Preprocessor  │───►│ Outlier Detector│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│TS Detector      │───►│Feature Engineer │───►│ Model Selector  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Insights      │◄───│    Pipeline     │───►│   Artifacts     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Classes

- **`MLPipeline`**: Main orchestrator coordinating all components
- **`DataLoader`**: Handles large-scale data loading with multiple engines
- **`OutlierDetector`**: Multi-algorithm outlier detection with ensemble methods
- **`TimeSeriesDetector`**: Intelligent temporal pattern analysis
- **`FeatureEngineer`**: Comprehensive feature creation and selection
- **`ModelSelector`**: Automated model selection and training
- **`Config`**: Type-safe configuration management

## 📊 Examples

### Time Series Analysis
```python
# The pipeline automatically detects time series data
# and applies appropriate models (Prophet, ARIMA, LSTM)

results = pipeline.run_complete_pipeline(
    data_source="sales_data.csv",
    target_column="sales",
    validation_strategy="timeseries"
)

# Access time series specific insights
temporal_analysis = results["results"]["temporal_analysis"]
seasonality = temporal_analysis["seasonality_analysis"]["has_seasonality"]
trend = temporal_analysis["seasonality_analysis"]["has_trend"]
```

### Cross-sectional Analysis
```python
# For non-temporal data, the pipeline uses
# traditional ML algorithms with AutoML

results = pipeline.run_complete_pipeline(
    data_source="customer_data.csv",
    target_column="churn_probability",
    validation_strategy="cv"
)

# Access model performance metrics
model_results = results["results"]["model_results"]
best_model = model_results["best_model"]
detailed_metrics = model_results["detailed_metrics"]
```

### Large Dataset Processing
```python
# Configure for large datasets
config = Config(
    data=DataConfig(
        use_big_data_engine=True,
        big_data_engine="dask",
        n_workers=8,
        chunk_size=50000
    )
)

pipeline = MLPipeline(config)

# Process multi-million row datasets efficiently
results = pipeline.run_complete_pipeline(
    data_source="large_dataset.csv",
    target_column="target"
)
```

## 🔧 Configuration Options

### Data Configuration
```yaml
data:
  data_source: "csv"  # csv, sql, parquet, json
  use_big_data_engine: true
  big_data_engine: "dask"  # dask, modin, vaex
  chunk_size: 10000
  n_workers: 4
  max_missing_percentage: 0.3
```

### Outlier Detection
```yaml
outlier_detection:
  methods: ["isolation_forest", "z_score", "iqr"]
  isolation_forest_contamination: 0.1
  z_score_threshold: 3.0
  remove_outliers: true
```

### Feature Engineering
```yaml
feature_engineering:
  scaling_method: "standard"  # standard, minmax, robust
  categorical_encoding: "auto"
  create_datetime_features: true
  feature_selection: true
  max_features: null
```

### Model Configuration
```yaml
model:
  automl_framework: "pycaret"  # pycaret, h2o, auto_sklearn
  time_series_models: ["prophet", "arima", "lstm"]
  cross_validation_folds: 5
  ensemble_methods: true
```

## 📈 Monitoring & Production

### MLflow Integration
```python
# Automatic experiment tracking
pipeline = MLPipeline(config)
results = pipeline.run_complete_pipeline(...)

# Models and metrics are automatically logged to MLflow
# Access via MLflow UI: http://localhost:5000
```

### Model Deployment
```python
# Save trained pipeline
artifacts = pipeline.save_pipeline_artifacts("./models/")

# Load for prediction
pipeline.load_pipeline_artifacts("./models/")
predictions = pipeline.predict_new_data(new_data)
```

### Performance Monitoring
```python
# Built-in performance tracking
performance_insights = results["results"]["insights"]["performance_insights"]

# Memory usage, processing speed, scalability assessment
memory_efficiency = performance_insights["resource_usage"]["memory_efficiency"]
processing_speed = performance_insights["resource_usage"]["processing_speed"]
```

## 📊 Advanced Visualization

The pipeline includes a comprehensive visualization suite that automatically generates professional-quality charts and interactive dashboards.

### Automatic Visualization Generation

All visualizations are automatically generated during pipeline execution:

```python
from src.ml_pipeline import MLPipeline

# Run pipeline with automatic visualization
pipeline = MLPipeline()
results = pipeline.run(data, target_column='target')

# Visualizations are automatically created and saved in 'visualizations' directory
# Dashboard is available at: visualizations/ml_pipeline_dashboard_YYYYMMDD_HHMMSS.html
```

### Standalone Visualization Usage

Use the visualizer independently for custom analysis:

```python
from src.ml_pipeline import AdvancedVisualizer
import pandas as pd
import numpy as np

# Initialize visualizer
visualizer = AdvancedVisualizer(output_dir="my_plots")

# 1. Time Series Forecast Plot
forecast_plots = visualizer.create_time_series_forecast_plot(
    actual_data=historical_data,
    predictions=predictions,
    forecast_horizon=future_forecast,
    confidence_intervals={'lower': lower_ci, 'upper': upper_ci},
    title="Sales Forecast with 95% Confidence Intervals"
)

# 2. Feature Importance Analysis
importance_plots = visualizer.create_feature_importance_plot(
    feature_names=feature_names,
    importance_scores=importance_scores,
    model_name="Random Forest",
    top_n=20
)

# 3. Model Performance Comparison
comparison_plots = visualizer.create_model_comparison_plot(
    model_results={
        'Random Forest': {'score': 0.85},
        'XGBoost': {'score': 0.82},
        'Linear Model': {'score': 0.75}
    },
    metric_name="R²"
)

# 4. Residual Analysis
residual_analysis = visualizer.create_residual_analysis_plot(
    actual=y_true,
    predicted=y_pred,
    model_name="Best Model"
)

# 5. Data Distribution Analysis
distribution_plots = visualizer.create_data_distribution_plots(
    data=dataframe,
    target_column='target',
    max_features=15
)

# 6. Outlier Visualization
outlier_plots = visualizer.create_outlier_visualization(
    data=dataframe,
    outlier_indices=outlier_indices,
    feature_columns=['feature1', 'feature2']
)

# 7. Comprehensive Dashboard
dashboard_path = visualizer.create_comprehensive_dashboard(pipeline_results)
print(f"Dashboard saved to: {dashboard_path}")
```

### Visualization Features

#### 📈 Time Series Visualizations
- **Forecast Plots**: Interactive time series with actual, fitted, and forecast values
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Autocorrelation**: ACF and PACF plots for model diagnostics

#### 🔍 Model Diagnostics
- **Residual Analysis**: Comprehensive residual plots (vs fitted, QQ, histogram, actual vs predicted)
- **Statistical Tests**: Normality tests, heteroscedasticity detection, autocorrelation analysis
- **Performance Metrics**: R², RMSE, MAE with visual representation
- **Cross-Validation**: Performance across folds visualization

#### 🎯 Feature Analysis
- **Importance Rankings**: Top N features with importance scores
- **Distribution Analysis**: Histograms and KDE plots for all features
- **Correlation Heatmaps**: Feature correlation matrices
- **Target Relationships**: Feature vs target scatter plots

#### 📊 Model Comparison
- **Performance Charts**: Bar charts comparing model metrics
- **Score Distributions**: Model performance distributions
- **Algorithm Analysis**: Performance by algorithm type
- **Hyperparameter Impact**: Parameter sensitivity analysis

#### 🚨 Data Quality Visualization
- **Outlier Detection**: 2D scatter plots highlighting outliers
- **Missing Data**: Heatmaps showing missing value patterns
- **Data Distribution**: Statistical distribution analysis
- **Quality Metrics**: Data quality score visualizations

#### 🎛️ Interactive Dashboards
- **Comprehensive Overview**: All visualizations in one HTML dashboard
- **Interactive Elements**: Zoom, pan, hover details, filtering
- **Export Options**: PNG, SVG, PDF export capabilities
- **Responsive Design**: Works on desktop and mobile devices

### Running Visualization Demo

```bash
# Run standalone visualization demo
python examples/visualization_demo.py

# This creates comprehensive examples of all visualization types
```

## 🧪 Testing & Validation

### Running Examples
```bash
# Run comprehensive examples
python examples/complete_pipeline_example.py

# This will demonstrate:
# - Time series data processing
# - Cross-sectional data analysis  
# - Custom configuration usage
# - Advanced visualization generation
```

### Unit Tests
```bash
# Run test suite
pytest tests/

# Run specific component tests
pytest tests/test_outlier_detector.py
pytest tests/test_feature_engineer.py
```

## 🎯 Advanced Features

### Custom Model Integration
```python
class CustomModel:
    def fit(self, X, y):
        # Your custom training logic
        pass
    
    def predict(self, X):
        # Your custom prediction logic
        pass

# Register custom model
pipeline.model_selector.register_custom_model("custom_model", CustomModel)
```

### Advanced Outlier Detection
```python
# Custom outlier detection ensemble
outlier_config = OutlierDetectionConfig(
    methods=["isolation_forest", "lof", "one_class_svm"],
    ensemble_method="voting",
    contamination_auto_tuning=True
)

outlier_detector = OutlierDetector(outlier_config)
```

### Time Series Forecasting
```python
# Automatic forecast generation
if pipeline.ts_detector.is_time_series:
    forecast = pipeline.model_selector.generate_forecast(
        horizon=30,  # 30 periods ahead
        confidence_intervals=True
    )
```

## 📚 Expert-Level Architecture Decisions

### **Memory Optimization**
- **Chunked Processing**: Large datasets processed in memory-efficient chunks
- **Data Type Optimization**: Automatic downcasting of numeric types
- **Lazy Evaluation**: Dask integration for out-of-core processing
- **Memory Monitoring**: Real-time memory usage tracking and optimization

### **Scalability Design**
- **Horizontal Scaling**: Multi-worker processing with Dask
- **Engine Abstraction**: Pluggable big data engines (Dask/Modin/Vaex)
- **Async Processing**: Non-blocking operations for I/O intensive tasks
- **Resource Management**: Automatic resource allocation and cleanup

### **Model Selection Strategy**
- **Data-Driven Selection**: Algorithm choice based on data characteristics
- **Ensemble Integration**: Multiple model combination strategies
- **Hyperparameter Optimization**: Bayesian optimization for model tuning
- **Cross-Validation**: Time-aware validation for temporal data

### **Production Considerations**
- **Configuration Validation**: Pydantic-based type-safe configuration
- **Error Recovery**: Graceful degradation and automatic fallbacks
- **Monitoring Integration**: Built-in drift detection and alerting
- **Version Control**: Automatic model and pipeline versioning

## 🤝 Contributing

This pipeline is designed for expert data scientists who need:
- **Production-ready code** with enterprise-grade error handling
- **Scalable processing** for large datasets
- **Automatic intelligence** to reduce manual intervention
- **Comprehensive monitoring** for production deployment
- **Modular design** for easy customization and extension

### Code Quality Standards
- **Type Hints**: Full type annotation for better IDE support
- **Documentation**: Comprehensive docstrings and expert-level comments
- **Error Handling**: Robust exception handling with detailed logging
- **Testing**: Unit tests with high coverage for all components
- **Performance**: Optimized algorithms with benchmarking

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with industry best practices and leveraging:
- **Scikit-learn** for core ML algorithms
- **Dask** for distributed computing
- **MLflow** for experiment tracking
- **Pydantic** for configuration management
- **Loguru** for advanced logging
- **Prophet** for time series forecasting
- **PyCaret** for AutoML capabilities

---

**Ready for production deployment with enterprise-grade reliability and scalability.**