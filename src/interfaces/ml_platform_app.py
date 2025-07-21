"""
Enterprise AI/ML Platform Streamlit Application

A comprehensive machine learning platform with automated insights, predictive modeling,
and advanced visualization capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import time
import io
import json
from pathlib import Path

# Import ML Platform modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ml_platform.config import MLConfig
from ml_platform.data_loader import DataLoader
from ml_platform.models import XGBoostModel, LightGBMModel, ProphetModel, AutoML
from ml_platform.visualization import VisualizationEngine, InsightsGenerator
from ml_platform.exceptions import MLPlatformError, DataLoadError, ModelError
from ml_platform.logger import get_logger

# Configure Streamlit page
st.set_page_config(
    page_title="Enterprise AI/ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_platform():
    """Initialize the ML platform components."""
    try:
        config = MLConfig()
        data_loader = DataLoader(config)
        viz_engine = VisualizationEngine(config)
        insights_generator = InsightsGenerator(config)
        automl = AutoML(config)
        logger = get_logger()
        
        return {
            'config': config,
            'data_loader': data_loader,
            'viz_engine': viz_engine,
            'insights_generator': insights_generator,
            'automl': automl,
            'logger': logger
        }
    except Exception as e:
        st.error(f"Failed to initialize platform: {str(e)}")
        return None

# Main application
def main():
    """Main application function."""
    # Initialize platform
    platform = initialize_platform()
    if platform is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🤖 Enterprise AI/ML Platform")
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "🏠 Home",
            "📊 Data Ingestion",
            "🔍 Exploratory Data Analysis",
            "🤖 AutoML Training",
            "📈 Model Performance",
            "🔮 Time Series Forecasting",
            "📋 Statistical Insights",
            "⚙️ Configuration"
        ]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Ingestion":
        show_data_ingestion_page(platform)
    elif page == "🔍 Exploratory Data Analysis":
        show_eda_page(platform)
    elif page == "🤖 AutoML Training":
        show_automl_page(platform)
    elif page == "📈 Model Performance":
        show_model_performance_page(platform)
    elif page == "🔮 Time Series Forecasting":
        show_time_series_page(platform)
    elif page == "📋 Statistical Insights":
        show_insights_page(platform)
    elif page == "⚙️ Configuration":
        show_configuration_page(platform)

def show_home_page():
    """Display the home page."""
    st.title("🤖 Enterprise AI/ML Platform")
    st.markdown("### Welcome to your production-ready machine learning platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 **Data Ingestion**
        - Multi-source support (CSV, SQL, APIs, Cloud Storage)
        - Automated data validation
        - Quality assessment
        - Smart type detection
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 **AutoML Capabilities**
        - XGBoost & LightGBM models
        - Automated hyperparameter tuning
        - Model comparison
        - Feature importance analysis
        """)
    
    with col3:
        st.markdown("""
        #### 📈 **Advanced Visualizations**
        - Interactive Plotly dashboards
        - Automated insights generation
        - Time series analysis
        - Statistical summaries
        """)
    
    st.markdown("---")
    
    # Platform overview
    st.markdown("### 🎯 Platform Features")
    
    features = [
        "🔄 **Automated EDA**: Generate comprehensive exploratory data analysis reports",
        "🎯 **Smart Modeling**: Automatic model selection and training",
        "📊 **Rich Visualizations**: Interactive charts and dashboards",
        "🔮 **Time Series**: Prophet-based forecasting capabilities",
        "☁️ **Cloud Ready**: Support for AWS, Azure, and GCP",
        "🛡️ **Enterprise Grade**: Logging, error handling, and configuration management"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    st.info("💡 **Getting Started**: Navigate to 'Data Ingestion' to load your dataset and begin your ML journey!")

def show_data_ingestion_page(platform):
    """Display the data ingestion page."""
    st.title("📊 Data Ingestion")
    st.markdown("Load data from various sources with automated validation and quality assessment.")
    
    # Data source selection
    source_type = st.selectbox(
        "Select Data Source:",
        ["CSV File Upload", "CSV URL", "SQL Database", "REST API", "Sample Dataset"]
    )
    
    df = None
    
    if source_type == "CSV File Upload":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif source_type == "CSV URL":
        url = st.text_input("Enter CSV URL:")
        if url and st.button("Load Data"):
            try:
                with st.spinner("Loading data from URL..."):
                    df = platform['data_loader'].load_data(url, source_type="csv")
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    elif source_type == "SQL Database":
        st.markdown("#### Database Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            database = st.text_input("Database")
            
        with col2:
            port = st.number_input("Port", value=5432)
            username = st.text_input("Username")
        
        password = st.text_input("Password", type="password")
        query = st.text_area("SQL Query or Table Name")
        
        if st.button("Connect and Load"):
            try:
                conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                with st.spinner("Executing query..."):
                    df = platform['data_loader'].load_data(query, source_type="sql", connection_string=conn_str)
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    elif source_type == "REST API":
        st.markdown("#### API Configuration")
        api_url = st.text_input("API Endpoint URL:")
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("HTTP Method", ["GET", "POST"])
        with col2:
            data_key = st.text_input("Data Key (for nested JSON)", value="data")
        
        headers = st.text_area("Headers (JSON format)", value="{}")
        
        if api_url and st.button("Fetch Data"):
            try:
                headers_dict = json.loads(headers) if headers.strip() else {}
                with st.spinner("Fetching data from API..."):
                    df = platform['data_loader'].load_data(
                        api_url, 
                        source_type="api",
                        method=method,
                        headers=headers_dict,
                        data_key=data_key
                    )
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    elif source_type == "Sample Dataset":
        dataset_name = st.selectbox(
            "Choose Sample Dataset:",
            ["Iris", "Boston Housing", "Wine Quality", "Titanic"]
        )
        
        if st.button("Load Sample Data"):
            df = load_sample_dataset(dataset_name)
            if df is not None:
                st.success(f"✅ Sample dataset loaded! Shape: {df.shape}")
    
    # Display loaded data
    if df is not None:
        # Store in session state
        st.session_state['df'] = df
        st.session_state['data_source'] = source_type
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10))
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data validation
        st.markdown("### ✅ Data Validation")
        with st.spinner("Validating data quality..."):
            validation_results = platform['data_loader'].validate_data(df)
        
        if validation_results['passed']:
            st.success("🎉 Data validation passed!")
        else:
            st.warning("⚠️ Data validation issues detected")
        
        # Display validation details
        col1, col2 = st.columns(2)
        
        with col1:
            if validation_results['issues']:
                st.markdown("#### 🚨 Issues")
                for issue in validation_results['issues']:
                    st.error(f"• {issue}")
        
        with col2:
            if validation_results['warnings']:
                st.markdown("#### ⚠️ Warnings")
                for warning in validation_results['warnings']:
                    st.warning(f"• {warning}")
        
        # Quality score
        quality_score = validation_results.get('summary', {}).get('data_quality_score', 0)
        st.markdown("### 📊 Data Quality Score")
        st.progress(quality_score / 100)
        st.write(f"**Score: {quality_score:.1f}/100**")

def show_eda_page(platform):
    """Display the EDA page."""
    st.title("🔍 Exploratory Data Analysis")
    
    if 'df' not in st.session_state:
        st.warning("📊 Please load data first from the Data Ingestion page.")
        return
    
    df = st.session_state['df']
    st.markdown(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Generate EDA dashboard
    with st.spinner("Generating comprehensive EDA dashboard..."):
        dashboard = platform['viz_engine'].create_eda_dashboard(df)
    
    # Overview
    st.markdown("## 📋 Dataset Overview")
    overview = dashboard['overview']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Column Types")
        st.json(overview['column_types'])
    
    with col2:
        st.markdown("### Data Quality")
        st.metric("Missing %", f"{overview['missing_percentage']:.1f}%")
        st.metric("Duplicates", overview['duplicates'])
        st.metric("Memory (MB)", f"{overview['memory_usage']:.1f}")
    
    with col3:
        st.markdown("### Column Categories")
        st.write("**Numeric:**", len(overview['numeric_columns']))
        st.write("**Categorical:**", len(overview['categorical_columns']))
        st.write("**Datetime:**", len(overview['datetime_columns']))
    
    # Automated insights
    st.markdown("## 🧠 Automated Insights")
    insights = dashboard['insights']
    for insight in insights:
        st.info(insight)
    
    # Distributions
    st.markdown("## 📊 Variable Distributions")
    distributions = dashboard['distributions']
    
    if 'numeric_distributions' in distributions:
        st.plotly_chart(distributions['numeric_distributions'], use_container_width=True)
    
    # Show categorical distributions
    categorical_plots = {k: v for k, v in distributions.items() if k.startswith('categorical_')}
    if categorical_plots:
        st.markdown("### Categorical Variables")
        for plot_name, fig in list(categorical_plots.items())[:5]:  # Show first 5
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlations
    if dashboard['correlations']:
        st.markdown("## 🔗 Correlation Analysis")
        correlations = dashboard['correlations']
        
        if 'correlation_heatmap' in correlations:
            st.plotly_chart(correlations['correlation_heatmap'], use_container_width=True)
        
        # High correlation scatter plots
        scatter_plots = {k: v for k, v in correlations.items() if k.startswith('scatter_')}
        if scatter_plots:
            st.markdown("### High Correlation Relationships")
            for plot_name, fig in list(scatter_plots.items())[:3]:  # Show first 3
                st.plotly_chart(fig, use_container_width=True)
    
    # Missing data analysis
    if dashboard['missing_data']:
        st.markdown("## ❓ Missing Data Analysis")
        missing_plots = dashboard['missing_data']
        
        for plot_name, fig in missing_plots.items():
            st.plotly_chart(fig, use_container_width=True)
    
    # Outlier analysis
    if dashboard['outliers']:
        st.markdown("## 🎯 Outlier Analysis")
        outlier_plots = dashboard['outliers']
        
        for plot_name, fig in outlier_plots.items():
            st.plotly_chart(fig, use_container_width=True)

def show_automl_page(platform):
    """Display the AutoML training page."""
    st.title("🤖 AutoML Training")
    
    if 'df' not in st.session_state:
        st.warning("📊 Please load data first from the Data Ingestion page.")
        return
    
    df = st.session_state['df']
    
    # Target selection
    st.markdown("## 🎯 Target Variable Selection")
    target_column = st.selectbox("Select target column:", df.columns)
    
    if target_column:
        # Feature selection
        st.markdown("## 📊 Feature Selection")
        available_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select features (leave empty to use all):",
            available_features,
            default=available_features
        )
        
        if not selected_features:
            selected_features = available_features
        
        # Model configuration
        st.markdown("## ⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            task_type = st.selectbox("Task Type:", ["auto", "classification", "regression"])
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            models_to_train = st.multiselect(
                "Models to train:",
                ["XGBoost", "LightGBM", "Both"],
                default=["Both"]
            )
            cross_validation = st.checkbox("Cross Validation", value=True)
        
        # Training
        if st.button("🚀 Start AutoML Training", type="primary"):
            try:
                # Prepare data
                X = df[selected_features]
                y = df[target_column]
                
                # Handle missing values
                if X.isnull().any().any():
                    st.info("Handling missing values in features...")
                    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).columns.tolist() else X.mode().iloc[0])
                
                if y.isnull().any():
                    st.info("Handling missing values in target...")
                    y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else y.mode().iloc[0])
                
                # Start training
                with st.spinner("Training models... This may take a few minutes."):
                    results = platform['automl'].auto_train(X, y, task_type=task_type)
                
                # Store results
                st.session_state['automl_results'] = results
                st.session_state['task_type'] = platform['automl']._detect_task_type(y)
                st.session_state['target_column'] = target_column
                st.session_state['selected_features'] = selected_features
                
                st.success("🎉 AutoML training completed successfully!")
                
                # Display results
                show_training_results(results, platform['automl']._detect_task_type(y))
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                platform['logger'].error(f"AutoML training error: {str(e)}")
    
    # Show previous results if available
    if 'automl_results' in st.session_state:
        st.markdown("## 📈 Previous Training Results")
        show_training_results(st.session_state['automl_results'], st.session_state['task_type'])

def show_training_results(results, task_type):
    """Display AutoML training results."""
    st.markdown("### 🏆 Model Performance Comparison")
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            row = {'Model': model_name}
            
            if task_type == "classification":
                row.update({
                    'Accuracy': f"{metrics.get('test_accuracy', 0):.4f}",
                    'Precision': f"{metrics.get('test_precision', 0):.4f}",
                    'Recall': f"{metrics.get('test_recall', 0):.4f}",
                    'F1 Score': f"{metrics.get('test_f1', 0):.4f}"
                })
            else:
                row.update({
                    'R² Score': f"{metrics.get('test_r2', 0):.4f}",
                    'MSE': f"{metrics.get('test_mse', 0):.4f}",
                    'MAE': f"{metrics.get('test_mae', 0):.4f}"
                })
            
            comparison_data.append(row)
        elif 'error' in result:
            comparison_data.append({
                'Model': model_name,
                'Status': f"❌ Error: {result['error']}"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # Feature importance for best model
    if hasattr(results, 'items'):
        for model_name, result in results.items():
            if 'model' in result and hasattr(result['model'], 'get_feature_importance'):
                try:
                    importance_df = result['model'].get_feature_importance()
                    if not importance_df.empty:
                        st.markdown(f"### 📊 Feature Importance - {model_name}")
                        fig = platform['viz_engine'].create_feature_importance_viz(importance_df)
                        st.plotly_chart(fig, use_container_width=True)
                        break
                except:
                    pass

def show_model_performance_page(platform):
    """Display model performance analysis."""
    st.title("📈 Model Performance Analysis")
    
    if 'automl_results' not in st.session_state:
        st.warning("🤖 Please train models first using the AutoML page.")
        return
    
    results = st.session_state['automl_results']
    task_type = st.session_state['task_type']
    
    # Performance visualizations
    st.markdown("## 📊 Performance Visualizations")
    
    try:
        perf_plots = platform['viz_engine'].create_model_performance_viz(results, task_type)
        
        for plot_name, fig in perf_plots.items():
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating performance visualizations: {str(e)}")
    
    # Detailed metrics
    st.markdown("## 📋 Detailed Metrics")
    
    for model_name, result in results.items():
        if 'metrics' in result:
            with st.expander(f"📈 {model_name} Metrics"):
                metrics = result['metrics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training Metrics")
                    if task_type == "classification":
                        st.metric("Accuracy", f"{metrics.get('train_accuracy', 0):.4f}")
                        st.metric("Precision", f"{metrics.get('train_precision', 0):.4f}")
                        st.metric("Recall", f"{metrics.get('train_recall', 0):.4f}")
                        st.metric("F1 Score", f"{metrics.get('train_f1', 0):.4f}")
                    else:
                        st.metric("R² Score", f"{metrics.get('train_r2', 0):.4f}")
                        st.metric("MSE", f"{metrics.get('train_mse', 0):.4f}")
                        st.metric("MAE", f"{metrics.get('train_mae', 0):.4f}")
                
                with col2:
                    st.markdown("#### Test Metrics")
                    if task_type == "classification":
                        st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
                        st.metric("Precision", f"{metrics.get('test_precision', 0):.4f}")
                        st.metric("Recall", f"{metrics.get('test_recall', 0):.4f}")
                        st.metric("F1 Score", f"{metrics.get('test_f1', 0):.4f}")
                    else:
                        st.metric("R² Score", f"{metrics.get('test_r2', 0):.4f}")
                        st.metric("MSE", f"{metrics.get('test_mse', 0):.4f}")
                        st.metric("MAE", f"{metrics.get('test_mae', 0):.4f}")

def show_time_series_page(platform):
    """Display time series forecasting page."""
    st.title("🔮 Time Series Forecasting")
    
    if 'df' not in st.session_state:
        st.warning("📊 Please load data first from the Data Ingestion page.")
        return
    
    df = st.session_state['df']
    
    # Column selection for time series
    st.markdown("## 📅 Time Series Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_column = st.selectbox("Select date column:", df.columns)
    
    with col2:
        value_column = st.selectbox("Select value column:", df.select_dtypes(include=[np.number]).columns)
    
    if date_column and value_column:
        # Prepare data for Prophet
        try:
            ts_df = df[[date_column, value_column]].copy()
            ts_df.columns = ['ds', 'y']
            ts_df['ds'] = pd.to_datetime(ts_df['ds'])
            ts_df = ts_df.dropna().sort_values('ds')
            
            st.markdown("### 📊 Time Series Preview")
            fig = platform['viz_engine'].create_time_series_viz(ts_df, 'ds', 'y')
            st.plotly_chart(fig, use_container_width=True)
            
            # Prophet configuration
            st.markdown("## ⚙️ Forecasting Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                forecast_periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=30)
                yearly_seasonality = st.checkbox("Yearly seasonality", value=True)
            
            with col2:
                weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
                daily_seasonality = st.checkbox("Daily seasonality", value=False)
            
            with col3:
                seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"])
                changepoint_prior_scale = st.slider("Changepoint prior scale", 0.001, 0.5, 0.05)
            
            # Train Prophet model
            if st.button("🚀 Generate Forecast", type="primary"):
                try:
                    with st.spinner("Training Prophet model and generating forecast..."):
                        # Initialize and train Prophet model
                        prophet_model = ProphetModel(platform['config'])
                        
                        # Train with configuration
                        metrics = prophet_model.fit(
                            ts_df,
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality,
                            seasonality_mode=seasonality_mode,
                            changepoint_prior_scale=changepoint_prior_scale
                        )
                        
                        # Generate forecast
                        forecast = prophet_model.predict(periods=forecast_periods)
                        
                        # Store results
                        st.session_state['prophet_model'] = prophet_model
                        st.session_state['forecast'] = forecast
                        st.session_state['ts_data'] = ts_df
                        
                        st.success("🎉 Forecast generated successfully!")
                        
                        # Display forecast
                        st.markdown("### 🔮 Forecast Results")
                        forecast_fig = platform['viz_engine'].create_time_series_viz(
                            ts_df, 'ds', 'y', forecast
                        )
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Display metrics
                        st.markdown("### 📊 Model Performance")
                        if 'error' not in metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                            with col2:
                                st.metric("MAPE", f"{metrics.get('mape', 0):.4f}%")
                            with col3:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                        
                        # Forecast table
                        st.markdown("### 📋 Forecast Data")
                        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
                        forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                        st.dataframe(forecast_display, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Forecasting failed: {str(e)}")
                    platform['logger'].error(f"Prophet forecasting error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error preparing time series data: {str(e)}")

def show_insights_page(platform):
    """Display statistical insights page."""
    st.title("📋 Statistical Insights")
    
    if 'df' not in st.session_state:
        st.warning("📊 Please load data first from the Data Ingestion page.")
        return
    
    df = st.session_state['df']
    
    # Generate comprehensive statistical summary
    with st.spinner("Generating comprehensive statistical analysis..."):
        summary = platform['insights_generator'].generate_statistical_summary(df)
    
    if 'error' in summary:
        st.error(f"❌ Error generating insights: {summary['error']}")
        return
    
    # Basic statistics
    if summary['basic_stats']:
        st.markdown("## 📊 Descriptive Statistics")
        
        basic_stats = summary['basic_stats']
        
        if 'descriptive_stats' in basic_stats:
            st.markdown("### Summary Statistics")
            stats_df = pd.DataFrame(basic_stats['descriptive_stats'])
            st.dataframe(stats_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        if 'skewness' in basic_stats:
            with col1:
                st.markdown("### Skewness")
                skew_df = pd.DataFrame(list(basic_stats['skewness'].items()), columns=['Variable', 'Skewness'])
                skew_df['Interpretation'] = skew_df['Skewness'].apply(
                    lambda x: "Highly Right Skewed" if x > 1 else
                              "Moderately Right Skewed" if x > 0.5 else
                              "Highly Left Skewed" if x < -1 else
                              "Moderately Left Skewed" if x < -0.5 else
                              "Approximately Normal"
                )
                st.dataframe(skew_df, use_container_width=True)
        
        if 'kurtosis' in basic_stats:
            with col2:
                st.markdown("### Kurtosis")
                kurt_df = pd.DataFrame(list(basic_stats['kurtosis'].items()), columns=['Variable', 'Kurtosis'])
                kurt_df['Interpretation'] = kurt_df['Kurtosis'].apply(
                    lambda x: "Platykurtic (Light-tailed)" if x < -1 else
                              "Leptokurtic (Heavy-tailed)" if x > 1 else
                              "Mesokurtic (Normal-like)"
                )
                st.dataframe(kurt_df, use_container_width=True)
    
    # Distribution analysis
    if summary['distribution_analysis']:
        st.markdown("## 📈 Distribution Analysis")
        
        dist_analysis = summary['distribution_analysis']
        
        if 'normality_tests' in dist_analysis:
            st.markdown("### Normality Tests")
            normality_data = []
            for var, test_result in dist_analysis['normality_tests'].items():
                if 'error' not in test_result:
                    normality_data.append({
                        'Variable': var,
                        'Test Statistic': f"{test_result['statistic']:.4f}",
                        'P-value': f"{test_result['p_value']:.4f}",
                        'Normal Distribution': "Yes" if test_result['is_normal'] else "No"
                    })
            
            if normality_data:
                normality_df = pd.DataFrame(normality_data)
                st.dataframe(normality_df, use_container_width=True)
                st.caption("📝 Normal distribution assumed if p-value > 0.05")
        
        if 'outlier_analysis' in dist_analysis:
            st.markdown("### Outlier Analysis")
            outlier_data = []
            for var, outlier_info in dist_analysis['outlier_analysis'].items():
                outlier_data.append({
                    'Variable': var,
                    'Outlier Count': outlier_info['outlier_count'],
                    'Outlier Percentage': f"{outlier_info['outlier_percentage']:.2f}%"
                })
            
            if outlier_data:
                outlier_df = pd.DataFrame(outlier_data)
                st.dataframe(outlier_df, use_container_width=True)
                st.caption("📝 Outliers detected using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)")
    
    # Relationship analysis
    if summary['relationship_analysis']:
        st.markdown("## 🔗 Relationship Analysis")
        
        rel_analysis = summary['relationship_analysis']
        
        if 'strong_correlations' in rel_analysis and rel_analysis['strong_correlations']:
            st.markdown("### Strong Correlations (|r| > 0.7)")
            corr_data = []
            for corr_info in rel_analysis['strong_correlations']:
                corr_data.append({
                    'Variable 1': corr_info['var1'],
                    'Variable 2': corr_info['var2'],
                    'Correlation': f"{corr_info['correlation']:.4f}",
                    'Strength': "Very Strong" if abs(corr_info['correlation']) > 0.9 else "Strong"
                })
            
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df, use_container_width=True)
        else:
            st.info("ℹ️ No strong correlations detected (threshold: |r| > 0.7)")
    
    # Data quality assessment
    if summary['quality_assessment']:
        st.markdown("## ✅ Data Quality Assessment")
        
        quality = summary['quality_assessment']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{quality['overall_score']:.1f}/100")
        with col2:
            st.metric("Completeness", f"{quality['completeness_score']:.1f}/100")
        with col3:
            st.metric("Consistency", f"{quality['consistency_score']:.1f}/100")
        with col4:
            st.metric("Validity", f"{quality['validity_score']:.1f}/100")
        
        if quality['issues']:
            st.markdown("### 🚨 Quality Issues")
            for issue in quality['issues']:
                st.error(f"• {issue}")

def show_configuration_page(platform):
    """Display configuration page."""
    st.title("⚙️ Configuration")
    st.markdown("Configure the ML platform settings and view system information.")
    
    # System information
    st.markdown("## 🖥️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Platform Status")
        st.success("🟢 ML Platform: Active")
        st.info("📊 Data Loader: Ready")
        st.info("🤖 AutoML: Ready")
        st.info("📈 Visualization: Ready")
    
    with col2:
        st.markdown("### Configuration")
        st.write(f"**Debug Mode:** {platform['config'].debug}")
        st.write(f"**Log Level:** {platform['config'].log_level}")
        st.write(f"**Max Workers:** {platform['config'].max_workers}")
        st.write(f"**Cache Enabled:** {platform['config'].cache_enabled}")
    
    # Model configuration
    st.markdown("## 🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### XGBoost Settings")
        st.write(f"**N Estimators:** {platform['config'].model.xgb_n_estimators}")
        st.write(f"**Max Depth:** {platform['config'].model.xgb_max_depth}")
        st.write(f"**Learning Rate:** {platform['config'].model.xgb_learning_rate}")
    
    with col2:
        st.markdown("### LightGBM Settings")
        st.write(f"**N Estimators:** {platform['config'].model.lgb_n_estimators}")
        st.write(f"**Max Depth:** {platform['config'].model.lgb_max_depth}")
        st.write(f"**Learning Rate:** {platform['config'].model.lgb_learning_rate}")
    
    # Environment variables
    st.markdown("## 🌐 Environment Variables")
    
    with st.expander("View Environment Configuration"):
        env_vars = [
            "LOG_LEVEL", "DEBUG", "MAX_WORKERS", "CACHE_ENABLED",
            "DB_HOST", "DB_PORT", "DB_NAME", "AWS_REGION"
        ]
        
        for var in env_vars:
            value = platform['config'].get(var.lower().replace('_', '.'), 'Not Set')
            st.write(f"**{var}:** {value}")

def load_sample_dataset(dataset_name):
    """Load sample datasets for testing."""
    try:
        if dataset_name == "Iris":
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            return df
        
        elif dataset_name == "Boston Housing":
            # Create a simple regression dataset
            np.random.seed(42)
            n_samples = 500
            df = pd.DataFrame({
                'rooms': np.random.normal(6, 1, n_samples),
                'age': np.random.uniform(0, 100, n_samples),
                'distance': np.random.exponential(3, n_samples),
                'price': np.random.normal(25, 10, n_samples)
            })
            return df
        
        elif dataset_name == "Wine Quality":
            # Create wine quality dataset
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'alcohol': np.random.normal(10, 2, n_samples),
                'acidity': np.random.normal(3, 0.5, n_samples),
                'sugar': np.random.exponential(2, n_samples),
                'pH': np.random.normal(3.2, 0.3, n_samples),
                'quality': np.random.choice([3, 4, 5, 6, 7, 8], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
            })
            return df
        
        elif dataset_name == "Titanic":
            # Create Titanic-like dataset
            np.random.seed(42)
            n_samples = 891
            df = pd.DataFrame({
                'age': np.random.normal(30, 12, n_samples),
                'fare': np.random.exponential(30, n_samples),
                'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
                'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
                'survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
            })
            return df
        
    except Exception as e:
        st.error(f"Error loading sample dataset: {str(e)}")
        return None

if __name__ == "__main__":
    main()