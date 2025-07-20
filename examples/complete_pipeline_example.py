#!/usr/bin/env python3
"""
Complete ML Pipeline Example

This example demonstrates the full capabilities of the ML pipeline:
1. Loading and profiling data
2. Automatic outlier detection
3. Time series vs cross-sectional data detection
4. Feature engineering
5. Model selection and training
6. Insights generation
7. Model deployment preparation

The pipeline is designed to be production-ready and handles both time series
and cross-sectional data automatically.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_pipeline import MLPipeline, Config
from ml_pipeline.utils.logger import setup_logging

def create_sample_time_series_data():
    """Create sample time series data for demonstration"""
    print("Creating sample time series data...")
    
    # Create 2 years of daily data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Generate synthetic data with trend, seasonality, and noise
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 150, n_samples)
    
    # Seasonal patterns
    yearly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    
    # Random noise
    noise = np.random.normal(0, 3, n_samples)
    
    # Some random outliers
    outlier_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    outlier_values = np.random.normal(0, 20, len(outlier_indices))
    
    # Target variable
    target = trend + yearly_seasonality + weekly_seasonality + noise
    target[outlier_indices] += outlier_values
    
    # Additional features
    data = pd.DataFrame({
        'date': dates,
        'sales': target,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'quarter': dates.quarter,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 2, n_samples),
        'marketing_spend': np.random.exponential(1000, n_samples),
        'competitor_price': 50 + np.random.normal(0, 5, n_samples),
        'economic_index': 100 + np.cumsum(np.random.normal(0, 0.1, n_samples))
    })
    
    return data

def create_sample_cross_sectional_data():
    """Create sample cross-sectional data for demonstration"""
    print("Creating sample cross-sectional data...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic customer data
    data = pd.DataFrame({
        'age': np.random.normal(40, 15, n_samples).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'loan_amount': np.random.lognormal(9, 0.8, n_samples),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'num_credit_cards': np.random.poisson(3, n_samples),
        'has_mortgage': np.random.binomial(1, 0.4, n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                           n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'city_size': np.random.choice(['Small', 'Medium', 'Large'], 
                                     n_samples, p=[0.3, 0.4, 0.3])
    })
    
    # Create target variable (loan default) with logical relationships
    default_probability = (
        -0.01 * data['age'] +
        -0.000001 * data['income'] +
        -0.01 * data['credit_score'] +
        0.00001 * data['loan_amount'] +
        -0.05 * data['employment_years'] +
        2 * data['debt_to_income'] +
        0.1 * data['num_credit_cards'] +
        -0.3 * data['has_mortgage'] +
        0.5  # base rate
    )
    
    # Convert to probabilities
    default_probability = 1 / (1 + np.exp(-default_probability))
    
    # Generate binary target
    data['loan_default'] = np.random.binomial(1, default_probability, n_samples)
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    data.loc[outlier_indices, 'income'] *= 10  # Very high income outliers
    
    return data

def run_time_series_example():
    """Run complete pipeline on time series data"""
    print("\n" + "="*80)
    print("TIME SERIES EXAMPLE")
    print("="*80)
    
    # Create sample data
    data = create_sample_time_series_data()
    
    # Save to CSV for pipeline
    data_path = "sample_time_series_data.csv"
    data.to_csv(data_path, index=False)
    
    try:
        # Initialize pipeline with configuration
        config = Config.from_yaml("config/default_config.yaml")
        
        # Override some settings for time series
        config.feature_engineering.create_datetime_features = True
        config.model.time_series_models = ["prophet", "arima"] 
        config.logging.level = "INFO"
        
        pipeline = MLPipeline(config)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            data_source=data_path,
            target_column="sales",
            test_size=0.2,
            validation_strategy="timeseries"
        )
        
        # Print results summary
        print("\n" + "-"*60)
        print("PIPELINE RESULTS SUMMARY")
        print("-"*60)
        
        summary = results["summary"]
        print(f"Data Shape: {summary['data_shape']}")
        print(f"Final Features: {summary['final_features']}")
        print(f"Best Model Score: {summary['best_model_score']:.4f}")
        print(f"Data Type: {summary['data_type']}")
        print(f"Recommendations: {summary['recommendations_count']}")
        
        # Print insights
        insights = results["results"]["insights"]
        print("\n" + "-"*60)
        print("KEY INSIGHTS")
        print("-"*60)
        
        # Data insights
        data_insights = insights["data_insights"]
        print(f"Data Quality Score: {data_insights['data_quality']['total_rows']} rows")
        print(f"Outliers Detected: {data_insights['outlier_impact'].get('outliers_detected', 'N/A')}")
        print(f"Has Seasonality: {data_insights['temporal_characteristics'].get('has_seasonality', 'N/A')}")
        print(f"Has Trend: {data_insights['temporal_characteristics'].get('has_trend', 'N/A')}")
        
        # Model insights
        model_insights = insights["model_insights"]
        print(f"Best Model: {model_insights['best_model']['name']}")
        print(f"Algorithm Type: {model_insights['best_model']['algorithm_type']}")
        
        # Print recommendations
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)
        for i, rec in enumerate(results["results"]["recommendations"][:5], 1):
            print(f"{i}. {rec}")
        
        # Cleanup
        pipeline.cleanup()
        
    finally:
        # Cleanup sample file
        if os.path.exists(data_path):
            os.remove(data_path)

def run_cross_sectional_example():
    """Run complete pipeline on cross-sectional data"""
    print("\n" + "="*80)
    print("CROSS-SECTIONAL EXAMPLE")
    print("="*80)
    
    # Create sample data
    data = create_sample_cross_sectional_data()
    
    # Save to CSV for pipeline
    data_path = "sample_cross_sectional_data.csv"
    data.to_csv(data_path, index=False)
    
    try:
        # Initialize pipeline with configuration
        config = Config.from_yaml("config/default_config.yaml")
        
        # Override some settings for cross-sectional data
        config.feature_engineering.create_datetime_features = False
        config.model.automl_framework = "pycaret"
        config.logging.level = "INFO"
        
        pipeline = MLPipeline(config)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            data_source=data_path,
            target_column="loan_default",
            test_size=0.2,
            validation_strategy="holdout"
        )
        
        # Print results summary
        print("\n" + "-"*60)
        print("PIPELINE RESULTS SUMMARY")
        print("-"*60)
        
        summary = results["summary"]
        print(f"Data Shape: {summary['data_shape']}")
        print(f"Final Features: {summary['final_features']}")
        print(f"Best Model Score: {summary['best_model_score']:.4f}")
        print(f"Data Type: {summary['data_type']}")
        print(f"Recommendations: {summary['recommendations_count']}")
        
        # Print insights
        insights = results["results"]["insights"]
        print("\n" + "-"*60)
        print("KEY INSIGHTS")
        print("-"*60)
        
        # Data insights
        data_insights = insights["data_insights"]
        print(f"Data Quality Score: {data_insights['data_quality']['total_rows']} rows")
        print(f"Missing Data: {data_insights['data_quality']['missing_data_percentage']:.2f}%")
        print(f"Outliers Detected: {data_insights['outlier_impact'].get('outliers_detected', 'N/A')}")
        
        # Model insights
        model_insights = insights["model_insights"]
        print(f"Best Model: {model_insights['best_model']['name']}")
        print(f"Algorithm Type: {model_insights['best_model']['algorithm_type']}")
        
        # Detailed metrics
        detailed_metrics = results["results"]["model_results"]["detailed_metrics"]
        print(f"Accuracy: {detailed_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Precision: {detailed_metrics.get('precision', 'N/A'):.4f}")
        print(f"Recall: {detailed_metrics.get('recall', 'N/A'):.4f}")
        print(f"F1 Score: {detailed_metrics.get('f1_score', 'N/A'):.4f}")
        
        # Business insights
        business_insights = insights["business_insights"]
        print(f"Predictive Power: {business_insights['predictive_power']['power_level']}")
        print(f"Deployment Readiness: {business_insights['deployment_readiness']['readiness']}")
        
        # Print recommendations
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)
        for i, rec in enumerate(results["results"]["recommendations"][:5], 1):
            print(f"{i}. {rec}")
        
        # Feature importance
        feature_importance = results["results"]["feature_engineering"].get("feature_importance", {})
        if feature_importance:
            print("\n" + "-"*60)
            print("TOP FEATURE IMPORTANCE")
            print("-"*60)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"{i:2d}. {feature}: {importance:.4f}")
        
        # Cleanup
        pipeline.cleanup()
        
    finally:
        # Cleanup sample file
        if os.path.exists(data_path):
            os.remove(data_path)

def run_custom_configuration_example():
    """Demonstrate custom configuration usage"""
    print("\n" + "="*80)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Create custom configuration
    custom_config = {
        "project_name": "custom_ml_project",
        "data": {
            "use_big_data_engine": False,  # Disable for faster demo
            "chunk_size": 5000
        },
        "outlier_detection": {
            "methods": ["isolation_forest", "z_score"],
            "remove_outliers": False  # Keep outliers for analysis
        },
        "feature_engineering": {
            "scaling_method": "robust",
            "feature_selection": True,
            "max_features": 20
        },
        "model": {
            "automl_framework": None,  # Disable AutoML for faster execution
            "cross_validation_folds": 3
        },
        "logging": {
            "level": "DEBUG",
            "log_file": None  # Console only
        }
    }
    
    print("Custom Configuration:")
    print(f"- Big Data Engine: {custom_config['data']['use_big_data_engine']}")
    print(f"- Outlier Methods: {custom_config['outlier_detection']['methods']}")
    print(f"- Remove Outliers: {custom_config['outlier_detection']['remove_outliers']}")
    print(f"- Scaling Method: {custom_config['feature_engineering']['scaling_method']}")
    print(f"- Max Features: {custom_config['feature_engineering']['max_features']}")
    print(f"- AutoML Framework: {custom_config['model']['automl_framework']}")
    
    # Create sample data (smaller for demo)
    data = create_sample_cross_sectional_data().sample(n=1000, random_state=42)
    data_path = "sample_custom_data.csv"
    data.to_csv(data_path, index=False)
    
    try:
        # Initialize pipeline with custom configuration
        pipeline = MLPipeline(custom_config)
        
        # Run pipeline step by step to show individual components
        print("\nRunning pipeline step by step...")
        
        # Step 1: Load data
        print("1. Loading and profiling data...")
        pipeline.load_and_profile_data(data_path)
        
        # Step 2: Outlier detection
        print("2. Detecting outliers...")
        pipeline.detect_and_handle_outliers()
        
        # Step 3: Data type detection
        print("3. Detecting data type...")
        pipeline.detect_data_type("loan_default")
        
        # Step 4: Feature engineering
        print("4. Engineering features...")
        pipeline.engineer_features("loan_default")
        
        # Step 5: Model training
        print("5. Training models...")
        pipeline.select_and_train_models("loan_default")
        
        # Step 6: Generate insights
        print("6. Generating insights...")
        pipeline.generate_insights_and_recommendations()
        
        # Get results
        results = pipeline.get_pipeline_results()
        
        print("\n" + "-"*60)
        print("CUSTOM PIPELINE RESULTS")
        print("-"*60)
        print(f"Best Model: {results['results']['model_results']['best_model']['name']}")
        print(f"Score: {results['results']['model_results']['best_model']['score']:.4f}")
        print(f"Features Used: {results['summary']['final_features']}")
        
        # Cleanup
        pipeline.cleanup()
        
    finally:
        # Cleanup sample file
        if os.path.exists(data_path):
            os.remove(data_path)


def run_visualization_demo():
    """Demonstrate advanced visualization capabilities"""
    
    print("\n" + "="*50)
    print("Example 4: Advanced Visualization Demo")
    print("="*50)
    
    from src.ml_pipeline import AdvancedVisualizer
    import numpy as np
    import pandas as pd
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize standalone visualizer
    visualizer = AdvancedVisualizer(output_dir="visualizations")
    
    print("Generating sample visualizations...")
    
    # 1. Time Series Forecast Plot Demo
    print("  - Creating time series forecast plot...")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    actual_data = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5) + 5 * np.sin(np.arange(100) * 2 * np.pi / 20),
        index=dates,
        name='actual'
    )
    predictions = actual_data + np.random.randn(100) * 2
    
    # Future forecast
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=20, freq='D')
    forecast_horizon = pd.Series(
        actual_data.iloc[-1] + np.cumsum(np.random.randn(20) * 0.5),
        index=future_dates,
        name='forecast'
    )
    
    # Confidence intervals
    confidence_intervals = {
        'lower': forecast_horizon - 5,
        'upper': forecast_horizon + 5
    }
    
    forecast_plots = visualizer.create_time_series_forecast_plot(
        actual_data=actual_data,
        predictions=predictions,
        forecast_horizon=forecast_horizon,
        confidence_intervals=confidence_intervals,
        title="Sample Time Series Forecast with Confidence Intervals"
    )
    
    # 2. Feature Importance Plot Demo
    print("  - Creating feature importance plot...")
    feature_names = [f'Feature_{i}' for i in range(20)]
    importance_scores = np.random.exponential(scale=0.1, size=20)
    importance_scores = importance_scores / importance_scores.sum()  # Normalize
    
    importance_plots = visualizer.create_feature_importance_plot(
        feature_names=feature_names,
        importance_scores=importance_scores,
        model_name="Random Forest"
    )
    
    # 3. Model Comparison Plot Demo
    print("  - Creating model comparison plot...")
    model_results = {
        'Random Forest': {'score': 0.85},
        'Gradient Boosting': {'score': 0.82},
        'Linear Regression': {'score': 0.75},
        'SVM': {'score': 0.78},
        'Neural Network': {'score': 0.80}
    }
    
    comparison_plots = visualizer.create_model_comparison_plot(
        model_results=model_results,
        metric_name="score"
    )
    
    # 4. Residual Analysis Demo
    print("  - Creating residual analysis plot...")
    n_samples = 200
    actual = np.random.randn(n_samples) * 10 + 50
    predicted = actual + np.random.randn(n_samples) * 3  # Add some prediction error
    
    residual_plots = visualizer.create_residual_analysis_plot(
        actual=actual,
        predicted=predicted,
        model_name="Demo Model"
    )
    
    # 5. Data Distribution Plot Demo
    print("  - Creating data distribution plots...")
    sample_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.exponential(2, 1000),
        'feature_3': np.random.uniform(-5, 5, 1000),
        'feature_4': np.random.gamma(2, 2, 1000),
        'target': np.random.randn(1000)
    })
    
    distribution_plots = visualizer.create_data_distribution_plots(
        data=sample_data,
        target_column='target'
    )
    
    # 6. Outlier Visualization Demo
    print("  - Creating outlier visualization...")
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 500)
    outlier_data = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 20)
    all_data = np.vstack([normal_data, outlier_data])
    
    outlier_df = pd.DataFrame(all_data, columns=['x', 'y'])
    outlier_indices = list(range(500, 520))  # Last 20 points are outliers
    
    outlier_plots = visualizer.create_outlier_visualization(
        data=outlier_df,
        outlier_indices=outlier_indices,
        feature_columns=['x', 'y']
    )
    
    # 7. Create comprehensive dashboard
    print("  - Creating comprehensive dashboard...")
    mock_pipeline_results = {
        'insights': {
            'model_insights': {
                'best_model': {'name': 'Random Forest', 'score': 0.85}
            },
            'data_insights': {
                'data_quality': {
                    'total_rows': 1000,
                    'total_features': 5,
                    'missing_data_percentage': 2.5
                }
            }
        }
    }
    
    dashboard_path = visualizer.create_comprehensive_dashboard(mock_pipeline_results)
    
    print(f"✓ Visualization demo completed successfully!")
    print(f"  - Generated {len(visualizer.generated_plots)} plot categories")
    print(f"  - Dashboard saved to: {dashboard_path}")
    print(f"  - All visualizations saved in 'visualizations' directory")
    print(f"  - Both interactive (HTML) and static (PNG) versions created")
    
    # Get summary
    summary = visualizer.get_generated_plots_summary()
    print(f"  - Total plots generated: {summary['total_plots']}")
    print(f"  - Plot types: {', '.join(summary['plot_types'])}")


def main():
    """Main function to run all examples"""
    print("ML Pipeline - Complete Examples")
    print("="*80)
    print("This demo showcases the full capabilities of the ML pipeline.")
    print("The pipeline automatically handles:")
    print("- Large dataset loading and processing")
    print("- Outlier detection and removal")  
    print("- Time series vs cross-sectional detection")
    print("- Feature engineering and selection")
    print("- Model selection and training")
    print("- Insights generation and recommendations")
    print("="*80)
    
    # Create necessary directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # Example 1: Time Series Data
        run_time_series_example()
        
        # Example 2: Cross-sectional Data  
        run_cross_sectional_example()
        
        # Example 3: Custom Configuration
        run_custom_configuration_example()
        
        # Example 4: Advanced Visualization Demo
        run_visualization_demo()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext Steps:")
        print("1. Check the 'outputs' directory for saved artifacts")
        print("2. Check the 'visualizations' directory for generated plots and dashboards")
        print("3. Review the logs for detailed execution information")
        print("4. Customize the configuration file for your specific use case")
        print("5. Use the pipeline with your own data")
        print("\nGenerated Visualizations:")
        print("- Time series forecast plots with confidence intervals")
        print("- Residual analysis and diagnostic plots")
        print("- Feature importance visualizations")
        print("- Model performance comparison charts")
        print("- Data distribution and outlier detection plots")
        print("- Interactive HTML dashboards")
        print("\nFor production deployment:")
        print("- Set up MLflow for experiment tracking")
        print("- Configure database connections for large datasets")
        print("- Enable AutoML frameworks for better model selection")
        print("- Set up monitoring and retraining schedules")
        print("- Integrate visualization dashboards into your monitoring system")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Please check the requirements and ensure all dependencies are installed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())