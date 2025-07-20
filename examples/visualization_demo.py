#!/usr/bin/env python3
"""
Standalone Visualization Demo

This script demonstrates how to use the AdvancedVisualizer independently
to create professional data science visualizations including:
- Time series forecasts with confidence intervals
- Residual analysis and model diagnostics
- Feature importance charts
- Model performance comparisons
- Data distribution plots
- Outlier detection visualizations
- Interactive HTML dashboards
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_pipeline.utils.visualizer import AdvancedVisualizer


def create_sample_time_series_data():
    """Create sample time series data for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    
    # Generate time series with trend, seasonality, and noise
    trend = np.linspace(100, 150, 150)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(150) / 30)  # 30-day seasonality
    noise = np.random.normal(0, 3, 150)
    actual_values = trend + seasonal + noise
    
    # Create predictions (with some error)
    prediction_error = np.random.normal(0, 2, 150)
    predicted_values = actual_values + prediction_error
    
    # Create future forecast
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    future_trend = np.linspace(150, 160, 30)
    future_seasonal = 10 * np.sin(2 * np.pi * np.arange(150, 180) / 30)
    future_noise = np.random.normal(0, 3, 30)
    future_values = future_trend + future_seasonal + future_noise
    
    return {
        'actual': pd.Series(actual_values, index=dates, name='actual'),
        'predictions': pd.Series(predicted_values, index=dates, name='predictions'),
        'forecast': pd.Series(future_values, index=future_dates, name='forecast'),
        'forecast_lower': pd.Series(future_values - 5, index=future_dates, name='lower_ci'),
        'forecast_upper': pd.Series(future_values + 5, index=future_dates, name='upper_ci')
    }


def create_sample_ml_data():
    """Create sample ML data for demonstration"""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 15
    
    # Generate feature data
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features
    true_importance = np.random.exponential(1, n_features)
    true_importance = true_importance / true_importance.sum()
    y_true = X @ true_importance + np.random.normal(0, 0.1, n_samples)
    
    # Create predictions with some error
    y_pred = y_true + np.random.normal(0, 0.2, n_samples)
    
    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y_true
    
    return {
        'data': data,
        'y_true': y_true,
        'y_pred': y_pred,
        'feature_names': feature_names,
        'feature_importance': true_importance
    }


def create_sample_outlier_data():
    """Create sample data with outliers for demonstration"""
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.multivariate_normal([2, 3], [[1, 0.5], [0.5, 1]], 800)
    
    # Outliers
    outlier_data = np.random.multivariate_normal([8, 8], [[0.5, 0], [0, 0.5]], 50)
    
    # Combine data
    all_data = np.vstack([normal_data, outlier_data])
    outlier_indices = list(range(800, 850))  # Indices of outliers
    
    data = pd.DataFrame(all_data, columns=['feature_x', 'feature_y'])
    
    return {
        'data': data,
        'outlier_indices': outlier_indices
    }


def demo_time_series_visualization(visualizer):
    """Demonstrate time series visualization capabilities"""
    print("\n1. Time Series Forecast Visualization")
    print("-" * 50)
    
    # Create sample time series data
    ts_data = create_sample_time_series_data()
    
    # Create confidence intervals
    confidence_intervals = {
        'lower': ts_data['forecast_lower'],
        'upper': ts_data['forecast_upper']
    }
    
    # Generate time series plot
    forecast_plots = visualizer.create_time_series_forecast_plot(
        actual_data=ts_data['actual'],
        predictions=ts_data['predictions'],
        forecast_horizon=ts_data['forecast'],
        confidence_intervals=confidence_intervals,
        title="Sales Forecast with Confidence Intervals"
    )
    
    print("✓ Time series forecast plot created")
    print(f"  - Historical data: {len(ts_data['actual'])} points")
    print(f"  - Forecast horizon: {len(ts_data['forecast'])} points")
    print(f"  - Confidence intervals included")
    
    return forecast_plots


def demo_residual_analysis(visualizer):
    """Demonstrate residual analysis visualization"""
    print("\n2. Residual Analysis Visualization")
    print("-" * 50)
    
    # Create sample ML data
    ml_data = create_sample_ml_data()
    
    # Generate residual analysis
    residual_results = visualizer.create_residual_analysis_plot(
        actual=ml_data['y_true'],
        predicted=ml_data['y_pred'],
        model_name="Random Forest Regressor"
    )
    
    print("✓ Residual analysis plots created")
    if 'statistics' in residual_results:
        stats = residual_results['statistics']
        print(f"  - Residual mean: {stats['mean']:.4f}")
        print(f"  - Residual std: {stats['std']:.4f}")
        print(f"  - Skewness: {stats['skewness']:.4f}")
        print(f"  - Durbin-Watson: {stats['durbin_watson']:.4f}")
    
    return residual_results


def demo_feature_importance(visualizer):
    """Demonstrate feature importance visualization"""
    print("\n3. Feature Importance Visualization")
    print("-" * 50)
    
    # Create sample ML data
    ml_data = create_sample_ml_data()
    
    # Generate feature importance plot
    importance_plots = visualizer.create_feature_importance_plot(
        feature_names=ml_data['feature_names'],
        importance_scores=ml_data['feature_importance'],
        model_name="Random Forest",
        top_n=10
    )
    
    print("✓ Feature importance plot created")
    print(f"  - Total features: {len(ml_data['feature_names'])}")
    print(f"  - Top features displayed: 10")
    print(f"  - Most important: {ml_data['feature_names'][np.argmax(ml_data['feature_importance'])]}")
    
    return importance_plots


def demo_model_comparison(visualizer):
    """Demonstrate model comparison visualization"""
    print("\n4. Model Performance Comparison")
    print("-" * 50)
    
    # Create sample model results
    model_results = {
        'Random Forest': {'score': 0.892},
        'Gradient Boosting': {'score': 0.885},
        'XGBoost': {'score': 0.878},
        'Linear Regression': {'score': 0.753},
        'SVM': {'score': 0.821},
        'Neural Network': {'score': 0.863},
        'KNN': {'score': 0.789},
        'Decision Tree': {'score': 0.734}
    }
    
    # Generate model comparison plot
    comparison_plots = visualizer.create_model_comparison_plot(
        model_results=model_results,
        metric_name="score"
    )
    
    print("✓ Model comparison plot created")
    print(f"  - Models compared: {len(model_results)}")
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['score'])
    print(f"  - Best model: {best_model} ({model_results[best_model]['score']:.3f})")
    
    return comparison_plots


def demo_data_distribution(visualizer):
    """Demonstrate data distribution visualization"""
    print("\n5. Data Distribution Visualization")
    print("-" * 50)
    
    # Create sample ML data
    ml_data = create_sample_ml_data()
    
    # Generate distribution plots
    distribution_plots = visualizer.create_data_distribution_plots(
        data=ml_data['data'],
        target_column='target',
        max_features=8
    )
    
    print("✓ Data distribution plots created")
    print(f"  - Features plotted: 8")
    print(f"  - Target variable: target")
    print(f"  - Distribution types: histograms with KDE")
    
    return distribution_plots


def demo_outlier_visualization(visualizer):
    """Demonstrate outlier detection visualization"""
    print("\n6. Outlier Detection Visualization")
    print("-" * 50)
    
    # Create sample outlier data
    outlier_data = create_sample_outlier_data()
    
    # Generate outlier visualization
    outlier_plots = visualizer.create_outlier_visualization(
        data=outlier_data['data'],
        outlier_indices=outlier_data['outlier_indices'],
        feature_columns=['feature_x', 'feature_y']
    )
    
    print("✓ Outlier visualization created")
    print(f"  - Total data points: {len(outlier_data['data'])}")
    print(f"  - Outliers detected: {len(outlier_data['outlier_indices'])}")
    print(f"  - Outlier percentage: {len(outlier_data['outlier_indices'])/len(outlier_data['data'])*100:.1f}%")
    
    return outlier_plots


def demo_comprehensive_dashboard(visualizer):
    """Demonstrate comprehensive dashboard creation"""
    print("\n7. Comprehensive Dashboard Creation")
    print("-" * 50)
    
    # Create mock pipeline results
    mock_results = {
        'insights': {
            'model_insights': {
                'best_model': {
                    'name': 'Random Forest',
                    'score': 0.892,
                    'algorithm_type': 'ensemble'
                },
                'model_comparison': {
                    'models_evaluated': 8,
                    'score_range': {'min_score': 0.734, 'max_score': 0.892, 'score_spread': 0.158}
                }
            },
            'data_insights': {
                'data_quality': {
                    'total_rows': 1000,
                    'total_features': 15,
                    'missing_data_percentage': 2.3,
                    'duplicate_rows': 5
                },
                'outlier_impact': {
                    'outliers_detected': 50,
                    'outlier_percentage': 5.0,
                    'severity': 'low'
                },
                'temporal_characteristics': {
                    'data_type': 'time_series',
                    'confidence': 0.85,
                    'has_seasonality': True,
                    'has_trend': True
                }
            },
            'business_insights': {
                'predictive_power': 'high',
                'deployment_readiness': 'production_ready'
            },
            'performance_insights': {
                'pipeline_performance': {
                    'total_execution_time': 45.6,
                    'data_size_mb': 12.5
                }
            }
        },
        'recommendations': [
            'Model performance is excellent (>85%). Consider deploying to production.',
            'Low outlier percentage detected. Data quality is good.',
            'Strong seasonality detected. Consider seasonal adjustments in forecasting.',
            'Feature engineering was successful. 15 features created optimal performance.'
        ]
    }
    
    # Generate comprehensive dashboard
    dashboard_path = visualizer.create_comprehensive_dashboard(mock_results)
    
    print("✓ Comprehensive dashboard created")
    print(f"  - Dashboard saved to: {dashboard_path}")
    print(f"  - Includes all generated visualizations")
    print(f"  - Interactive HTML format")
    print(f"  - Performance metrics included")
    
    return dashboard_path


def main():
    """Main demonstration function"""
    print("Advanced ML Visualization Demo")
    print("=" * 60)
    print("This demo showcases comprehensive visualization capabilities")
    print("for machine learning pipelines and data analysis.")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("demo_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    print(f"\nInitializing Advanced Visualizer...")
    print(f"Output directory: {output_dir.absolute()}")
    
    visualizer = AdvancedVisualizer(
        output_dir=str(output_dir),
        style_theme="plotly_white",
        color_palette="viridis"
    )
    
    print("✓ Visualizer initialized successfully")
    
    # Run all demonstrations
    demos = [
        demo_time_series_visualization,
        demo_residual_analysis,
        demo_feature_importance,
        demo_model_comparison,
        demo_data_distribution,
        demo_outlier_visualization,
        demo_comprehensive_dashboard
    ]
    
    results = {}
    for demo_func in demos:
        try:
            result = demo_func(visualizer)
            results[demo_func.__name__] = result
        except Exception as e:
            print(f"✗ Error in {demo_func.__name__}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMO COMPLETED")
    print("=" * 60)
    
    summary = visualizer.get_generated_plots_summary()
    print(f"✓ Total plot categories: {summary['total_plots']}")
    print(f"✓ Plot types: {', '.join(summary['plot_types'])}")
    print(f"✓ Output directory: {summary['output_directory']}")
    
    print(f"\nGenerated Files:")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        for file in sorted(files):
            print(f"  - {file.name}")
    
    print(f"\nNext Steps:")
    print(f"1. Open the generated HTML files in your browser for interactive plots")
    print(f"2. Check PNG files for high-quality static visualizations")
    print(f"3. Use the dashboard HTML for comprehensive analysis overview")
    print(f"4. Integrate these visualizations into your ML pipeline")
    
    return 0


if __name__ == "__main__":
    exit(main())