#!/usr/bin/env python3
"""
Complete Data Analysis Example
Demonstrates the full data analysis pipeline using the custom toolkit.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from utils.data_loader import DataLoader, DataValidator
from analysis.data_profiler import DataProfiler, DataComparer
from visualization.advanced_plots import AdvancedVisualizer, InteractiveDashboard
from ml_models.model_pipeline import AutoMLPipeline, ModelInterpreter

def create_sample_dataset():
    """Create a comprehensive sample dataset for demonstration."""
    
    np.random.seed(42)
    
    # Generate 1000 samples
    n_samples = 1000
    
    # Create diverse feature types
    data = {
        # Numeric features
        'age': np.random.normal(35, 12, n_samples).astype(int).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.8, n_samples).astype(int),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int).clip(300, 850),
        'years_employed': np.random.exponential(5, n_samples).clip(0, 40),
        
        # Categorical features
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                    n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                          n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
                                n_samples, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
        
        # Date feature
        'application_date': [datetime.now() - timedelta(days=int(x)) 
                           for x in np.random.exponential(30, n_samples)],
        
        # Binary target variable (loan approval)
        'loan_approved': None  # Will be calculated based on features
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic relationships for target variable
    # Higher income, credit score, and education level increase approval probability
    approval_probability = (
        0.3 +  # Base probability
        (df['income'] / 100000) * 0.2 +  # Income effect
        ((df['credit_score'] - 300) / 550) * 0.3 +  # Credit score effect
        (df['education'].map({'High School': 0, 'Bachelor': 0.1, 'Master': 0.15, 'PhD': 0.2})) +
        (df['employment_type'].map({'Full-time': 0.1, 'Part-time': 0, 'Self-employed': 0.05, 'Unemployed': -0.3}))
    ).clip(0, 1)
    
    df['loan_approved'] = np.random.binomial(1, approval_probability, n_samples)
    
    # Introduce some missing values realistically
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'years_employed'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    return df

def main():
    """Main function demonstrating the complete data analysis workflow."""
    
    print("=" * 80)
    print("COMPREHENSIVE DATA ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ========================================
    # 1. DATA LOADING AND VALIDATION
    # ========================================
    print("1. DATA LOADING AND VALIDATION")
    print("-" * 40)
    
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Save and reload to demonstrate data loading
    df.to_csv('data/sample_loan_data.csv', index=False)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load the data
    df_loaded = loader.load_csv('data/sample_loan_data.csv')
    print(f"✓ Loaded dataset with shape: {df_loaded.shape}")
    
    # Validate the data
    validator = DataValidator()
    validation_results = validator.validate_dataframe(
        df_loaded, 
        required_columns=['age', 'income', 'loan_approved'],
        min_rows=500
    )
    
    if validation_results['is_valid']:
        print("✓ Data validation passed")
    else:
        print("⚠ Data validation issues found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    
    print(f"✓ Dataset memory usage: {validation_results['summary']['memory_usage_mb']:.2f} MB")
    print()
    
    # ========================================
    # 2. DATA PROFILING AND EXPLORATION
    # ========================================
    print("2. DATA PROFILING AND EXPLORATION")
    print("-" * 40)
    
    # Initialize data profiler
    profiler = DataProfiler(df_loaded)
    
    # Generate comprehensive profile
    print("Generating comprehensive data profile...")
    profile_report = profiler.generate_profile_report()
    
    print(f"✓ Analyzed {profile_report['dataset_info']['column_count']} columns")
    print(f"✓ Found {profile_report['dataset_info']['total_missing_values']} missing values "
          f"({profile_report['dataset_info']['missing_percentage']:.1f}%)")
    print(f"✓ Detected {profile_report['dataset_info']['duplicate_rows']} duplicate rows")
    
    # Show data quality issues
    issues = profile_report['data_quality_issues']
    if any(issues.values()):
        print("\n⚠ Data Quality Issues Found:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"  - {issue_type.replace('_', ' ').title()}: {len(issue_list)} items")
    
    # Generate and display summary report
    summary_report = profiler.generate_summary_report()
    print("\n" + "="*50)
    print("DATA PROFILE SUMMARY")
    print("="*50)
    print(summary_report[:1000] + "..." if len(summary_report) > 1000 else summary_report)
    print()
    
    # ========================================
    # 3. ADVANCED VISUALIZATIONS
    # ========================================
    print("3. ADVANCED VISUALIZATIONS")
    print("-" * 40)
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer(figsize=(12, 8))
    
    # Create output directory for plots
    os.makedirs('output/plots', exist_ok=True)
    
    print("Creating advanced visualizations...")
    
    # 1. Correlation heatmap
    fig1 = visualizer.correlation_heatmap(
        df_loaded, 
        save_path='output/plots/correlation_heatmap.png'
    )
    print("✓ Created correlation heatmap")
    plt.close(fig1)
    
    # 2. Distribution analysis
    numeric_cols = df_loaded.select_dtypes(include=[np.number]).columns.tolist()
    if 'loan_approved' in numeric_cols:
        numeric_cols.remove('loan_approved')  # Remove target from distribution analysis
    
    fig2 = visualizer.distribution_analysis(
        df_loaded, 
        columns=numeric_cols[:4],  # Limit to first 4 for readability
        save_path='output/plots/distribution_analysis.png'
    )
    print("✓ Created distribution analysis")
    plt.close(fig2)
    
    # 3. Outlier analysis
    fig3 = visualizer.outlier_analysis(
        df_loaded, 
        columns=['income', 'credit_score'],
        save_path='output/plots/outlier_analysis.png'
    )
    print("✓ Created outlier analysis")
    plt.close(fig3)
    
    # 4. Categorical analysis
    categorical_cols = ['education', 'employment_type', 'city']
    fig4 = visualizer.categorical_analysis(
        df_loaded, 
        cat_cols=categorical_cols,
        target_col='loan_approved',
        save_path='output/plots/categorical_analysis.png'
    )
    print("✓ Created categorical analysis")
    plt.close(fig4)
    
    # 5. Missing data visualization
    fig5 = visualizer.missing_data_visualization(
        df_loaded, 
        save_path='output/plots/missing_data_analysis.png'
    )
    print("✓ Created missing data visualization")
    plt.close(fig5)
    
    # 6. Interactive dashboard
    dashboard = InteractiveDashboard()
    interactive_fig = dashboard.create_overview_dashboard(df_loaded)
    interactive_fig.write_html('output/plots/interactive_dashboard.html')
    print("✓ Created interactive dashboard (saved as HTML)")
    
    print()
    
    # ========================================
    # 4. MACHINE LEARNING PIPELINE
    # ========================================
    print("4. MACHINE LEARNING PIPELINE")
    print("-" * 40)
    
    # Prepare data for ML
    feature_columns = ['age', 'income', 'credit_score', 'years_employed', 
                      'education', 'employment_type', 'city']
    target_column = 'loan_approved'
    
    # Remove rows with missing target
    ml_df = df_loaded.dropna(subset=[target_column]).copy()
    
    X = ml_df[feature_columns]
    y = ml_df[target_column]
    
    print(f"Training models on {len(X)} samples with {len(feature_columns)} features...")
    
    # Initialize AutoML pipeline
    automl = AutoMLPipeline(
        problem_type='auto',  # Will auto-detect classification
        cv_folds=5,
        test_size=0.2,
        random_state=42
    )
    
    # Train models
    results = automl.fit(X, y, hyperparameter_tuning=True)
    
    print(f"✓ Trained {len(results['models'])} models")
    print(f"✓ Best model: {results['best_model_name']} (Score: {results['best_score']:.4f})")
    print(f"✓ Training time: {results['training_time']:.2f} seconds")
    
    # Get model comparison
    comparison_df = automl.get_model_comparison()
    print("\nModel Performance Comparison:")
    print(comparison_df.round(4))
    
    # Get feature importance
    if results['best_model_name'] in results['feature_importance']:
        importance_df = automl.get_feature_importance(top_n=10)
        print(f"\nTop Features ({results['best_model_name']}):")
        print(importance_df.round(4))
    
    # Save the model
    automl.save_model('output/best_model.joblib')
    print("✓ Model saved")
    
    # Generate comprehensive report
    ml_report = automl.generate_report()
    with open('output/ml_model_report.txt', 'w') as f:
        f.write(ml_report)
    print("✓ ML report saved")
    
    print()
    
    # ========================================
    # 5. MODEL INTERPRETATION
    # ========================================
    print("5. MODEL INTERPRETATION")
    print("-" * 40)
    
    # Get training data for SHAP background
    X_train = X.iloc[:800]  # Use first 800 samples as training background
    X_test = X.iloc[800:810]  # Use 10 samples for explanation
    
    try:
        # Initialize model interpreter
        interpreter = ModelInterpreter(
            model=automl.best_model,
            X_train=X_train,
            feature_names=feature_columns
        )
        
        print("Generating model explanations...")
        
        # Explain a single prediction
        single_explanation = interpreter.explain_prediction(X_test.iloc[:1], plot=False)
        
        if 'error' not in single_explanation:
            print("✓ Generated single prediction explanation")
            print(f"  Prediction: {single_explanation['prediction']}")
            print("  Top feature contributions:")
            sorted_contributions = sorted(
                single_explanation['feature_contributions'].items(),
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            for feature, contribution in sorted_contributions:
                print(f"    {feature}: {contribution:.4f}")
        
        # Global feature importance
        global_importance = interpreter.global_feature_importance(X_test, plot=False)
        
        if 'error' not in global_importance:
            print("✓ Generated global feature importance")
            print("  Top global features:")
            for feature, importance in global_importance['sorted_importance'][:5]:
                print(f"    {feature}: {importance:.4f}")
    
    except Exception as e:
        print(f"⚠ Model interpretation failed: {e}")
        print("  This may be due to missing SHAP package or model compatibility")
    
    print()
    
    # ========================================
    # 6. COMPARATIVE ANALYSIS
    # ========================================
    print("6. COMPARATIVE ANALYSIS")
    print("-" * 40)
    
    # Create a modified dataset for comparison
    df_modified = df_loaded.copy()
    
    # Simulate some changes (e.g., economic improvement)
    mask = df_modified['income'] < 50000
    df_modified.loc[mask, 'income'] *= 1.1  # 10% income increase for lower income
    df_modified.loc[mask, 'credit_score'] += 20  # Credit score improvement
    
    # Compare datasets
    comparer = DataComparer()
    comparison = comparer.compare_dataframes(
        df_loaded, df_modified,
        name1="Original Dataset", 
        name2="Modified Dataset"
    )
    
    print("✓ Dataset comparison completed")
    print(f"  Common columns: {len(comparison['column_comparison']['common_columns'])}")
    print(f"  Shape match: {comparison['basic_comparison']['shape_match']}")
    
    # Show statistical differences for key numeric columns
    if comparison['statistical_comparison']:
        print("  Statistical differences detected in:")
        for col, stats in comparison['statistical_comparison'].items():
            orig_mean = stats.get('Original Dataset_mean', 0)
            mod_mean = stats.get('Modified Dataset_mean', 0)
            if orig_mean and mod_mean:
                pct_change = ((mod_mean - orig_mean) / orig_mean) * 100
                if abs(pct_change) > 1:  # Show only significant changes
                    print(f"    {col}: {pct_change:+.2f}% change in mean")
    
    print()
    
    # ========================================
    # 7. GENERATE FINAL REPORTS
    # ========================================
    print("7. GENERATING FINAL REPORTS")
    print("-" * 40)
    
    # Create comprehensive analysis report
    report_content = f"""
DATA ANALYSIS REPORT
====================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
=================
This report presents a comprehensive analysis of the loan approval dataset, 
including data quality assessment, exploratory analysis, predictive modeling, 
and model interpretation.

DATASET OVERVIEW
================
• Total Records: {len(df_loaded):,}
• Features: {len(feature_columns)}
• Target Variable: {target_column}
• Missing Values: {profile_report['dataset_info']['total_missing_values']} ({profile_report['dataset_info']['missing_percentage']:.1f}%)
• Duplicate Records: {profile_report['dataset_info']['duplicate_rows']}

KEY FINDINGS
============
1. Data Quality: {len([i for issues in profile_report['data_quality_issues'].values() for i in issues])} potential issues identified
2. Target Distribution: {(df_loaded[target_column].value_counts().to_dict())}
3. Best Predictive Model: {results['best_model_name']} (Accuracy: {results['best_score']:.4f})
4. Model Training Time: {results['training_time']:.2f} seconds

RECOMMENDATIONS
===============
1. Address data quality issues identified in the analysis
2. Consider feature engineering for improved model performance
3. Monitor model performance over time for drift detection
4. Implement the {results['best_model_name']} model for production use

TECHNICAL DETAILS
=================
{ml_report}

FILES GENERATED
===============
• Correlation Heatmap: output/plots/correlation_heatmap.png
• Distribution Analysis: output/plots/distribution_analysis.png
• Outlier Analysis: output/plots/outlier_analysis.png
• Categorical Analysis: output/plots/categorical_analysis.png
• Missing Data Analysis: output/plots/missing_data_analysis.png
• Interactive Dashboard: output/plots/interactive_dashboard.html
• Best Model: output/best_model.joblib
• ML Model Report: output/ml_model_report.txt
"""
    
    # Save the comprehensive report
    with open('output/comprehensive_analysis_report.txt', 'w') as f:
        f.write(report_content)
    
    print("✓ Comprehensive analysis report saved")
    print("✓ All visualizations saved to output/plots/")
    print("✓ Model and reports saved to output/")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Summary of outputs:")
    print("• Comprehensive report: output/comprehensive_analysis_report.txt")
    print("• Visualizations: output/plots/ (6 files)")
    print("• Trained model: output/best_model.joblib")
    print("• ML report: output/ml_model_report.txt")
    print()
    print("This demonstrates a complete end-to-end data analysis workflow")
    print("suitable for production data science projects.")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    
    # Run the main analysis
    main()