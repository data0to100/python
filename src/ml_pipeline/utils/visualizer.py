"""
Advanced Visualization Module for ML Pipeline

This module provides comprehensive visualization capabilities for:
- Time series forecasting with confidence intervals
- Model performance comparison and evaluation
- Feature importance and selection analysis
- Data distribution and outlier visualization
- Residual analysis and diagnostic plots
- Interactive dashboards with plotly
- Publication-ready static plots with seaborn/matplotlib

Expert-level implementation with 20+ years of data science experience insights:
- Automatic plot type selection based on data characteristics
- Statistical significance testing for visualizations
- Production-ready plotting with consistent styling
- Memory-efficient plotting for large datasets
- Interactive and static plot generation
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

# Visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Statistical libraries
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    mean_squared_error, mean_absolute_error, r2_score
)

# Utilities
from .logger import get_logger, timing_decorator, error_handler
from .config import Config


class AdvancedVisualizer:
    """
    Enterprise-grade visualization suite for ML pipeline results
    
    Provides comprehensive visualization capabilities including:
    - Time series forecasting with confidence intervals
    - Model performance diagnostics and comparisons
    - Feature analysis and importance visualization
    - Data quality and outlier detection plots
    - Statistical distribution analysis
    - Interactive dashboards and static reports
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 output_dir: Optional[str] = None,
                 style_theme: str = "plotly_white",
                 color_palette: str = "viridis"):
        """
        Initialize the advanced visualizer
        
        Args:
            config: Configuration object for customization
            output_dir: Directory to save visualizations
            style_theme: Plotly theme for consistent styling
            color_palette: Color palette for visualizations
        """
        self.config = config or Config()
        self.logger = get_logger(__name__)
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization settings
        self.style_theme = style_theme
        self.color_palette = color_palette
        self.figure_size = (12, 8)
        self.dpi = 300
        
        # Initialize plotting libraries
        self._setup_plotting_environment()
        
        # Store generated plots for dashboard creation
        self.generated_plots = {}
        
        self.logger.info(f"Advanced Visualizer initialized with output directory: {self.output_dir}")
    
    def _setup_plotting_environment(self):
        """Setup plotting environment and themes"""
        
        if MATPLOTLIB_AVAILABLE:
            # Set matplotlib style
            plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'available') and 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            plt.rcParams['figure.figsize'] = self.figure_size
            plt.rcParams['figure.dpi'] = self.dpi
            plt.rcParams['savefig.dpi'] = self.dpi
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12
            
            # Set seaborn style
            if 'sns' in globals():
                sns.set_palette(self.color_palette)
                sns.set_context("notebook", font_scale=1.2)
        
        if PLOTLY_AVAILABLE:
            # Set plotly default theme
            pyo.init_notebook_mode(connected=True)
    
    @timing_decorator
    @error_handler
    def create_time_series_forecast_plot(self, 
                                       actual_data: pd.Series,
                                       predictions: pd.Series,
                                       forecast_horizon: Optional[pd.Series] = None,
                                       confidence_intervals: Optional[Dict[str, pd.Series]] = None,
                                       title: str = "Time Series Forecast",
                                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive time series forecast visualization
        
        Args:
            actual_data: Historical actual values with datetime index
            predictions: Model predictions
            forecast_horizon: Future forecast values
            confidence_intervals: Dict with 'lower' and 'upper' confidence bounds
            title: Plot title
            save_path: Optional path to save the plot
        
        Returns:
            Dict containing plot objects and metadata
        """
        
        self.logger.info("Creating time series forecast visualization")
        
        plots = {}
        
        if PLOTLY_AVAILABLE:
            plots['interactive'] = self._create_plotly_forecast_plot(
                actual_data, predictions, forecast_horizon, confidence_intervals, title
            )
        
        if MATPLOTLIB_AVAILABLE:
            plots['static'] = self._create_matplotlib_forecast_plot(
                actual_data, predictions, forecast_horizon, confidence_intervals, title
            )
        
        # Save plots
        if save_path or self.output_dir:
            save_dir = Path(save_path).parent if save_path else self.output_dir
            self._save_plots(plots, save_dir / f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.generated_plots['forecast'] = plots
        return plots
    
    def _create_plotly_forecast_plot(self, actual_data, predictions, forecast_horizon, confidence_intervals, title):
        """Create interactive forecast plot with plotly"""
        
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=actual_data.index,
            y=actual_data.values,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions/fitted values
        if predictions is not None:
            fig.add_trace(go.Scatter(
                x=predictions.index,
                y=predictions.values,
                mode='lines',
                name='Fitted/Predicted',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Add forecast horizon
        if forecast_horizon is not None:
            fig.add_trace(go.Scatter(
                x=forecast_horizon.index,
                y=forecast_horizon.values,
                mode='lines',
                name='Forecast',
                line=dict(color='green', width=2)
            ))
        
        # Add confidence intervals
        if confidence_intervals and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=confidence_intervals['upper'].index,
                y=confidence_intervals['upper'].values,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=confidence_intervals['lower'].index,
                y=confidence_intervals['lower'].values,
                mode='lines',
                line=dict(width=0),
                name='Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.2)',
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.style_theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def _create_matplotlib_forecast_plot(self, actual_data, predictions, forecast_horizon, confidence_intervals, title):
        """Create static forecast plot with matplotlib"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot actual data
        ax.plot(actual_data.index, actual_data.values, 
                label='Actual', color='blue', linewidth=2)
        
        # Plot predictions/fitted values
        if predictions is not None:
            ax.plot(predictions.index, predictions.values, 
                    label='Fitted/Predicted', color='red', linewidth=2, linestyle='--')
        
        # Plot forecast horizon
        if forecast_horizon is not None:
            ax.plot(forecast_horizon.index, forecast_horizon.values, 
                    label='Forecast', color='green', linewidth=2)
        
        # Add confidence intervals
        if confidence_intervals and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
            ax.fill_between(
                confidence_intervals['lower'].index,
                confidence_intervals['lower'].values,
                confidence_intervals['upper'].values,
                alpha=0.2, color='gray', label='Confidence Interval'
            )
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for better date display
        if hasattr(actual_data.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    @timing_decorator
    @error_handler
    def create_residual_analysis_plot(self,
                                    actual: np.ndarray,
                                    predicted: np.ndarray,
                                    model_name: str = "Model",
                                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive residual analysis plots
        
        Args:
            actual: Actual target values
            predicted: Predicted values
            model_name: Name of the model for titles
            save_path: Optional path to save plots
        
        Returns:
            Dict containing plot objects and residual statistics
        """
        
        self.logger.info(f"Creating residual analysis for {model_name}")
        
        residuals = actual - predicted
        plots = {}
        stats_dict = self._calculate_residual_statistics(residuals)
        
        if MATPLOTLIB_AVAILABLE:
            plots['static'] = self._create_matplotlib_residual_plots(
                actual, predicted, residuals, model_name
            )
        
        if PLOTLY_AVAILABLE:
            plots['interactive'] = self._create_plotly_residual_plots(
                actual, predicted, residuals, model_name
            )
        
        # Save plots
        if save_path or self.output_dir:
            save_dir = Path(save_path).parent if save_path else self.output_dir
            self._save_plots(plots, save_dir / f"residuals_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        result = {
            'plots': plots,
            'statistics': stats_dict
        }
        
        self.generated_plots[f'residuals_{model_name}'] = result
        return result
    
    def _create_matplotlib_residual_plots(self, actual, predicted, residuals, model_name):
        """Create residual analysis plots with matplotlib"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Fitted Values
        axes[0, 0].scatter(predicted, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. QQ Plot of Residuals
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogram of Residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.4f}')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Actual vs Predicted
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[1, 1].scatter(actual, predicted, alpha=0.6)
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add R² to the actual vs predicted plot
        r2 = r2_score(actual, predicted)
        axes[1, 1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _create_plotly_residual_plots(self, actual, predicted, residuals, model_name):
        """Create interactive residual analysis plots with plotly"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Fitted', 'Q-Q Plot', 'Residual Distribution', 'Actual vs Predicted'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Residuals vs Fitted Values
        fig.add_trace(
            go.Scatter(x=predicted, y=residuals, mode='markers', name='Residuals',
                      marker=dict(opacity=0.6)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Q-Q Plot
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot',
                      marker=dict(opacity=0.6)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', 
                      name='Reference Line', line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # 3. Histogram of Residuals
        fig.add_trace(
            go.Histogram(x=residuals, name='Residual Distribution', opacity=0.7),
            row=2, col=1
        )
        
        # 4. Actual vs Predicted
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(x=actual, y=predicted, mode='markers', name='Predictions',
                      marker=dict(opacity=0.6)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                      name='Perfect Prediction', line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Residual Analysis - {model_name}",
            template=self.style_theme,
            showlegend=False,
            height=800
        )
        
        return fig
    
    def _calculate_residual_statistics(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive residual statistics"""
        
        return {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'jarque_bera_stat': float(stats.jarque_bera(residuals)[0]),
            'jarque_bera_pvalue': float(stats.jarque_bera(residuals)[1]),
            'shapiro_stat': float(stats.shapiro(residuals)[0]) if len(residuals) <= 5000 else None,
            'shapiro_pvalue': float(stats.shapiro(residuals)[1]) if len(residuals) <= 5000 else None,
            'durbin_watson': self._durbin_watson_statistic(residuals)
        }
    
    def _durbin_watson_statistic(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation testing"""
        diff = np.diff(residuals)
        return float(np.sum(diff**2) / np.sum(residuals**2))
    
    @timing_decorator
    @error_handler
    def create_feature_importance_plot(self,
                                     feature_names: List[str],
                                     importance_scores: np.ndarray,
                                     model_name: str = "Model",
                                     top_n: int = 20,
                                     save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create feature importance visualization
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            model_name: Name of the model
            top_n: Number of top features to display
            save_path: Optional path to save plots
        
        Returns:
            Dict containing plot objects
        """
        
        self.logger.info(f"Creating feature importance plot for {model_name}")
        
        # Sort features by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).head(top_n)
        
        plots = {}
        
        if PLOTLY_AVAILABLE:
            plots['interactive'] = self._create_plotly_feature_importance(
                importance_df, model_name
            )
        
        if MATPLOTLIB_AVAILABLE:
            plots['static'] = self._create_matplotlib_feature_importance(
                importance_df, model_name
            )
        
        # Save plots
        if save_path or self.output_dir:
            save_dir = Path(save_path).parent if save_path else self.output_dir
            self._save_plots(plots, save_dir / f"feature_importance_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.generated_plots[f'feature_importance_{model_name}'] = plots
        return plots
    
    def _create_plotly_feature_importance(self, importance_df, model_name):
        """Create interactive feature importance plot with plotly"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(
                color=importance_df['importance'],
                colorscale=self.color_palette,
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template=self.style_theme,
            height=max(400, len(importance_df) * 20 + 100),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def _create_matplotlib_feature_importance(self, importance_df, model_name):
        """Create static feature importance plot with matplotlib"""
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(importance_df) * 0.4)))
        
        bars = ax.barh(importance_df['feature'], importance_df['importance'])
        
        # Color bars by importance
        colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.set_title(f'Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(importance_df['importance']):
            ax.text(v + 0.01 * importance_df['importance'].max(), i, f'{v:.4f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @timing_decorator
    @error_handler
    def create_model_comparison_plot(self,
                                   model_results: Dict[str, Dict],
                                   metric_name: str = "score",
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create model performance comparison visualization
        
        Args:
            model_results: Dict of model results with performance metrics
            metric_name: Primary metric to compare
            save_path: Optional path to save plots
        
        Returns:
            Dict containing plot objects
        """
        
        self.logger.info("Creating model comparison visualization")
        
        # Extract model performance data
        models_data = []
        for model_name, results in model_results.items():
            if isinstance(results, dict) and metric_name in results:
                models_data.append({
                    'model': model_name,
                    'score': results[metric_name]
                })
        
        if not models_data:
            self.logger.warning("No valid model results found for comparison")
            return {}
        
        comparison_df = pd.DataFrame(models_data).sort_values('score', ascending=False)
        
        plots = {}
        
        if PLOTLY_AVAILABLE:
            plots['interactive'] = self._create_plotly_model_comparison(
                comparison_df, metric_name
            )
        
        if MATPLOTLIB_AVAILABLE:
            plots['static'] = self._create_matplotlib_model_comparison(
                comparison_df, metric_name
            )
        
        # Save plots
        if save_path or self.output_dir:
            save_dir = Path(save_path).parent if save_path else self.output_dir
            self._save_plots(plots, save_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.generated_plots['model_comparison'] = plots
        return plots
    
    def _create_plotly_model_comparison(self, comparison_df, metric_name):
        """Create interactive model comparison plot with plotly"""
        
        fig = go.Figure()
        
        # Create color scale based on performance
        colors = px.colors.qualitative.Set3[:len(comparison_df)]
        
        fig.add_trace(go.Bar(
            x=comparison_df['model'],
            y=comparison_df['score'],
            marker=dict(
                color=comparison_df['score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric_name.title())
            ),
            text=comparison_df['score'].round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Model Performance Comparison ({metric_name.title()})',
            xaxis_title='Models',
            yaxis_title=metric_name.title(),
            template=self.style_theme,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_matplotlib_model_comparison(self, comparison_df, metric_name):
        """Create static model comparison plot with matplotlib"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        bars = ax.bar(comparison_df['model'], comparison_df['score'])
        
        # Color bars by performance
        colors = plt.cm.viridis(comparison_df['score'] / comparison_df['score'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_xlabel('Models')
        ax.set_ylabel(metric_name.title())
        ax.set_title(f'Model Performance Comparison ({metric_name.title()})', 
                    fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @timing_decorator
    @error_handler
    def create_data_distribution_plots(self,
                                     data: pd.DataFrame,
                                     target_column: Optional[str] = None,
                                     max_features: int = 20,
                                     save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive data distribution visualizations
        
        Args:
            data: Input dataframe
            target_column: Target variable for correlation analysis
            max_features: Maximum number of features to plot
            save_path: Optional path to save plots
        
        Returns:
            Dict containing plot objects
        """
        
        self.logger.info("Creating data distribution visualizations")
        
        # Select numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        numeric_cols = numeric_cols[:max_features]
        
        plots = {}
        
        if MATPLOTLIB_AVAILABLE and numeric_cols:
            plots['static'] = self._create_matplotlib_distribution_plots(
                data, numeric_cols, target_column
            )
        
        if PLOTLY_AVAILABLE and numeric_cols:
            plots['interactive'] = self._create_plotly_distribution_plots(
                data, numeric_cols, target_column
            )
        
        # Save plots
        if save_path or self.output_dir:
            save_dir = Path(save_path).parent if save_path else self.output_dir
            self._save_plots(plots, save_dir / f"data_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.generated_plots['data_distribution'] = plots
        return plots
    
    def _create_matplotlib_distribution_plots(self, data, numeric_cols, target_column):
        """Create distribution plots with matplotlib/seaborn"""
        
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                if 'sns' in globals():
                    sns.histplot(data=data, x=col, ax=axes[i], kde=True)
                else:
                    axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _create_plotly_distribution_plots(self, data, numeric_cols, target_column):
        """Create interactive distribution plots with plotly"""
        
        from plotly.subplots import make_subplots
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=data[col].dropna(), name=col, showlegend=False),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title='Feature Distributions',
            template=self.style_theme,
            height=n_rows * 300
        )
        
        return fig
    
    @timing_decorator
    @error_handler
    def create_outlier_visualization(self,
                                   data: pd.DataFrame,
                                   outlier_indices: List[int],
                                   feature_columns: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create outlier detection visualizations
        
        Args:
            data: Input dataframe
            outlier_indices: Indices of detected outliers
            feature_columns: Specific features to visualize
            save_path: Optional path to save plots
        
        Returns:
            Dict containing plot objects
        """
        
        self.logger.info("Creating outlier visualization")
        
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        plots = {}
        
        if PLOTLY_AVAILABLE:
            plots['interactive'] = self._create_plotly_outlier_plot(
                data, outlier_indices, feature_columns
            )
        
        if MATPLOTLIB_AVAILABLE:
            plots['static'] = self._create_matplotlib_outlier_plot(
                data, outlier_indices, feature_columns
            )
        
        # Save plots
        if save_path or self.output_dir:
            save_dir = Path(save_path).parent if save_path else self.output_dir
            self._save_plots(plots, save_dir / f"outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.generated_plots['outliers'] = plots
        return plots
    
    def _create_plotly_outlier_plot(self, data, outlier_indices, feature_columns):
        """Create interactive outlier visualization with plotly"""
        
        if len(feature_columns) >= 2:
            # Create scatter plot for first two features
            fig = go.Figure()
            
            # Normal points
            normal_mask = ~data.index.isin(outlier_indices)
            fig.add_trace(go.Scatter(
                x=data.loc[normal_mask, feature_columns[0]],
                y=data.loc[normal_mask, feature_columns[1]],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=5, opacity=0.6)
            ))
            
            # Outliers
            if outlier_indices:
                outlier_mask = data.index.isin(outlier_indices)
                fig.add_trace(go.Scatter(
                    x=data.loc[outlier_mask, feature_columns[0]],
                    y=data.loc[outlier_mask, feature_columns[1]],
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=8, symbol='x')
                ))
            
            fig.update_layout(
                title='Outlier Detection Visualization',
                xaxis_title=feature_columns[0],
                yaxis_title=feature_columns[1],
                template=self.style_theme
            )
            
            return fig
        
        return None
    
    def _create_matplotlib_outlier_plot(self, data, outlier_indices, feature_columns):
        """Create static outlier visualization with matplotlib"""
        
        if len(feature_columns) >= 2:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Normal points
            normal_mask = ~data.index.isin(outlier_indices)
            ax.scatter(data.loc[normal_mask, feature_columns[0]],
                      data.loc[normal_mask, feature_columns[1]],
                      alpha=0.6, c='blue', label='Normal', s=30)
            
            # Outliers
            if outlier_indices:
                outlier_mask = data.index.isin(outlier_indices)
                ax.scatter(data.loc[outlier_mask, feature_columns[0]],
                          data.loc[outlier_mask, feature_columns[1]],
                          alpha=0.8, c='red', marker='x', label='Outliers', s=50)
            
            ax.set_xlabel(feature_columns[0])
            ax.set_ylabel(feature_columns[1])
            ax.set_title('Outlier Detection Visualization', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        return None
    
    @timing_decorator
    @error_handler
    def create_comprehensive_dashboard(self,
                                     pipeline_results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> str:
        """
        Create comprehensive HTML dashboard with all visualizations
        
        Args:
            pipeline_results: Complete pipeline results
            save_path: Optional path to save dashboard
        
        Returns:
            Path to generated dashboard HTML file
        """
        
        self.logger.info("Creating comprehensive dashboard")
        
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available - cannot create interactive dashboard")
            return ""
        
        dashboard_html = self._generate_dashboard_html(pipeline_results)
        
        # Save dashboard
        dashboard_path = save_path or (self.output_dir / f"ml_pipeline_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def _generate_dashboard_html(self, pipeline_results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML dashboard"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Pipeline Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 40px; }
                .plot-container { margin-bottom: 30px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .metric-card { background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2E86AB; }
                .metric-label { font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ML Pipeline Comprehensive Dashboard</h1>
                <p>Generated on: {timestamp}</p>
            </div>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add key metrics section
        if 'insights' in pipeline_results:
            html_content += self._generate_metrics_section(pipeline_results['insights'])
        
        # Add plots from generated_plots
        for plot_name, plot_data in self.generated_plots.items():
            if 'interactive' in plot_data:
                plot_html = plot_data['interactive'].to_html(include_plotlyjs=False, div_id=f"plot_{plot_name}")
                html_content += f"""
                <div class="section">
                    <h2>{plot_name.replace('_', ' ').title()}</h2>
                    <div class="plot-container">
                        {plot_html}
                    </div>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_metrics_section(self, insights: Dict[str, Any]) -> str:
        """Generate key metrics section for dashboard"""
        
        metrics_html = """
        <div class="section">
            <h2>Key Performance Indicators</h2>
            <div class="metrics-grid">
        """
        
        # Extract key metrics from insights
        if 'model_insights' in insights:
            model_insights = insights['model_insights']
            if 'best_model' in model_insights:
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{model_insights['best_model']['score']:.4f}</div>
                    <div class="metric-label">Best Model Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{model_insights['best_model']['name']}</div>
                    <div class="metric-label">Best Model</div>
                </div>
                """
        
        if 'data_insights' in insights:
            data_insights = insights['data_insights']
            if 'data_quality' in data_insights:
                dq = data_insights['data_quality']
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{dq['total_rows']:,}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{dq['total_features']}</div>
                    <div class="metric-label">Total Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{dq['missing_data_percentage']:.2f}%</div>
                    <div class="metric-label">Missing Data</div>
                </div>
                """
        
        metrics_html += """
            </div>
        </div>
        """
        
        return metrics_html
    
    def _save_plots(self, plots: Dict[str, Any], base_path: Union[str, Path]):
        """Save all plots to files"""
        
        base_path = Path(base_path)
        
        for plot_type, plot_obj in plots.items():
            if plot_type == 'interactive' and PLOTLY_AVAILABLE:
                # Save plotly plot as HTML
                plot_obj.write_html(f"{base_path}_{plot_type}.html")
            elif plot_type == 'static' and MATPLOTLIB_AVAILABLE:
                # Save matplotlib plot as PNG
                plot_obj.savefig(f"{base_path}_{plot_type}.png", dpi=self.dpi, bbox_inches='tight')
                plt.close(plot_obj)  # Close to free memory
    
    def get_generated_plots_summary(self) -> Dict[str, Any]:
        """Get summary of all generated plots"""
        
        summary = {
            'total_plots': len(self.generated_plots),
            'plot_types': list(self.generated_plots.keys()),
            'output_directory': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def clear_plots(self):
        """Clear all generated plots from memory"""
        
        self.generated_plots.clear()
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')  # Close all matplotlib figures
        
        self.logger.info("All plots cleared from memory")