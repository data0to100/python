"""
Advanced visualization module for the ML Platform.

Provides comprehensive visualization capabilities using Plotly, Altair, and automated insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .config import MLConfig
from .exceptions import VisualizationError
from .logger import get_logger, structured_logging


class VisualizationEngine:
    """Enterprise-grade visualization engine with multiple library support."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        """Initialize visualization engine.
        
        Args:
            config: ML Platform configuration
        """
        self.config = config or MLConfig()
        self.logger = get_logger()
        
        # Configure Altair
        alt.data_transformers.enable('json')
        
    def create_eda_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive EDA dashboard.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing all visualizations and insights
        """
        with structured_logging("eda_dashboard_creation"):
            try:
                dashboard = {
                    'overview': self._generate_overview(df),
                    'distributions': self._create_distribution_plots(df),
                    'correlations': self._create_correlation_analysis(df),
                    'missing_data': self._create_missing_data_analysis(df),
                    'outliers': self._create_outlier_analysis(df),
                    'insights': self._generate_automated_insights(df)
                }
                
                return dashboard
                
            except Exception as e:
                raise VisualizationError(f"Failed to create EDA dashboard: {str(e)}", viz_type="eda_dashboard")
    
    def _generate_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset overview."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicates': df.duplicated().sum(),
            'column_types': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols),
                'datetime': len(datetime_cols)
            },
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols
        }
    
    def _create_distribution_plots(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create distribution plots for all variables."""
        plots = {}
        
        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Create subplot for all numeric distributions
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=numeric_cols,
                vertical_spacing=0.08
            )
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title="Numeric Variable Distributions",
                height=300 * n_rows,
                showlegend=False
            )
            plots['numeric_distributions'] = fig
        
        # Categorical distributions
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:10]:  # Limit to top 10 categorical columns
            value_counts = df[col].value_counts().head(20)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {col}",
                labels={'x': col, 'y': 'Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            plots[f'categorical_{col}'] = fig
        
        return plots
    
    def _create_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create correlation analysis visualizations."""
        plots = {}
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return plots
        
        # Correlation heatmap
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={'size': 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=600
        )
        plots['correlation_heatmap'] = fig
        
        # Pairwise scatter plots for highly correlated variables
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            # Create scatter plots for top 6 correlations
            for i, (var1, var2, corr) in enumerate(high_corr_pairs[:6]):
                fig = px.scatter(
                    df, x=var1, y=var2,
                    title=f"{var1} vs {var2} (r={corr:.3f})",
                    trendline="ols"
                )
                plots[f'scatter_{var1}_{var2}'] = fig
        
        return plots
    
    def _create_missing_data_analysis(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create missing data analysis visualizations."""
        plots = {}
        
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        if missing_data.sum() == 0:
            return plots
        
        # Missing data bar chart
        missing_cols = missing_percentage[missing_percentage > 0].sort_values(ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(x=missing_cols.values, y=missing_cols.index, orientation='h')
        ])
        
        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Columns",
            height=max(400, len(missing_cols) * 25)
        )
        plots['missing_data_bar'] = fig
        
        # Missing data heatmap
        if len(df.columns) <= 50:  # Only for manageable number of columns
            missing_matrix = df.isnull().astype(int)
            
            fig = go.Figure(data=go.Heatmap(
                z=missing_matrix.T.values,
                x=missing_matrix.index,
                y=missing_matrix.columns,
                colorscale=['white', 'red'],
                showscale=False
            ))
            
            fig.update_layout(
                title="Missing Data Pattern",
                xaxis_title="Row Index",
                yaxis_title="Columns",
                height=600
            )
            plots['missing_data_heatmap'] = fig
        
        return plots
    
    def _create_outlier_analysis(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create outlier analysis visualizations."""
        plots = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return plots
        
        # Box plots for outlier detection
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            fig.add_trace(
                go.Box(y=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Outlier Detection (Box Plots)",
            height=300 * n_rows
        )
        plots['outlier_boxplots'] = fig
        
        # Z-score based outlier detection
        outlier_counts = {}
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_counts[col] = np.sum(z_scores > 3)
        
        if any(count > 0 for count in outlier_counts.values()):
            fig = go.Figure(data=[
                go.Bar(x=list(outlier_counts.keys()), y=list(outlier_counts.values()))
            ])
            
            fig.update_layout(
                title="Outlier Count by Column (Z-score > 3)",
                xaxis_title="Columns",
                yaxis_title="Number of Outliers"
            )
            plots['outlier_counts'] = fig
        
        return plots
    
    def _generate_automated_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate automated insights from the data."""
        insights = []
        
        try:
            # Dataset size insights
            if len(df) > 1000000:
                insights.append(f"📊 Large dataset detected with {len(df):,} rows - consider sampling for faster processing")
            elif len(df) < 1000:
                insights.append(f"⚠️ Small dataset with only {len(df)} rows - be cautious about overfitting")
            
            # Missing data insights
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 20:
                insights.append(f"🚨 High missing data rate ({missing_pct:.1f}%) - consider data quality investigation")
            elif missing_pct > 5:
                insights.append(f"⚠️ Moderate missing data ({missing_pct:.1f}%) - consider imputation strategies")
            
            # Duplicate insights
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
            if duplicate_pct > 10:
                insights.append(f"🔄 High duplicate rate ({duplicate_pct:.1f}%) - consider deduplication")
            
            # Column type insights
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > len(numeric_cols) * 2:
                insights.append("📝 Dataset is primarily categorical - consider encoding strategies for ML models")
            
            # High cardinality insights
            high_cardinality_cols = []
            for col in categorical_cols:
                if df[col].nunique() > len(df) * 0.8:
                    high_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                insights.append(f"🎯 High cardinality columns detected: {', '.join(high_cardinality_cols)} - consider feature engineering")
            
            # Skewness insights
            skewed_cols = []
            for col in numeric_cols:
                if abs(df[col].skew()) > 2:
                    skewed_cols.append(col)
            
            if skewed_cols:
                insights.append(f"📈 Highly skewed columns: {', '.join(skewed_cols)} - consider transformations")
            
            # Correlation insights
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.9:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                
                if high_corr_pairs:
                    insights.append(f"🔗 High correlation detected between: {', '.join([f'{p[0]} & {p[1]}' for p in high_corr_pairs[:3]])}")
            
            # Memory usage insights
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            if memory_mb > 1000:
                insights.append(f"💾 Large memory footprint ({memory_mb:.1f} MB) - consider data type optimization")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return ["❌ Error generating automated insights"]
    
    def create_model_performance_viz(self, model_results: Dict[str, Any], task_type: str) -> Dict[str, go.Figure]:
        """Create model performance visualizations.
        
        Args:
            model_results: Results from model training
            task_type: Type of task (classification or regression)
            
        Returns:
            Dictionary of performance visualizations
        """
        plots = {}
        
        try:
            if task_type == "classification":
                plots.update(self._create_classification_plots(model_results))
            else:
                plots.update(self._create_regression_plots(model_results))
            
            return plots
            
        except Exception as e:
            raise VisualizationError(f"Failed to create model performance viz: {str(e)}", viz_type="model_performance")
    
    def _create_classification_plots(self, model_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create classification-specific visualizations."""
        plots = {}
        
        # Model comparison bar chart
        models_data = []
        for model_name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                models_data.append({
                    'model': model_name,
                    'accuracy': metrics.get('test_accuracy', 0),
                    'precision': metrics.get('test_precision', 0),
                    'recall': metrics.get('test_recall', 0),
                    'f1': metrics.get('test_f1', 0)
                })
        
        if models_data:
            df_models = pd.DataFrame(models_data)
            
            fig = go.Figure()
            
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=df_models['model'],
                    y=df_models[metric]
                ))
            
            fig.update_layout(
                title="Classification Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group',
                yaxis=dict(range=[0, 1])
            )
            plots['model_comparison'] = fig
        
        return plots
    
    def _create_regression_plots(self, model_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create regression-specific visualizations."""
        plots = {}
        
        # Model comparison bar chart
        models_data = []
        for model_name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                models_data.append({
                    'model': model_name,
                    'r2': metrics.get('test_r2', 0),
                    'mse': metrics.get('test_mse', 0),
                    'mae': metrics.get('test_mae', 0)
                })
        
        if models_data:
            df_models = pd.DataFrame(models_data)
            
            # R2 comparison
            fig = go.Figure(data=[
                go.Bar(x=df_models['model'], y=df_models['r2'])
            ])
            
            fig.update_layout(
                title="R² Score Comparison",
                xaxis_title="Models",
                yaxis_title="R² Score"
            )
            plots['r2_comparison'] = fig
            
            # Error metrics comparison
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Mean Squared Error', 'Mean Absolute Error']
            )
            
            fig.add_trace(
                go.Bar(x=df_models['model'], y=df_models['mse'], name='MSE'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df_models['model'], y=df_models['mae'], name='MAE'),
                row=1, col=2
            )
            
            fig.update_layout(title="Error Metrics Comparison", showlegend=False)
            plots['error_comparison'] = fig
        
        return plots
    
    def create_feature_importance_viz(self, feature_importance: pd.DataFrame, top_n: int = 20) -> go.Figure:
        """Create feature importance visualization.
        
        Args:
            feature_importance: DataFrame with feature names and importance scores
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        try:
            top_features = feature_importance.head(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h'
                )
            ])
            
            fig.update_layout(
                title=f"Top {top_n} Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, top_n * 25)
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create feature importance viz: {str(e)}", viz_type="feature_importance")
    
    def create_time_series_viz(self, df: pd.DataFrame, date_col: str, value_col: str, 
                              forecast_df: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create time series visualization with optional forecast.
        
        Args:
            df: Historical data
            date_col: Date column name
            value_col: Value column name
            forecast_df: Optional forecast data
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[value_col],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast data
            if forecast_df is not None:
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence intervals
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        name='Confidence Interval'
                    ))
            
            fig.update_layout(
                title="Time Series Analysis",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create time series viz: {str(e)}", viz_type="time_series")
    
    def create_cluster_analysis_viz(self, df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, go.Figure]:
        """Create cluster analysis visualizations.
        
        Args:
            df: DataFrame with numeric data
            n_clusters: Number of clusters
            
        Returns:
            Dictionary of cluster visualizations
        """
        plots = {}
        
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df.columns) < 2:
                return plots
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(numeric_df)
            
            # PCA for visualization
            if len(numeric_df.columns) > 2:
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(numeric_df)
                
                fig = px.scatter(
                    x=pca_data[:, 0],
                    y=pca_data[:, 1],
                    color=clusters,
                    title=f"K-Means Clustering (PCA) - {n_clusters} Clusters",
                    labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                           'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
                )
                plots['cluster_pca'] = fig
            
            # First two numeric columns
            if len(numeric_df.columns) >= 2:
                cols = numeric_df.columns[:2]
                fig = px.scatter(
                    numeric_df,
                    x=cols[0],
                    y=cols[1],
                    color=clusters,
                    title=f"K-Means Clustering - {n_clusters} Clusters"
                )
                plots['cluster_scatter'] = fig
            
            return plots
            
        except Exception as e:
            raise VisualizationError(f"Failed to create cluster analysis: {str(e)}", viz_type="cluster_analysis")
    
    def create_altair_viz(self, df: pd.DataFrame, chart_type: str, **kwargs) -> alt.Chart:
        """Create Altair visualization.
        
        Args:
            df: DataFrame
            chart_type: Type of chart (scatter, bar, line, histogram)
            **kwargs: Chart configuration
            
        Returns:
            Altair chart
        """
        try:
            if chart_type == "scatter":
                chart = alt.Chart(df).mark_circle(size=60).encode(
                    x=kwargs.get('x'),
                    y=kwargs.get('y'),
                    color=kwargs.get('color'),
                    tooltip=list(df.columns)
                ).interactive()
            
            elif chart_type == "bar":
                chart = alt.Chart(df).mark_bar().encode(
                    x=kwargs.get('x'),
                    y=kwargs.get('y'),
                    color=kwargs.get('color')
                )
            
            elif chart_type == "line":
                chart = alt.Chart(df).mark_line().encode(
                    x=kwargs.get('x'),
                    y=kwargs.get('y'),
                    color=kwargs.get('color')
                )
            
            elif chart_type == "histogram":
                chart = alt.Chart(df).mark_bar().encode(
                    alt.X(kwargs.get('column'), bin=True),
                    y='count()'
                )
            
            else:
                raise VisualizationError(f"Unsupported chart type: {chart_type}", viz_type="altair")
            
            return chart.resolve_scale(color='independent')
            
        except Exception as e:
            raise VisualizationError(f"Failed to create Altair viz: {str(e)}", viz_type="altair", library="altair")


class InsightsGenerator:
    """Generate automated insights from data and models."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        """Initialize insights generator."""
        self.config = config or MLConfig()
        self.logger = get_logger()
    
    def generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Statistical summary
        """
        summary = {
            'basic_stats': {},
            'distribution_analysis': {},
            'relationship_analysis': {},
            'quality_assessment': {}
        }
        
        try:
            # Basic statistics
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                summary['basic_stats'] = {
                    'descriptive_stats': numeric_df.describe().to_dict(),
                    'skewness': numeric_df.skew().to_dict(),
                    'kurtosis': numeric_df.kurtosis().to_dict()
                }
            
            # Distribution analysis
            summary['distribution_analysis'] = self._analyze_distributions(df)
            
            # Relationship analysis
            if len(numeric_df.columns) >= 2:
                summary['relationship_analysis'] = self._analyze_relationships(numeric_df)
            
            # Data quality assessment
            summary['quality_assessment'] = self._assess_data_quality(df)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating statistical summary: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variable distributions."""
        analysis = {
            'normality_tests': {},
            'distribution_types': {},
            'outlier_analysis': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            if len(data) > 3:
                # Normality test
                try:
                    stat, p_value = stats.normaltest(data)
                    analysis['normality_tests'][col] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                except:
                    analysis['normality_tests'][col] = {'error': 'Test failed'}
                
                # Outlier detection using IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                
                analysis['outlier_analysis'][col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(data)) * 100
                }
        
        return analysis
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between variables."""
        analysis = {
            'correlation_matrix': df.corr().to_dict(),
            'strong_correlations': [],
            'multicollinearity_check': {}
        }
        
        # Find strong correlations
        corr_matrix = df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    analysis['strong_correlations'].append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return analysis
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        assessment = {
            'completeness_score': 0,
            'consistency_score': 0,
            'validity_score': 0,
            'overall_score': 0,
            'issues': []
        }
        
        try:
            # Completeness (missing data)
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            assessment['completeness_score'] = max(0, 100 - missing_pct)
            
            # Consistency (duplicates)
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
            assessment['consistency_score'] = max(0, 100 - duplicate_pct * 2)
            
            # Validity (basic checks)
            validity_issues = 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Check for infinite values
                if np.isinf(df[col]).any():
                    validity_issues += 1
                    assessment['issues'].append(f"Infinite values in {col}")
                
                # Check for extremely large values (potential data entry errors)
                if (df[col] > df[col].quantile(0.99) * 100).any():
                    validity_issues += 1
                    assessment['issues'].append(f"Extreme outliers in {col}")
            
            assessment['validity_score'] = max(0, 100 - validity_issues * 10)
            
            # Overall score
            assessment['overall_score'] = np.mean([
                assessment['completeness_score'],
                assessment['consistency_score'],
                assessment['validity_score']
            ])
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {str(e)}")
            return {"error": str(e)}