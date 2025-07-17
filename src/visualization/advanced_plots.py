"""
Advanced Data Visualization Utilities
Comprehensive plotting functions for data analysis and exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings
from scipy import stats
import logging

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """
    Advanced visualization utility for comprehensive data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer with default settings.
        
        Args:
            figsize: Default figure size for matplotlib plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
        # Color palettes
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#F39C12',
            'success': '#27AE60',
            'danger': '#E74C3C',
            'warning': '#F1C40F',
            'info': '#8E44AD'
        }
    
    def correlation_heatmap(self, 
                           df: pd.DataFrame, 
                           method: str = 'pearson',
                           annot: bool = True,
                           mask_upper: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create an advanced correlation heatmap.
        
        Args:
            df: DataFrame containing numeric data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells with correlation values
            mask_upper: Whether to mask upper triangle
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr(method=method)
        
        # Create mask for upper triangle if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=annot, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True, 
                   linewidths=0.5, 
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def distribution_analysis(self, 
                             df: pd.DataFrame, 
                             columns: Optional[List[str]] = None,
                             plot_type: str = 'auto',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive distribution analysis plots.
        
        Args:
            df: DataFrame to analyze
            columns: Specific columns to analyze (if None, all numeric columns)
            plot_type: Type of plot ('hist', 'kde', 'box', 'violin', 'auto')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = df[column].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data for {column}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Choose plot type automatically or use specified
            if plot_type == 'auto':
                if len(data.unique()) < 20:
                    # Bar plot for categorical-like data
                    data.value_counts().plot(kind='bar', ax=ax)
                else:
                    # Histogram with KDE for continuous data
                    sns.histplot(data, kde=True, ax=ax, alpha=0.7)
            elif plot_type == 'hist':
                sns.histplot(data, ax=ax, alpha=0.7)
            elif plot_type == 'kde':
                sns.kdeplot(data, ax=ax)
            elif plot_type == 'box':
                sns.boxplot(y=data, ax=ax)
            elif plot_type == 'violin':
                sns.violinplot(y=data, ax=ax)
            
            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}', alpha=0.8)
            ax.axvline(median_val, color='green', linestyle='--', 
                      label=f'Median: {median_val:.2f}', alpha=0.8)
            
            ax.set_title(f'{column}\n(μ={mean_val:.2f}, σ={std_val:.2f})', 
                        fontweight='bold')
            ax.legend(fontsize=8)
        
        # Remove empty subplots
        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def pairplot_advanced(self, 
                         df: pd.DataFrame, 
                         hue: Optional[str] = None,
                         diag_kind: str = 'auto',
                         save_path: Optional[str] = None) -> sns.PairGrid:
        """
        Create an advanced pairplot with statistical information.
        
        Args:
            df: DataFrame to plot
            hue: Column name for color encoding
            diag_kind: Type of diagonal plots ('auto', 'hist', 'kde')
            save_path: Path to save the plot
            
        Returns:
            Seaborn PairGrid object
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for pairplot")
        
        # Limit to reasonable number of columns for performance
        if len(numeric_cols) > 6:
            logger.warning(f"Too many columns ({len(numeric_cols)}). Using first 6.")
            numeric_cols = numeric_cols[:6]
        
        plot_df = df[numeric_cols + ([hue] if hue else [])].copy()
        
        # Create the pairplot
        if diag_kind == 'auto':
            diag_kind = 'hist'
        
        g = sns.pairplot(plot_df, 
                        hue=hue, 
                        diag_kind=diag_kind,
                        plot_kws={'alpha': 0.6, 's': 20},
                        diag_kws={'alpha': 0.7})
        
        # Add correlation coefficients to upper triangle
        def corrfunc(x, y, **kws):
            r, p = stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(f'r = {r:.2f}\np = {p:.3f}',
                       xy=(0.1, 0.9), xycoords=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        
        g.map_upper(corrfunc)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return g
    
    def outlier_analysis(self, 
                        df: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        method: str = 'iqr',
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive outlier analysis visualization.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to analyze (if None, all numeric columns)
            method: Outlier detection method ('iqr', 'zscore', 'isolation')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = df[column].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data for {column}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Detect outliers based on method
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > 3]
                
            # Create box plot with outliers highlighted
            bp = ax.boxplot(data, patch_artist=True, 
                           boxprops=dict(facecolor=self.colors['primary'], alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            
            # Add outlier count
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100
            
            ax.set_title(f'{column}\nOutliers: {outlier_count} ({outlier_pct:.1f}%)', 
                        fontweight='bold')
            ax.set_ylabel('Value')
            
            # Add text with statistics
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nSkew: {data.skew():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Remove empty subplots
        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def interactive_scatter_plot(self, 
                               df: pd.DataFrame, 
                               x: str, 
                               y: str,
                               color: Optional[str] = None,
                               size: Optional[str] = None,
                               hover_data: Optional[List[str]] = None,
                               title: Optional[str] = None) -> go.Figure:
        """
        Create an interactive scatter plot using Plotly.
        
        Args:
            df: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            color: Column name for color encoding
            size: Column name for size encoding
            hover_data: Additional columns to show on hover
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if title is None:
            title = f'{y} vs {x}'
        
        fig = px.scatter(df, 
                        x=x, 
                        y=y, 
                        color=color,
                        size=size,
                        hover_data=hover_data,
                        title=title,
                        template='plotly_white')
        
        # Add trendline
        if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
            # Calculate correlation
            corr = df[x].corr(df[y])
            
            # Add trendline
            fig.add_scatter(x=df[x], y=np.poly1d(np.polyfit(df[x], df[y], 1))(df[x]),
                           mode='lines', name=f'Trendline (r={corr:.3f})',
                           line=dict(color='red', dash='dash'))
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_font_size=16
        )
        
        return fig
    
    def time_series_analysis(self, 
                            df: pd.DataFrame, 
                            date_col: str, 
                            value_cols: List[str],
                            resample_freq: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive time series analysis plots.
        
        Args:
            df: DataFrame containing time series data
            date_col: Name of the date column
            value_cols: List of value column names to plot
            resample_freq: Resampling frequency ('D', 'W', 'M', etc.)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Prepare data
        ts_df = df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.set_index(date_col).sort_index()
        
        # Resample if requested
        if resample_freq:
            ts_df = ts_df.resample(resample_freq).mean()
        
        # Create subplots
        n_plots = len(value_cols) + 1  # +1 for correlation plot if multiple series
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot each time series
        for i, col in enumerate(value_cols):
            ax = axes[i]
            
            # Plot the time series
            ts_df[col].plot(ax=ax, linewidth=2, color=list(self.colors.values())[i % len(self.colors)])
            
            # Add trend line
            x_numeric = np.arange(len(ts_df))
            z = np.polyfit(x_numeric, ts_df[col].dropna(), 1)
            trend_line = np.poly1d(z)(x_numeric)
            ax.plot(ts_df.index, trend_line, '--', alpha=0.8, color='red', 
                   label=f'Trend (slope: {z[0]:.4f})')
            
            # Calculate and display statistics
            series_stats = {
                'Mean': ts_df[col].mean(),
                'Std': ts_df[col].std(),
                'Min': ts_df[col].min(),
                'Max': ts_df[col].max()
            }
            
            stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in series_stats.items()])
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            ax.set_title(f'{col} Over Time', fontweight='bold', fontsize=14)
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add correlation heatmap if multiple series
        if len(value_cols) > 1:
            ax = axes[-1]
            corr_matrix = ts_df[value_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, ax=ax)
            ax.set_title('Cross-Correlation Matrix', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def categorical_analysis(self, 
                           df: pd.DataFrame, 
                           cat_cols: List[str],
                           target_col: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive categorical variable analysis.
        
        Args:
            df: DataFrame to analyze
            cat_cols: List of categorical column names
            target_col: Target variable for comparison
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        n_cols = min(2, len(cat_cols))
        n_rows = (len(cat_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cat_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if target_col and target_col in df.columns:
                # Create grouped analysis
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    # Box plot for numeric target
                    sns.boxplot(data=df, x=col, y=target_col, ax=ax)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    # Stacked bar chart for categorical target
                    ct = pd.crosstab(df[col], df[target_col], normalize='index')
                    ct.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Simple value counts
                value_counts = df[col].value_counts()
                
                # Limit to top 10 categories for readability
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    title_suffix = f" (Top 10 of {df[col].nunique()})"
                else:
                    title_suffix = ""
                
                value_counts.plot(kind='bar', ax=ax, alpha=0.8, 
                                color=self.colors['primary'])
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentages
                total = value_counts.sum()
                for j, v in enumerate(value_counts.values):
                    ax.text(j, v + total * 0.01, f'{v/total*100:.1f}%', 
                           ha='center', fontweight='bold')
            
            ax.set_title(f'{col}{title_suffix if "title_suffix" in locals() else ""}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(cat_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def missing_data_visualization(self, 
                                  df: pd.DataFrame, 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization for missing data patterns.
        
        Args:
            df: DataFrame to analyze
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Missing data heatmap
        missing_data = df.isnull()
        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Missing Data Pattern', fontweight='bold')
        
        # 2. Missing data by column
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            missing_counts.plot(kind='bar', ax=axes[0, 1], color=self.colors['danger'])
            axes[0, 1].set_title('Missing Values by Column', fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add percentages
            total_rows = len(df)
            for i, v in enumerate(missing_counts.values):
                axes[0, 1].text(i, v + total_rows * 0.01, f'{v/total_rows*100:.1f}%', 
                               ha='center', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No missing data found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=14, fontweight='bold')
        
        # 3. Missing data correlation
        if len(missing_counts) > 1:
            missing_corr = df.isnull().corr()
            sns.heatmap(missing_corr, annot=True, cmap='RdYlBu_r', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Missing Data Correlation', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient missing data for correlation', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12)
        
        # 4. Completeness by row
        completeness = df.isnull().sum(axis=1)
        completeness_dist = completeness.value_counts().sort_index()
        
        completeness_dist.plot(kind='bar', ax=axes[1, 1], color=self.colors['info'])
        axes[1, 1].set_title('Row Completeness Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Missing Values per Row')
        axes[1, 1].set_ylabel('Number of Rows')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class InteractiveDashboard:
    """
    Create interactive dashboards using Plotly.
    """
    
    @staticmethod
    def create_overview_dashboard(df: pd.DataFrame) -> go.Figure:
        """
        Create an interactive overview dashboard.
        
        Args:
            df: DataFrame to create dashboard for
            
        Returns:
            Plotly figure object with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Data Types Distribution', 'Missing Data', 
                          'Numeric Distributions', 'Correlation Matrix'],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # 1. Data types pie chart
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), 
                  values=dtype_counts.values,
                  name="Data Types"),
            row=1, col=1
        )
        
        # 2. Missing data bar chart
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig.add_trace(
                go.Bar(x=missing_data.index, 
                      y=missing_data.values,
                      name="Missing Values"),
                row=1, col=2
            )
        
        # 3. Numeric distributions (first numeric column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_numeric = df[numeric_cols[0]].dropna()
            fig.add_trace(
                go.Histogram(x=first_numeric, 
                           name=f"{numeric_cols[0]} Distribution"),
                row=2, col=1
            )
        
        # 4. Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          colorscale='RdYlBu',
                          name="Correlation"),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Dataset Overview Dashboard",
            title_font_size=20,
            showlegend=False
        )
        
        return fig