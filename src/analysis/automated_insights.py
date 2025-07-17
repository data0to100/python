"""
Automated Insights Generator
Provides AI-powered insights and recommendations for data analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Insight:
    """Data class to represent an analytical insight."""
    category: str
    title: str
    description: str
    importance: str  # 'high', 'medium', 'low'
    evidence: Dict[str, Any]
    recommendation: str
    confidence: float  # 0.0 to 1.0

class AutomatedInsightsGenerator:
    """
    Automated insights generator that analyzes datasets and provides
    actionable insights and recommendations.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the insights generator.
        
        Args:
            confidence_threshold: Minimum confidence level for insights
        """
        self.confidence_threshold = confidence_threshold
        self.insights = []
        
    def generate_insights(self, df: pd.DataFrame, target_column: Optional[str] = None) -> List[Insight]:
        """
        Generate comprehensive insights for a dataset.
        
        Args:
            df: Input dataframe
            target_column: Target column for supervised analysis
            
        Returns:
            List of insights
        """
        self.insights = []
        
        logger.info("Generating automated insights...")
        
        # Data quality insights
        self._analyze_data_quality(df)
        
        # Distribution insights
        self._analyze_distributions(df)
        
        # Correlation insights
        self._analyze_correlations(df, target_column)
        
        # Outlier insights
        self._analyze_outliers(df)
        
        # Feature importance insights
        if target_column and target_column in df.columns:
            self._analyze_feature_importance(df, target_column)
        
        # Clustering insights
        self._analyze_clusters(df)
        
        # Trend insights for time series
        self._analyze_trends(df)
        
        # Filter by confidence threshold
        high_confidence_insights = [
            insight for insight in self.insights 
            if insight.confidence >= self.confidence_threshold
        ]
        
        # Sort by importance and confidence
        importance_order = {'high': 3, 'medium': 2, 'low': 1}
        high_confidence_insights.sort(
            key=lambda x: (importance_order[x.importance], x.confidence),
            reverse=True
        )
        
        logger.info(f"Generated {len(high_confidence_insights)} high-confidence insights")
        return high_confidence_insights
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> None:
        """Analyze data quality issues."""
        
        # Missing data analysis
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_pct[missing_pct > 20].index.tolist()
        
        if high_missing_cols:
            self.insights.append(Insight(
                category="Data Quality",
                title="High Missing Data Detected",
                description=f"Columns {high_missing_cols} have >20% missing values",
                importance="high",
                evidence={"missing_percentages": missing_pct[high_missing_cols].to_dict()},
                recommendation="Consider imputation strategies or feature engineering",
                confidence=0.9
            ))
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            importance = "high" if duplicate_pct > 5 else "medium"
            
            self.insights.append(Insight(
                category="Data Quality",
                title="Duplicate Records Found",
                description=f"Found {duplicate_count} duplicate records ({duplicate_pct:.1f}%)",
                importance=importance,
                evidence={"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_pct},
                recommendation="Review and remove or consolidate duplicate records",
                confidence=0.95
            ))
    
    def _analyze_distributions(self, df: pd.DataFrame) -> None:
        """Analyze variable distributions."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].nunique() < 2:
                continue
                
            # Skewness analysis
            skewness = stats.skew(df[col].dropna())
            if abs(skewness) > 1:
                self.insights.append(Insight(
                    category="Distribution",
                    title=f"Highly Skewed Distribution: {col}",
                    description=f"Column '{col}' has skewness of {skewness:.2f}",
                    importance="medium",
                    evidence={"skewness": skewness},
                    recommendation="Consider log transformation or other normalization techniques",
                    confidence=0.8
                ))
            
            # Outlier analysis using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                outlier_pct = (len(outliers) / len(df)) * 100
                self.insights.append(Insight(
                    category="Distribution",
                    title=f"High Outlier Concentration: {col}",
                    description=f"Column '{col}' has {outlier_pct:.1f}% outliers",
                    importance="medium",
                    evidence={"outlier_percentage": outlier_pct, "outlier_count": len(outliers)},
                    recommendation="Investigate outliers - may indicate data quality issues or interesting patterns",
                    confidence=0.85
                ))
    
    def _analyze_correlations(self, df: pd.DataFrame, target_column: Optional[str] = None) -> None:
        """Analyze correlations between variables."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
            
        corr_matrix = df[numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            self.insights.append(Insight(
                category="Correlation",
                title="Strong Correlations Detected",
                description=f"Found {len(high_corr_pairs)} highly correlated variable pairs",
                importance="medium",
                evidence={"high_correlations": high_corr_pairs},
                recommendation="Consider dimensionality reduction or feature selection",
                confidence=0.9
            ))
        
        # Target correlation analysis
        if target_column and target_column in numeric_cols:
            target_corrs = corr_matrix[target_column].abs().sort_values(ascending=False)
            top_corrs = target_corrs.head(6)[1:]  # Exclude self-correlation
            
            if len(top_corrs) > 0 and top_corrs.iloc[0] > 0.5:
                self.insights.append(Insight(
                    category="Feature Importance",
                    title=f"Strong Predictors for {target_column}",
                    description=f"Top correlated features: {', '.join(top_corrs.head(3).index)}",
                    importance="high",
                    evidence={"top_correlations": top_corrs.head(5).to_dict()},
                    recommendation="Focus on these features for predictive modeling",
                    confidence=0.85
                ))
    
    def _analyze_outliers(self, df: pd.DataFrame) -> None:
        """Analyze outliers across the dataset."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Multivariate outlier detection using Mahalanobis distance
        if len(numeric_cols) >= 2:
            try:
                data_clean = df[numeric_cols].dropna()
                if len(data_clean) > 0:
                    # Standardize the data
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_clean)
                    
                    # Calculate Mahalanobis distance
                    cov_matrix = np.cov(data_scaled.T)
                    inv_cov_matrix = np.linalg.pinv(cov_matrix)
                    
                    distances = []
                    for i in range(len(data_scaled)):
                        diff = data_scaled[i] - np.mean(data_scaled, axis=0)
                        distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                        distances.append(distance)
                    
                    # Identify outliers (distance > 3 standard deviations)
                    threshold = np.mean(distances) + 3 * np.std(distances)
                    outliers = [i for i, d in enumerate(distances) if d > threshold]
                    
                    if len(outliers) > len(data_clean) * 0.02:  # More than 2% outliers
                        outlier_pct = (len(outliers) / len(data_clean)) * 100
                        self.insights.append(Insight(
                            category="Outliers",
                            title="Multivariate Outliers Detected",
                            description=f"Found {len(outliers)} multivariate outliers ({outlier_pct:.1f}%)",
                            importance="medium",
                            evidence={"outlier_count": len(outliers), "outlier_percentage": outlier_pct},
                            recommendation="Investigate multivariate outliers for data quality or interesting patterns",
                            confidence=0.8
                        ))
            except Exception as e:
                logger.warning(f"Multivariate outlier analysis failed: {e}")
    
    def _analyze_feature_importance(self, df: pd.DataFrame, target_column: str) -> None:
        """Analyze feature importance using mutual information."""
        
        try:
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != target_column]
            X = df[feature_cols]
            y = df[target_column]
            
            # Handle missing values
            X_clean = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
            y_clean = y.dropna()
            
            # Align X and y
            common_idx = X_clean.index.intersection(y_clean.index)
            X_clean = X_clean.loc[common_idx]
            y_clean = y_clean.loc[common_idx]
            
            if len(X_clean) == 0:
                return
            
            # Encode categorical variables
            X_encoded = pd.get_dummies(X_clean, drop_first=True)
            
            # Calculate mutual information
            if y_clean.dtype in ['object', 'category'] or y_clean.nunique() < 10:
                # Classification
                mi_scores = mutual_info_classif(X_encoded, y_clean)
            else:
                # Regression
                mi_scores = mutual_info_regression(X_encoded, y_clean)
            
            # Get top features
            feature_importance = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
            top_features = feature_importance.head(5)
            
            if top_features.iloc[0] > 0.1:  # Significant mutual information
                self.insights.append(Insight(
                    category="Feature Importance",
                    title="Important Features Identified",
                    description=f"Top predictive features for {target_column}",
                    importance="high",
                    evidence={"feature_scores": top_features.to_dict()},
                    recommendation="Focus on these features for model development",
                    confidence=0.8
                ))
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
    
    def _analyze_clusters(self, df: pd.DataFrame) -> None:
        """Analyze natural clusters in the data."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
            
        try:
            # Prepare data
            data_clean = df[numeric_cols].dropna()
            if len(data_clean) < 10:
                return
                
            # Standardize
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_clean)
            
            # Try different numbers of clusters
            inertias = []
            k_range = range(2, min(10, len(data_clean)//2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (simple method)
            if len(inertias) >= 3:
                # Calculate rate of change
                rates = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
                optimal_k = k_range[rates.index(max(rates)) + 1]
                
                # Perform clustering with optimal k
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(data_scaled)
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(data_scaled, clusters)
                
                if sil_score > 0.3:  # Reasonable clustering
                    self.insights.append(Insight(
                        category="Clustering",
                        title="Natural Clusters Detected",
                        description=f"Data shows {optimal_k} distinct clusters (silhouette score: {sil_score:.2f})",
                        importance="medium",
                        evidence={"optimal_clusters": optimal_k, "silhouette_score": sil_score},
                        recommendation="Consider cluster-based analysis or segmentation",
                        confidence=0.7
                    ))
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
    
    def _analyze_trends(self, df: pd.DataFrame) -> None:
        """Analyze trends in time-based data."""
        
        # Look for date/datetime columns
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        
        for col in date_cols:
            try:
                # Try to convert to datetime if not already
                if df[col].dtype == 'object':
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    if date_series.isna().sum() > len(df) * 0.5:
                        continue  # Too many conversion failures
                else:
                    date_series = df[col]
                
                # Check if we have a reasonable date range
                date_range = date_series.max() - date_series.min()
                if date_range.days < 7:  # Less than a week
                    continue
                
                # Analyze trends in numeric columns over time
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    # Create time series
                    ts_data = df[[col, num_col]].copy()
                    ts_data[col] = date_series
                    ts_data = ts_data.dropna().sort_values(col)
                    
                    if len(ts_data) < 10:
                        continue
                    
                    # Simple trend analysis using correlation with time
                    ts_data['time_numeric'] = (ts_data[col] - ts_data[col].min()).dt.days
                    correlation = ts_data['time_numeric'].corr(ts_data[num_col])
                    
                    if abs(correlation) > 0.3:  # Moderate correlation
                        trend_direction = "increasing" if correlation > 0 else "decreasing"
                        self.insights.append(Insight(
                            category="Trends",
                            title=f"Time Trend in {num_col}",
                            description=f"{num_col} shows {trend_direction} trend over time (correlation: {correlation:.2f})",
                            importance="medium",
                            evidence={"correlation": correlation, "trend_direction": trend_direction},
                            recommendation=f"Monitor {trend_direction} trend in {num_col} for business implications",
                            confidence=0.7
                        ))
                        break  # Only report one trend per date column
                        
            except Exception as e:
                logger.warning(f"Trend analysis failed for column {col}: {e}")
    
    def generate_summary_report(self, insights: List[Insight]) -> str:
        """Generate a formatted summary report of insights."""
        
        if not insights:
            return "No significant insights found with the current confidence threshold."
        
        report = "🔍 AUTOMATED DATA ANALYSIS INSIGHTS\n"
        report += "=" * 50 + "\n\n"
        
        # Group by category
        categories = {}
        for insight in insights:
            if insight.category not in categories:
                categories[insight.category] = []
            categories[insight.category].append(insight)
        
        # Generate report by category
        for category, cat_insights in categories.items():
            report += f"📊 {category.upper()}\n"
            report += "-" * 30 + "\n"
            
            for i, insight in enumerate(cat_insights, 1):
                importance_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                report += f"{i}. {importance_emoji[insight.importance]} {insight.title}\n"
                report += f"   Description: {insight.description}\n"
                report += f"   Recommendation: {insight.recommendation}\n"
                report += f"   Confidence: {insight.confidence:.1%}\n\n"
        
        # Summary statistics
        high_importance = len([i for i in insights if i.importance == "high"])
        medium_importance = len([i for i in insights if i.importance == "medium"])
        low_importance = len([i for i in insights if i.importance == "low"])
        
        report += "📈 SUMMARY\n"
        report += "-" * 30 + "\n"
        report += f"Total Insights: {len(insights)}\n"
        report += f"High Priority: {high_importance}\n"
        report += f"Medium Priority: {medium_importance}\n"
        report += f"Low Priority: {low_importance}\n"
        report += f"Average Confidence: {np.mean([i.confidence for i in insights]):.1%}\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.lognormal(10.5, 0.8, n_samples),
        'score': np.random.normal(100, 15, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Add some missing values
    sample_data.loc[sample_data.sample(50).index, 'income'] = np.nan
    
    # Add some outliers
    sample_data.loc[sample_data.sample(20).index, 'score'] = np.random.normal(200, 10, 20)
    
    # Generate insights
    generator = AutomatedInsightsGenerator()
    insights = generator.generate_insights(sample_data, target_column='target')
    
    # Print report
    print(generator.generate_summary_report(insights))