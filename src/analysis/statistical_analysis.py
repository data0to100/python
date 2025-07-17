"""
Advanced Statistical Analysis Module
Comprehensive statistical testing and analysis utilities for data analysts.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import (
    normaltest, shapiro, kstest, chi2_contingency, fisher_exact,
    pearsonr, spearmanr, kendalltau, mannwhitneyu, wilcoxon,
    kruskal, friedmanchisquare, levene, bartlett, f_oneway,
    ttest_ind, ttest_rel, ttest_1samp
)
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, coint
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis toolkit for data analysts.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize StatisticalAnalyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        self.results = {}
    
    def normality_tests(self, data: pd.Series, method: str = 'all') -> Dict[str, Any]:
        """
        Perform comprehensive normality testing.
        
        Args:
            data: Series to test for normality
            method: Test method ('shapiro', 'jarque_bera', 'dagostino', 'all')
            
        Returns:
            Dictionary with test results
        """
        results = {'variable': data.name if data.name else 'unnamed'}
        
        # Remove missing values
        clean_data = data.dropna()
        
        if len(clean_data) < 3:
            return {'error': 'Insufficient data for normality testing'}
        
        if method in ['shapiro', 'all']:
            if len(clean_data) <= 5000:  # Shapiro-Wilk works best for smaller samples
                stat, p_value = shapiro(clean_data)
                results['shapiro_wilk'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > self.alpha,
                    'interpretation': f"Data {'appears' if p_value > self.alpha else 'does not appear'} to be normally distributed"
                }
        
        if method in ['jarque_bera', 'all']:
            stat, p_value = jarque_bera(clean_data)
            results['jarque_bera'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > self.alpha,
                'interpretation': f"Data {'appears' if p_value > self.alpha else 'does not appear'} to be normally distributed"
            }
        
        if method in ['dagostino', 'all']:
            stat, p_value = normaltest(clean_data)
            results['dagostino_pearson'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > self.alpha,
                'interpretation': f"Data {'appears' if p_value > self.alpha else 'does not appear'} to be normally distributed"
            }
        
        return results
    
    def correlation_analysis(self, 
                           df: pd.DataFrame, 
                           method: str = 'pearson',
                           return_matrix: bool = False) -> Dict[str, Any]:
        """
        Comprehensive correlation analysis.
        
        Args:
            df: DataFrame for correlation analysis
            method: Correlation method ('pearson', 'spearman', 'kendall', 'all')
            return_matrix: Whether to return correlation matrices
            
        Returns:
            Dictionary with correlation results
        """
        results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        numeric_df = df[numeric_cols].dropna()
        
        if method in ['pearson', 'all']:
            corr_matrix = numeric_df.corr(method='pearson')
            results['pearson'] = {
                'matrix': corr_matrix if return_matrix else None,
                'significant_correlations': self._find_significant_correlations(
                    numeric_df, 'pearson'
                ),
                'strongest_correlation': self._find_strongest_correlation(corr_matrix)
            }
        
        if method in ['spearman', 'all']:
            corr_matrix = numeric_df.corr(method='spearman')
            results['spearman'] = {
                'matrix': corr_matrix if return_matrix else None,
                'significant_correlations': self._find_significant_correlations(
                    numeric_df, 'spearman'
                ),
                'strongest_correlation': self._find_strongest_correlation(corr_matrix)
            }
        
        if method in ['kendall', 'all']:
            corr_matrix = numeric_df.corr(method='kendall')
            results['kendall'] = {
                'matrix': corr_matrix if return_matrix else None,
                'significant_correlations': self._find_significant_correlations(
                    numeric_df, 'kendall'
                ),
                'strongest_correlation': self._find_strongest_correlation(corr_matrix)
            }
        
        return results
    
    def hypothesis_tests(self, 
                        group1: pd.Series, 
                        group2: Optional[pd.Series] = None,
                        test_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform appropriate hypothesis tests based on data characteristics.
        
        Args:
            group1: First group of data
            group2: Second group of data (for two-sample tests)
            test_type: Type of test ('auto', 'parametric', 'nonparametric')
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Clean data
        group1_clean = group1.dropna()
        
        if group2 is not None:
            group2_clean = group2.dropna()
            
            # Two-sample tests
            if test_type == 'auto':
                # Check normality and equal variances
                norm1 = self.normality_tests(group1_clean, 'shapiro')
                norm2 = self.normality_tests(group2_clean, 'shapiro')
                
                if ('shapiro_wilk' in norm1 and norm1['shapiro_wilk']['is_normal'] and
                    'shapiro_wilk' in norm2 and norm2['shapiro_wilk']['is_normal']):
                    # Check equal variances
                    levene_stat, levene_p = levene(group1_clean, group2_clean)
                    equal_var = levene_p > self.alpha
                    
                    # Independent t-test
                    stat, p_value = ttest_ind(group1_clean, group2_clean, equal_var=equal_var)
                    results['t_test'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'equal_variances': equal_var,
                        'test_type': 'Independent t-test'
                    }
                else:
                    # Mann-Whitney U test (non-parametric)
                    stat, p_value = mannwhitneyu(group1_clean, group2_clean, 
                                               alternative='two-sided')
                    results['mann_whitney'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'test_type': 'Mann-Whitney U test'
                    }
            
            # Effect size (Cohen's d)
            cohens_d = self._calculate_cohens_d(group1_clean, group2_clean)
            results['effect_size'] = {
                'cohens_d': cohens_d,
                'magnitude': self._interpret_cohens_d(cohens_d)
            }
        
        else:
            # One-sample tests
            # Test against population mean of 0
            stat, p_value = ttest_1samp(group1_clean, 0)
            results['one_sample_t'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'test_type': 'One-sample t-test against 0'
            }
        
        return results
    
    def anova_analysis(self, 
                      data: pd.DataFrame, 
                      dependent_var: str, 
                      independent_vars: List[str]) -> Dict[str, Any]:
        """
        Perform ANOVA analysis.
        
        Args:
            data: DataFrame containing the data
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names
            
        Returns:
            Dictionary with ANOVA results
        """
        results = {}
        
        # One-way ANOVA for each independent variable
        for var in independent_vars:
            groups = []
            group_names = []
            
            for group_name in data[var].unique():
                if pd.notna(group_name):
                    group_data = data[data[var] == group_name][dependent_var].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        group_names.append(group_name)
            
            if len(groups) >= 2:
                # Perform one-way ANOVA
                f_stat, p_value = f_oneway(*groups)
                
                results[var] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'groups': group_names,
                    'group_means': [group.mean() for group in groups]
                }
                
                # Post-hoc analysis if significant
                if p_value < self.alpha:
                    results[var]['post_hoc'] = self._tukey_hsd_analysis(groups, group_names)
        
        return results
    
    def regression_diagnostics(self, 
                             X: pd.DataFrame, 
                             y: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive regression diagnostics.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with diagnostic results
        """
        # Add constant for statsmodels
        X_with_const = sm.add_constant(X)
        
        # Fit OLS model
        model = sm.OLS(y, X_with_const).fit()
        
        results = {
            'model_summary': str(model.summary()),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic
        }
        
        # Residual analysis
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # Normality of residuals
        jb_stat, jb_pvalue = jarque_bera(residuals)
        results['residual_normality'] = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > self.alpha
        }
        
        # Homoscedasticity tests
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_with_const)
        results['homoscedasticity'] = {
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_pvalue': bp_pvalue,
            'is_homoscedastic': bp_pvalue > self.alpha
        }
        
        # Autocorrelation
        dw_stat = durbin_watson(residuals)
        results['autocorrelation'] = {
            'durbin_watson': dw_stat,
            'interpretation': self._interpret_durbin_watson(dw_stat)
        }
        
        # Multicollinearity (VIF)
        if X.shape[1] > 1:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                              for i in range(X.shape[1])]
            results['multicollinearity'] = {
                'vif_scores': vif_data.to_dict('records'),
                'high_vif_variables': vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
            }
        
        return results
    
    def time_series_tests(self, ts: pd.Series) -> Dict[str, Any]:
        """
        Perform time series stationarity tests.
        
        Args:
            ts: Time series data
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(ts.dropna())
        results['adf_test'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < self.alpha
        }
        
        # KPSS test
        kpss_result = kpss(ts.dropna())
        results['kpss_test'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > self.alpha
        }
        
        return results
    
    def _find_significant_correlations(self, df: pd.DataFrame, method: str) -> List[Dict]:
        """Find statistically significant correlations."""
        significant_corrs = []
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # Avoid duplicates
                    if method == 'pearson':
                        corr, p_value = pearsonr(df[col1], df[col2])
                    elif method == 'spearman':
                        corr, p_value = spearmanr(df[col1], df[col2])
                    elif method == 'kendall':
                        corr, p_value = kendalltau(df[col1], df[col2])
                    
                    if p_value < self.alpha:
                        significant_corrs.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr,
                            'p_value': p_value,
                            'strength': self._interpret_correlation_strength(abs(corr))
                        })
        
        return sorted(significant_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _find_strongest_correlation(self, corr_matrix: pd.DataFrame) -> Dict:
        """Find the strongest correlation in a correlation matrix."""
        # Create a copy and set diagonal to 0
        corr_copy = corr_matrix.copy()
        np.fill_diagonal(corr_copy.values, 0)
        
        # Find the maximum absolute correlation
        max_corr = corr_copy.abs().max().max()
        max_loc = corr_copy.abs().stack().idxmax()
        
        return {
            'variables': list(max_loc),
            'correlation': corr_matrix.loc[max_loc],
            'absolute_correlation': max_corr,
            'strength': self._interpret_correlation_strength(max_corr)
        }
    
    def _calculate_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation_strength(self, corr: float) -> str:
        """Interpret correlation strength."""
        if corr < 0.1:
            return "negligible"
        elif corr < 0.3:
            return "weak"
        elif corr < 0.5:
            return "moderate"
        elif corr < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def _interpret_durbin_watson(self, dw: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw < 1.5:
            return "positive autocorrelation"
        elif dw > 2.5:
            return "negative autocorrelation"
        else:
            return "no autocorrelation"
    
    def _tukey_hsd_analysis(self, groups: List, group_names: List) -> Dict:
        """Perform Tukey HSD post-hoc analysis."""
        # This is a simplified implementation
        # For full Tukey HSD, consider using statsmodels.stats.multicomp
        results = {}
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i < j:
                    stat, p_value = ttest_ind(group1, group2)
                    results[f"{group_names[i]}_vs_{group_names[j]}"] = {
                        'mean_diff': group1.mean() - group2.mean(),
                        't_statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha
                    }
        
        return results
    
    def comprehensive_analysis_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive statistical analysis report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_shape': df.shape,
            'analysis_results': {}
        }
        
        # Correlation analysis
        try:
            corr_results = self.correlation_analysis(df, method='all', return_matrix=True)
            report['analysis_results']['correlations'] = corr_results
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
        
        # Normality tests for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        normality_results = {}
        
        for col in numeric_cols:
            try:
                normality_results[col] = self.normality_tests(df[col])
            except Exception as e:
                logger.warning(f"Normality test failed for {col}: {e}")
        
        report['analysis_results']['normality_tests'] = normality_results
        
        return report