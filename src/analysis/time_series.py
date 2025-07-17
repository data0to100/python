"""
Comprehensive Time Series Analysis Module
Advanced time series analysis, forecasting, and decomposition utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta

# Statistical libraries
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Prophet for forecasting (if available)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis toolkit.
    """
    
    def __init__(self, ts: pd.Series, date_column: Optional[str] = None):
        """
        Initialize TimeSeriesAnalyzer.
        
        Args:
            ts: Time series data (should have datetime index)
            date_column: Name of date column if not in index
        """
        self.original_ts = ts.copy()
        
        # Ensure datetime index
        if date_column and date_column in ts.index.names:
            self.ts = ts.copy()
        elif isinstance(ts.index, pd.DatetimeIndex):
            self.ts = ts.copy()
        else:
            # Try to convert index to datetime
            try:
                self.ts = ts.copy()
                self.ts.index = pd.to_datetime(self.ts.index)
            except:
                raise ValueError("Could not convert index to datetime. Please provide proper datetime index.")
        
        # Sort by index
        self.ts = self.ts.sort_index()
        
        # Basic properties
        self.frequency = self._infer_frequency()
        self.n_periods = len(self.ts)
        
    def _infer_frequency(self) -> Optional[str]:
        """Infer the frequency of the time series."""
        try:
            return pd.infer_freq(self.ts.index)
        except:
            return None
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic time series statistics.
        
        Returns:
            Dictionary with basic statistics
        """
        stats_dict = {
            'count': self.ts.count(),
            'mean': self.ts.mean(),
            'std': self.ts.std(),
            'min': self.ts.min(),
            'max': self.ts.max(),
            'median': self.ts.median(),
            'skewness': self.ts.skew(),
            'kurtosis': self.ts.kurtosis(),
            'start_date': self.ts.index.min(),
            'end_date': self.ts.index.max(),
            'frequency': self.frequency,
            'missing_values': self.ts.isnull().sum(),
            'missing_percentage': (self.ts.isnull().sum() / len(self.ts)) * 100
        }
        
        # Time span
        time_span = self.ts.index.max() - self.ts.index.min()
        stats_dict['time_span_days'] = time_span.days
        
        return stats_dict
    
    def stationarity_tests(self) -> Dict[str, Any]:
        """
        Perform comprehensive stationarity tests.
        
        Returns:
            Dictionary with test results
        """
        results = {}
        clean_ts = self.ts.dropna()
        
        if len(clean_ts) < 10:
            return {'error': 'Insufficient data for stationarity tests'}
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(clean_ts, autolag='AIC')
            results['adf_test'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'lags_used': adf_result[2],
                'n_observations': adf_result[3],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05,
                'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
            }
        except Exception as e:
            results['adf_test'] = {'error': str(e)}
        
        # KPSS test
        try:
            kpss_result = kpss(clean_ts, regression='c')
            results['kpss_test'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'lags_used': kpss_result[2],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05,
                'interpretation': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
            }
        except Exception as e:
            results['kpss_test'] = {'error': str(e)}
        
        return results
    
    def seasonal_decomposition(self, 
                             model: str = 'additive',
                             period: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform seasonal decomposition of time series.
        
        Args:
            model: Type of decomposition ('additive' or 'multiplicative')
            period: Period for decomposition (auto-detected if None)
            
        Returns:
            Dictionary with decomposition components
        """
        clean_ts = self.ts.dropna()
        
        if len(clean_ts) < 24:  # Need at least 2 complete periods
            return {'error': 'Insufficient data for seasonal decomposition'}
        
        # Auto-detect period if not provided
        if period is None:
            if self.frequency:
                freq_map = {
                    'D': 365,  # Daily -> yearly seasonality
                    'W': 52,   # Weekly -> yearly seasonality
                    'M': 12,   # Monthly -> yearly seasonality
                    'Q': 4,    # Quarterly -> yearly seasonality
                    'H': 24,   # Hourly -> daily seasonality
                }
                period = freq_map.get(self.frequency, 12)
            else:
                period = min(12, len(clean_ts) // 2)
        
        try:
            decomposition = seasonal_decompose(clean_ts, model=model, period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'original': clean_ts,
                'model': model,
                'period': period,
                'seasonal_strength': self._calculate_seasonal_strength(decomposition),
                'trend_strength': self._calculate_trend_strength(decomposition)
            }
        except Exception as e:
            return {'error': f'Decomposition failed: {str(e)}'}
    
    def autocorrelation_analysis(self, lags: int = 40) -> Dict[str, Any]:
        """
        Perform autocorrelation and partial autocorrelation analysis.
        
        Args:
            lags: Number of lags to calculate
            
        Returns:
            Dictionary with ACF and PACF results
        """
        clean_ts = self.ts.dropna()
        
        if len(clean_ts) < lags + 10:
            lags = max(1, len(clean_ts) // 3)
        
        try:
            # Calculate ACF and PACF
            acf_values, acf_confint = acf(clean_ts, nlags=lags, alpha=0.05)
            pacf_values, pacf_confint = pacf(clean_ts, nlags=lags, alpha=0.05)
            
            return {
                'acf': {
                    'values': acf_values,
                    'confidence_intervals': acf_confint,
                    'significant_lags': self._find_significant_lags(acf_values, acf_confint)
                },
                'pacf': {
                    'values': pacf_values,
                    'confidence_intervals': pacf_confint,
                    'significant_lags': self._find_significant_lags(pacf_values, pacf_confint)
                },
                'ljung_box_test': self._ljung_box_test(clean_ts)
            }
        except Exception as e:
            return {'error': f'Autocorrelation analysis failed: {str(e)}'}
    
    def arima_modeling(self, 
                      order: Optional[Tuple[int, int, int]] = None,
                      seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                      auto_select: bool = True) -> Dict[str, Any]:
        """
        Fit ARIMA/SARIMA models to the time series.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_select: Whether to automatically select best parameters
            
        Returns:
            Dictionary with model results
        """
        clean_ts = self.ts.dropna()
        
        if len(clean_ts) < 20:
            return {'error': 'Insufficient data for ARIMA modeling'}
        
        results = {}
        
        try:
            if auto_select:
                # Simple grid search for best parameters
                best_aic = np.inf
                best_order = None
                best_model = None
                
                # Test different orders
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                model = ARIMA(clean_ts, order=(p, d, q))
                                fitted_model = model.fit()
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_order = (p, d, q)
                                    best_model = fitted_model
                            except:
                                continue
                
                if best_model is not None:
                    results['best_arima'] = {
                        'order': best_order,
                        'aic': best_aic,
                        'bic': best_model.bic,
                        'model_summary': str(best_model.summary()),
                        'residuals': best_model.resid,
                        'fitted_values': best_model.fittedvalues
                    }
            
            # Fit specified model if order is given
            if order is not None:
                if seasonal_order is not None:
                    model = SARIMAX(clean_ts, order=order, seasonal_order=seasonal_order)
                else:
                    model = ARIMA(clean_ts, order=order)
                
                fitted_model = model.fit()
                
                results['specified_model'] = {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'model_summary': str(fitted_model.summary()),
                    'residuals': fitted_model.resid,
                    'fitted_values': fitted_model.fittedvalues
                }
        
        except Exception as e:
            results['error'] = f'ARIMA modeling failed: {str(e)}'
        
        return results
    
    def exponential_smoothing(self, 
                            trend: Optional[str] = None,
                            seasonal: Optional[str] = None,
                            seasonal_periods: Optional[int] = None) -> Dict[str, Any]:
        """
        Fit exponential smoothing models.
        
        Args:
            trend: Trend component ('add', 'mul', None)
            seasonal: Seasonal component ('add', 'mul', None)
            seasonal_periods: Number of periods in season
            
        Returns:
            Dictionary with model results
        """
        clean_ts = self.ts.dropna()
        
        if len(clean_ts) < 10:
            return {'error': 'Insufficient data for exponential smoothing'}
        
        try:
            # Auto-detect seasonal periods if not provided
            if seasonal is not None and seasonal_periods is None:
                if self.frequency == 'D':
                    seasonal_periods = 7  # Weekly seasonality
                elif self.frequency == 'M':
                    seasonal_periods = 12  # Yearly seasonality
                else:
                    seasonal_periods = min(12, len(clean_ts) // 3)
            
            model = ExponentialSmoothing(
                clean_ts,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            
            fitted_model = model.fit()
            
            return {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': clean_ts - fitted_model.fittedvalues,
                'parameters': fitted_model.params,
                'model_summary': str(fitted_model.summary())
            }
        
        except Exception as e:
            return {'error': f'Exponential smoothing failed: {str(e)}'}
    
    def forecast(self, 
                steps: int = 30,
                method: str = 'auto',
                confidence_intervals: bool = True) -> Dict[str, Any]:
        """
        Generate forecasts using various methods.
        
        Args:
            steps: Number of steps to forecast
            method: Forecasting method ('arima', 'exponential', 'prophet', 'auto')
            confidence_intervals: Whether to include confidence intervals
            
        Returns:
            Dictionary with forecast results
        """
        clean_ts = self.ts.dropna()
        
        if len(clean_ts) < 20:
            return {'error': 'Insufficient data for forecasting'}
        
        results = {}
        
        # ARIMA forecast
        if method in ['arima', 'auto']:
            try:
                arima_result = self.arima_modeling(auto_select=True)
                if 'best_arima' in arima_result:
                    # Refit the best model and forecast
                    best_order = arima_result['best_arima']['order']
                    model = ARIMA(clean_ts, order=best_order)
                    fitted_model = model.fit()
                    
                    forecast_result = fitted_model.forecast(steps=steps)
                    if confidence_intervals:
                        forecast_ci = fitted_model.get_forecast(steps=steps).conf_int()
                        results['arima_forecast'] = {
                            'forecast': forecast_result,
                            'confidence_intervals': forecast_ci,
                            'model_order': best_order
                        }
                    else:
                        results['arima_forecast'] = {
                            'forecast': forecast_result,
                            'model_order': best_order
                        }
            except Exception as e:
                results['arima_forecast'] = {'error': str(e)}
        
        # Prophet forecast (if available)
        if method in ['prophet', 'auto'] and PROPHET_AVAILABLE:
            try:
                prophet_forecast = self._prophet_forecast(clean_ts, steps)
                results['prophet_forecast'] = prophet_forecast
            except Exception as e:
                results['prophet_forecast'] = {'error': str(e)}
        
        # Simple exponential smoothing
        if method in ['exponential', 'auto']:
            try:
                exp_result = self.exponential_smoothing()
                if 'error' not in exp_result:
                    # Simple forecast using last fitted value
                    last_fitted = exp_result['fitted_values'].iloc[-1]
                    simple_forecast = [last_fitted] * steps
                    results['exponential_forecast'] = {
                        'forecast': pd.Series(simple_forecast),
                        'method': 'simple_exponential'
                    }
            except Exception as e:
                results['exponential_forecast'] = {'error': str(e)}
        
        return results
    
    def _prophet_forecast(self, ts: pd.Series, steps: int) -> Dict[str, Any]:
        """Generate forecast using Prophet."""
        if not PROPHET_AVAILABLE:
            return {'error': 'Prophet not available'}
        
        # Prepare data for Prophet
        df = ts.reset_index()
        df.columns = ['ds', 'y']
        
        # Fit Prophet model
        model = Prophet()
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        
        return {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps),
            'full_forecast': forecast,
            'components': model.predict(future)[['ds', 'trend', 'seasonal', 'yearly']].tail(steps)
        }
    
    def _calculate_seasonal_strength(self, decomposition) -> float:
        """Calculate seasonal strength metric."""
        try:
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            return seasonal_var / (seasonal_var + residual_var)
        except:
            return 0.0
    
    def _calculate_trend_strength(self, decomposition) -> float:
        """Calculate trend strength metric."""
        try:
            trend_var = np.var(decomposition.trend.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            return trend_var / (trend_var + residual_var)
        except:
            return 0.0
    
    def _find_significant_lags(self, values: np.ndarray, confint: np.ndarray) -> List[int]:
        """Find statistically significant lags."""
        significant_lags = []
        
        for i, (value, ci) in enumerate(zip(values[1:], confint[1:]), 1):
            if value < ci[0] or value > ci[1]:
                significant_lags.append(i)
        
        return significant_lags
    
    def _ljung_box_test(self, ts: pd.Series, lags: int = 10) -> Dict[str, Any]:
        """Perform Ljung-Box test for autocorrelation in residuals."""
        try:
            result = acorr_ljungbox(ts, lags=lags, return_df=True)
            return {
                'statistics': result['lb_stat'].tolist(),
                'p_values': result['lb_pvalue'].tolist(),
                'significant_autocorrelation': any(result['lb_pvalue'] < 0.05)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def plot_time_series(self, 
                        include_decomposition: bool = False,
                        include_acf_pacf: bool = False) -> go.Figure:
        """
        Create comprehensive time series plots.
        
        Args:
            include_decomposition: Whether to include decomposition plots
            include_acf_pacf: Whether to include ACF/PACF plots
            
        Returns:
            Plotly figure
        """
        if include_decomposition and include_acf_pacf:
            subplot_titles = ['Original Time Series', 'Trend', 'Seasonal', 'Residual', 'ACF', 'PACF']
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=subplot_titles,
                vertical_spacing=0.05
            )
        elif include_decomposition:
            subplot_titles = ['Original Time Series', 'Trend', 'Seasonal', 'Residual']
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.05
            )
        elif include_acf_pacf:
            subplot_titles = ['Original Time Series', 'ACF', 'PACF']
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1
            )
        else:
            fig = go.Figure()
        
        # Original time series
        fig.add_trace(
            go.Scatter(
                x=self.ts.index,
                y=self.ts.values,
                mode='lines',
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        if include_decomposition:
            decomp = self.seasonal_decomposition()
            if 'error' not in decomp:
                # Trend
                fig.add_trace(
                    go.Scatter(
                        x=decomp['trend'].index,
                        y=decomp['trend'].values,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Seasonal
                fig.add_trace(
                    go.Scatter(
                        x=decomp['seasonal'].index,
                        y=decomp['seasonal'].values,
                        mode='lines',
                        name='Seasonal',
                        line=dict(color='green'),
                        showlegend=False
                    ),
                    row=3, col=1
                )
                
                # Residual
                fig.add_trace(
                    go.Scatter(
                        x=decomp['residual'].index,
                        y=decomp['residual'].values,
                        mode='lines',
                        name='Residual',
                        line=dict(color='orange'),
                        showlegend=False
                    ),
                    row=4, col=1
                )
        
        if include_acf_pacf:
            acf_result = self.autocorrelation_analysis()
            if 'error' not in acf_result:
                # ACF
                lags = range(len(acf_result['acf']['values']))
                fig.add_trace(
                    go.Bar(
                        x=list(lags),
                        y=acf_result['acf']['values'],
                        name='ACF',
                        showlegend=False
                    ),
                    row=2 if not include_decomposition else 2,
                    col=2 if include_decomposition else 1
                )
                
                # PACF
                fig.add_trace(
                    go.Bar(
                        x=list(lags),
                        y=acf_result['pacf']['values'],
                        name='PACF',
                        showlegend=False
                    ),
                    row=3 if not include_decomposition else 3,
                    col=2 if include_decomposition else 1
                )
        
        fig.update_layout(
            title='Time Series Analysis',
            height=800 if include_decomposition or include_acf_pacf else 400,
            showlegend=True
        )
        
        return fig
    
    def comprehensive_analysis_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive time series analysis report.
        
        Returns:
            Dictionary with all analysis results
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'basic_statistics': self.basic_statistics(),
            'stationarity_tests': self.stationarity_tests(),
            'seasonal_decomposition': self.seasonal_decomposition(),
            'autocorrelation_analysis': self.autocorrelation_analysis(),
            'arima_modeling': self.arima_modeling(auto_select=True),
            'exponential_smoothing': self.exponential_smoothing(),
            'forecast': self.forecast(steps=30, method='auto')
        }
        
        return report