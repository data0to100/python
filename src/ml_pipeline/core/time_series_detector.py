"""
Intelligent Time Series Detection and Analysis

This module provides automatic detection and analysis of time series data:
- Automatic identification of time series vs cross-sectional data
- Detection of datetime columns and temporal patterns
- Seasonality and trend analysis using advanced statistical methods
- Autocorrelation and partial autocorrelation analysis
- Stationarity testing (ADF, KPSS, Phillips-Perron)
- Frequency detection and regularity assessment
- Time series decomposition and feature extraction
- Recommendation engine for appropriate modeling approaches
"""

import re
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest

# Time series analysis libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# Pattern detection
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Advanced time series libraries
try:
    import pymannkendall as mk
    MANNKENDALL_AVAILABLE = True
except ImportError:
    MANNKENDALL_AVAILABLE = False

try:
    from scipy.signal import find_peaks, periodogram
    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False

from ..utils.logger import get_logger, timing_decorator, error_handler


class TimeSeriesDetector:
    """
    Advanced time series detection and analysis system
    
    Features:
    - Automatic detection of temporal data patterns
    - Sophisticated datetime column identification
    - Statistical analysis of temporal dependencies
    - Seasonality and trend detection using multiple methods
    - Stationarity testing with multiple statistical tests
    - Frequency analysis and regularity assessment
    - Time series decomposition and feature extraction
    - Intelligent recommendations for modeling approaches
    """
    
    def __init__(self):
        """Initialize the TimeSeriesDetector"""
        self.logger = get_logger()
        
        # Analysis results storage
        self.analysis_results = {}
        self.datetime_columns = []
        self.is_time_series = False
        self.temporal_features = {}
        
        # Common datetime patterns for detection
        self.datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}-\w{3}-\d{4}',  # DD-MMM-YYYY
            r'\w{3}\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
        ]
    
    @timing_decorator("time_series_detection")
    @error_handler
    def analyze_temporal_structure(self, 
                                 data: pd.DataFrame,
                                 target_column: Optional[str] = None,
                                 datetime_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of temporal structure in data
        
        Args:
            data: Input DataFrame
            target_column: Target variable for time series analysis
            datetime_column: Explicit datetime column (optional)
        
        Returns:
            Dictionary containing complete temporal analysis results
        """
        self.logger.logger.info("Starting comprehensive temporal structure analysis")
        
        # Step 1: Detect datetime columns
        datetime_info = self._detect_datetime_columns(data, datetime_column)
        
        # Step 2: Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(data, datetime_info)
        
        # Step 3: Assess time series characteristics
        ts_characteristics = self._assess_time_series_characteristics(
            data, datetime_info, target_column
        )
        
        # Step 4: Perform statistical time series tests
        statistical_tests = self._perform_statistical_tests(
            data, datetime_info, target_column
        )
        
        # Step 5: Detect seasonality and trends
        seasonality_analysis = self._analyze_seasonality_and_trends(
            data, datetime_info, target_column
        )
        
        # Step 6: Frequency and regularity analysis
        frequency_analysis = self._analyze_frequency_and_regularity(
            data, datetime_info
        )
        
        # Step 7: Generate final classification and recommendations
        classification = self._classify_data_type(
            datetime_info, temporal_patterns, ts_characteristics, 
            statistical_tests, seasonality_analysis
        )
        
        # Compile comprehensive results
        self.analysis_results = {
            "datetime_detection": datetime_info,
            "temporal_patterns": temporal_patterns,
            "time_series_characteristics": ts_characteristics,
            "statistical_tests": statistical_tests,
            "seasonality_analysis": seasonality_analysis,
            "frequency_analysis": frequency_analysis,
            "classification": classification,
            "recommendations": self._generate_recommendations(classification)
        }
        
        # Update instance variables
        self.datetime_columns = datetime_info.get("detected_columns", [])
        self.is_time_series = classification.get("is_time_series", False)
        
        self.logger.logger.info(
            f"Temporal analysis completed. Data type: {'Time Series' if self.is_time_series else 'Cross-sectional'}"
        )
        
        return self.analysis_results
    
    def _detect_datetime_columns(self, 
                                data: pd.DataFrame,
                                explicit_column: Optional[str] = None) -> Dict[str, Any]:
        """Detect datetime columns using multiple strategies"""
        
        detected_columns = []
        column_analysis = {}
        
        # Strategy 1: Use explicitly provided column
        if explicit_column and explicit_column in data.columns:
            detected_columns.append(explicit_column)
            column_analysis[explicit_column] = {
                "detection_method": "explicit",
                "confidence": 1.0,
                "data_type": str(data[explicit_column].dtype)
            }
        
        # Strategy 2: Check pandas datetime types
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                if col not in detected_columns:
                    detected_columns.append(col)
                    column_analysis[col] = {
                        "detection_method": "pandas_dtype",
                        "confidence": 1.0,
                        "data_type": str(data[col].dtype)
                    }
        
        # Strategy 3: Pattern-based detection for string columns
        for col in data.columns:
            if col not in detected_columns and data[col].dtype == 'object':
                datetime_score = self._score_datetime_column(data[col])
                if datetime_score > 0.7:  # High confidence threshold
                    detected_columns.append(col)
                    column_analysis[col] = {
                        "detection_method": "pattern_based",
                        "confidence": datetime_score,
                        "data_type": "object"
                    }
        
        # Strategy 4: Name-based heuristics
        datetime_name_patterns = [
            r'.*date.*', r'.*time.*', r'.*timestamp.*', 
            r'.*created.*', r'.*modified.*', r'.*updated.*',
            r'.*year.*', r'.*month.*', r'.*day.*'
        ]
        
        for col in data.columns:
            if col not in detected_columns:
                col_lower = col.lower()
                for pattern in datetime_name_patterns:
                    if re.match(pattern, col_lower):
                        # Try to convert and validate
                        try:
                            pd.to_datetime(data[col].dropna().head(100))
                            detected_columns.append(col)
                            column_analysis[col] = {
                                "detection_method": "name_heuristic",
                                "confidence": 0.6,
                                "data_type": str(data[col].dtype)
                            }
                            break
                        except:
                            pass
        
        # Analyze detected datetime columns
        primary_datetime_column = None
        if detected_columns:
            # Select primary datetime column (highest confidence or most complete)
            best_column = max(
                detected_columns,
                key=lambda col: (
                    column_analysis[col]["confidence"],
                    -data[col].isnull().sum(),  # Prefer columns with fewer nulls
                    len(data[col].dropna().unique())  # Prefer columns with more unique values
                )
            )
            primary_datetime_column = best_column
        
        return {
            "detected_columns": detected_columns,
            "column_analysis": column_analysis,
            "primary_datetime_column": primary_datetime_column,
            "total_detected": len(detected_columns)
        }
    
    def _score_datetime_column(self, series: pd.Series, sample_size: int = 1000) -> float:
        """Score how likely a column contains datetime values"""
        
        # Sample data for efficiency
        sample_data = series.dropna().astype(str).head(sample_size)
        
        if len(sample_data) == 0:
            return 0.0
        
        total_score = 0.0
        pattern_matches = 0
        
        for value in sample_data:
            value_str = str(value).strip()
            
            # Check against datetime patterns
            for pattern in self.datetime_patterns:
                if re.search(pattern, value_str):
                    pattern_matches += 1
                    break
            
            # Try pandas datetime conversion
            try:
                pd.to_datetime(value_str)
                total_score += 1.0
            except:
                pass
        
        # Calculate confidence score
        conversion_score = total_score / len(sample_data)
        pattern_score = pattern_matches / len(sample_data)
        
        # Weighted average
        final_score = 0.7 * conversion_score + 0.3 * pattern_score
        
        return min(final_score, 1.0)
    
    def _analyze_temporal_patterns(self, 
                                 data: pd.DataFrame, 
                                 datetime_info: Dict) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        
        if not datetime_info["detected_columns"]:
            return {"has_temporal_patterns": False, "reason": "No datetime columns detected"}
        
        primary_col = datetime_info["primary_datetime_column"]
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data[primary_col]):
                datetime_series = pd.to_datetime(data[primary_col], errors='coerce')
            else:
                datetime_series = data[primary_col]
            
            # Remove NaT values
            datetime_series = datetime_series.dropna()
            
            if len(datetime_series) < 2:
                return {"has_temporal_patterns": False, "reason": "Insufficient datetime data"}
            
            # Sort datetime series
            datetime_series = datetime_series.sort_values()
            
            # Calculate time differences
            time_diffs = datetime_series.diff().dropna()
            
            # Analyze temporal characteristics
            temporal_analysis = {
                "has_temporal_patterns": True,
                "time_span": {
                    "start": datetime_series.min(),
                    "end": datetime_series.max(),
                    "duration": datetime_series.max() - datetime_series.min()
                },
                "time_differences": {
                    "min_diff": time_diffs.min(),
                    "max_diff": time_diffs.max(),
                    "mean_diff": time_diffs.mean(),
                    "std_diff": time_diffs.std()
                },
                "regularity": self._assess_temporal_regularity(time_diffs),
                "gaps": self._detect_temporal_gaps(datetime_series),
                "duplicates": datetime_series.duplicated().sum()
            }
            
            return temporal_analysis
            
        except Exception as e:
            return {
                "has_temporal_patterns": False, 
                "reason": f"Error analyzing temporal patterns: {e}"
            }
    
    def _assess_temporal_regularity(self, time_diffs: pd.Series) -> Dict[str, Any]:
        """Assess the regularity of temporal intervals"""
        
        if len(time_diffs) == 0:
            return {"is_regular": False, "reason": "No time differences"}
        
        # Convert to seconds for analysis
        diff_seconds = time_diffs.dt.total_seconds()
        
        # Calculate coefficient of variation
        cv = diff_seconds.std() / diff_seconds.mean() if diff_seconds.mean() > 0 else float('inf')
        
        # Detect common intervals
        mode_diff = diff_seconds.mode().iloc[0] if len(diff_seconds.mode()) > 0 else diff_seconds.median()
        
        # Classify regularity
        is_regular = cv < 0.1  # Less than 10% variation
        
        # Infer frequency
        inferred_frequency = self._infer_frequency(mode_diff)
        
        return {
            "is_regular": is_regular,
            "coefficient_of_variation": cv,
            "mode_interval_seconds": mode_diff,
            "inferred_frequency": inferred_frequency,
            "regularity_score": max(0, 1 - cv)  # Higher score = more regular
        }
    
    def _infer_frequency(self, mode_diff_seconds: float) -> str:
        """Infer frequency from mode time difference"""
        
        # Common frequency mappings (in seconds)
        frequencies = {
            1: "1S",  # Second
            60: "1T",  # Minute
            3600: "1H",  # Hour
            86400: "1D",  # Day
            604800: "1W",  # Week
            2629746: "1M",  # Month (average)
            31556952: "1Y"  # Year (average)
        }
        
        # Find closest frequency
        closest_freq = min(frequencies.keys(), key=lambda x: abs(x - mode_diff_seconds))
        
        # If close enough (within 10%), return the frequency
        if abs(closest_freq - mode_diff_seconds) / closest_freq < 0.1:
            return frequencies[closest_freq]
        
        # Custom frequency for other intervals
        if mode_diff_seconds < 60:
            return f"{int(mode_diff_seconds)}S"
        elif mode_diff_seconds < 3600:
            return f"{int(mode_diff_seconds/60)}T"
        elif mode_diff_seconds < 86400:
            return f"{int(mode_diff_seconds/3600)}H"
        else:
            return f"{int(mode_diff_seconds/86400)}D"
    
    def _detect_temporal_gaps(self, datetime_series: pd.Series) -> Dict[str, Any]:
        """Detect significant gaps in temporal data"""
        
        time_diffs = datetime_series.diff().dropna()
        
        if len(time_diffs) == 0:
            return {"has_gaps": False, "gap_count": 0}
        
        # Define gap threshold as 3 standard deviations above mean
        mean_diff = time_diffs.mean()
        std_diff = time_diffs.std()
        gap_threshold = mean_diff + 3 * std_diff
        
        # Identify gaps
        gaps = time_diffs[time_diffs > gap_threshold]
        
        gap_analysis = {
            "has_gaps": len(gaps) > 0,
            "gap_count": len(gaps),
            "largest_gap": gaps.max() if len(gaps) > 0 else pd.Timedelta(0),
            "gap_threshold": gap_threshold,
            "gap_locations": gaps.index.tolist() if len(gaps) > 0 else []
        }
        
        return gap_analysis
    
    def _assess_time_series_characteristics(self, 
                                          data: pd.DataFrame,
                                          datetime_info: Dict,
                                          target_column: Optional[str]) -> Dict[str, Any]:
        """Assess characteristics that indicate time series data"""
        
        characteristics = {
            "has_datetime_column": len(datetime_info["detected_columns"]) > 0,
            "datetime_column_count": len(datetime_info["detected_columns"]),
            "has_target_column": target_column is not None and target_column in data.columns,
            "temporal_ordering_likelihood": 0.0,
            "sequential_patterns": {}
        }
        
        if not characteristics["has_datetime_column"]:
            return characteristics
        
        primary_datetime = datetime_info["primary_datetime_column"]
        
        # Check if data appears to be ordered temporally
        if primary_datetime:
            try:
                datetime_col = pd.to_datetime(data[primary_datetime], errors='coerce')
                
                # Check if data is mostly sorted by datetime
                sorted_ratio = (datetime_col == datetime_col.sort_values()).mean()
                characteristics["temporal_ordering_likelihood"] = sorted_ratio
                
                # Analyze sequential patterns in target column
                if target_column and target_column in data.columns:
                    characteristics["sequential_patterns"] = self._analyze_sequential_patterns(
                        data, datetime_col, target_column
                    )
                
            except Exception as e:
                self.logger.logger.warning(f"Error assessing time series characteristics: {e}")
        
        return characteristics
    
    def _analyze_sequential_patterns(self, 
                                   data: pd.DataFrame,
                                   datetime_col: pd.Series,
                                   target_column: str) -> Dict[str, Any]:
        """Analyze sequential patterns in target variable"""
        
        try:
            # Create time-ordered dataset
            df_sorted = data.copy()
            df_sorted['datetime_parsed'] = datetime_col
            df_sorted = df_sorted.dropna(subset=['datetime_parsed', target_column])
            df_sorted = df_sorted.sort_values('datetime_parsed')
            
            if len(df_sorted) < 10:
                return {"insufficient_data": True}
            
            target_values = df_sorted[target_column]
            
            # Check if target is numeric
            if not pd.api.types.is_numeric_dtype(target_values):
                return {"non_numeric_target": True}
            
            # Calculate autocorrelation
            autocorr_lag1 = target_values.autocorr(lag=1) if len(target_values) > 1 else 0
            
            # Calculate rolling statistics
            rolling_mean = target_values.rolling(window=min(10, len(target_values)//2)).mean()
            rolling_std = target_values.rolling(window=min(10, len(target_values)//2)).std()
            
            # Trend analysis
            x = np.arange(len(target_values))
            y = target_values.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            sequential_analysis = {
                "autocorrelation_lag1": autocorr_lag1,
                "has_autocorrelation": abs(autocorr_lag1) > 0.1,
                "trend": {
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "has_trend": p_value < 0.05 and abs(r_value) > 0.1
                },
                "variability": {
                    "mean_rolling_std": rolling_std.mean(),
                    "std_rolling_std": rolling_std.std(),
                    "coefficient_of_variation": target_values.std() / target_values.mean() if target_values.mean() != 0 else 0
                }
            }
            
            return sequential_analysis
            
        except Exception as e:
            return {"error": f"Failed to analyze sequential patterns: {e}"}
    
    def _perform_statistical_tests(self, 
                                 data: pd.DataFrame,
                                 datetime_info: Dict,
                                 target_column: Optional[str]) -> Dict[str, Any]:
        """Perform statistical tests for time series characteristics"""
        
        test_results = {
            "tests_performed": [],
            "stationarity_tests": {},
            "autocorrelation_tests": {},
            "normality_tests": {}
        }
        
        if not target_column or target_column not in data.columns:
            return test_results
        
        if not datetime_info["detected_columns"]:
            return test_results
        
        try:
            # Prepare time series data
            primary_datetime = datetime_info["primary_datetime_column"]
            datetime_col = pd.to_datetime(data[primary_datetime], errors='coerce')
            
            df_ts = data.copy()
            df_ts['datetime_parsed'] = datetime_col
            df_ts = df_ts.dropna(subset=['datetime_parsed', target_column])
            df_ts = df_ts.sort_values('datetime_parsed')
            
            if len(df_ts) < 30:  # Need sufficient data for statistical tests
                return test_results
            
            target_series = df_ts[target_column]
            
            if not pd.api.types.is_numeric_dtype(target_series):
                return test_results
            
            # Stationarity tests
            test_results["stationarity_tests"] = self._perform_stationarity_tests(target_series)
            test_results["tests_performed"].append("stationarity")
            
            # Autocorrelation tests
            test_results["autocorrelation_tests"] = self._perform_autocorrelation_tests(target_series)
            test_results["tests_performed"].append("autocorrelation")
            
            # Normality tests
            test_results["normality_tests"] = self._perform_normality_tests(target_series)
            test_results["tests_performed"].append("normality")
            
        except Exception as e:
            test_results["error"] = f"Statistical tests failed: {e}"
        
        return test_results
    
    def _perform_stationarity_tests(self, series: pd.Series) -> Dict[str, Any]:
        """Perform stationarity tests (ADF, KPSS)"""
        
        stationarity_results = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            stationarity_results["adf"] = {
                "test_statistic": adf_result[0],
                "p_value": adf_result[1],
                "critical_values": adf_result[4],
                "is_stationary": adf_result[1] < 0.05
            }
            
            # KPSS test
            kpss_result = kpss(series.dropna(), regression='c')
            stationarity_results["kpss"] = {
                "test_statistic": kpss_result[0],
                "p_value": kpss_result[1],
                "critical_values": kpss_result[3],
                "is_stationary": kpss_result[1] > 0.05  # Null hypothesis is stationarity
            }
            
            # Combined assessment
            adf_stationary = stationarity_results["adf"]["is_stationary"]
            kpss_stationary = stationarity_results["kpss"]["is_stationary"]
            
            if adf_stationary and kpss_stationary:
                stationarity_results["overall_assessment"] = "stationary"
            elif not adf_stationary and not kpss_stationary:
                stationarity_results["overall_assessment"] = "non_stationary"
            else:
                stationarity_results["overall_assessment"] = "inconclusive"
                
        except Exception as e:
            stationarity_results["error"] = f"Stationarity tests failed: {e}"
        
        return stationarity_results
    
    def _perform_autocorrelation_tests(self, series: pd.Series) -> Dict[str, Any]:
        """Perform autocorrelation analysis"""
        
        autocorr_results = {}
        
        try:
            series_clean = series.dropna()
            
            if len(series_clean) < 20:
                return {"error": "Insufficient data for autocorrelation tests"}
            
            # Calculate ACF and PACF
            max_lags = min(20, len(series_clean) // 4)
            
            acf_values, acf_confint = acf(series_clean, nlags=max_lags, alpha=0.05)
            pacf_values, pacf_confint = pacf(series_clean, nlags=max_lags, alpha=0.05)
            
            autocorr_results["acf"] = {
                "values": acf_values.tolist(),
                "confidence_intervals": acf_confint.tolist(),
                "significant_lags": [i for i, val in enumerate(acf_values[1:], 1) 
                                   if abs(val) > 2/np.sqrt(len(series_clean))]
            }
            
            autocorr_results["pacf"] = {
                "values": pacf_values.tolist(),
                "confidence_intervals": pacf_confint.tolist(),
                "significant_lags": [i for i, val in enumerate(pacf_values[1:], 1) 
                                   if abs(val) > 2/np.sqrt(len(series_clean))]
            }
            
            # Ljung-Box test for independence
            lb_result = acorr_ljungbox(series_clean, lags=min(10, len(series_clean)//5), return_df=True)
            autocorr_results["ljung_box"] = {
                "statistics": lb_result['lb_stat'].tolist(),
                "p_values": lb_result['lb_pvalue'].tolist(),
                "has_autocorrelation": any(lb_result['lb_pvalue'] < 0.05)
            }
            
        except Exception as e:
            autocorr_results["error"] = f"Autocorrelation tests failed: {e}"
        
        return autocorr_results
    
    def _perform_normality_tests(self, series: pd.Series) -> Dict[str, Any]:
        """Perform normality tests"""
        
        normality_results = {}
        
        try:
            series_clean = series.dropna()
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = jarque_bera(series_clean)
            normality_results["jarque_bera"] = {
                "statistic": jb_stat,
                "p_value": jb_pvalue,
                "is_normal": jb_pvalue > 0.05
            }
            
            # Shapiro-Wilk test (for smaller samples)
            if len(series_clean) <= 5000:
                sw_stat, sw_pvalue = stats.shapiro(series_clean)
                normality_results["shapiro_wilk"] = {
                    "statistic": sw_stat,
                    "p_value": sw_pvalue,
                    "is_normal": sw_pvalue > 0.05
                }
            
            # D'Agostino's normality test
            da_stat, da_pvalue = normaltest(series_clean)
            normality_results["dagostino"] = {
                "statistic": da_stat,
                "p_value": da_pvalue,
                "is_normal": da_pvalue > 0.05
            }
            
        except Exception as e:
            normality_results["error"] = f"Normality tests failed: {e}"
        
        return normality_results
    
    def _analyze_seasonality_and_trends(self, 
                                      data: pd.DataFrame,
                                      datetime_info: Dict,
                                      target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze seasonality and trend patterns"""
        
        seasonality_results = {
            "has_seasonality": False,
            "has_trend": False,
            "decomposition": None,
            "seasonal_patterns": {}
        }
        
        if not target_column or not datetime_info["detected_columns"]:
            return seasonality_results
        
        try:
            # Prepare time series
            primary_datetime = datetime_info["primary_datetime_column"]
            datetime_col = pd.to_datetime(data[primary_datetime], errors='coerce')
            
            df_ts = data.copy()
            df_ts['datetime_parsed'] = datetime_col
            df_ts = df_ts.dropna(subset=['datetime_parsed', target_column])
            df_ts = df_ts.sort_values('datetime_parsed')
            df_ts = df_ts.set_index('datetime_parsed')
            
            if len(df_ts) < 24:  # Need at least 2 seasonal cycles
                return seasonality_results
            
            target_series = df_ts[target_column]
            
            if not pd.api.types.is_numeric_dtype(target_series):
                return seasonality_results
            
            # Seasonal decomposition
            try:
                # Try different seasonal periods
                for period in [12, 24, 7, 4]:  # Monthly, daily, weekly, quarterly
                    if len(target_series) >= 2 * period:
                        decomposition = seasonal_decompose(
                            target_series, 
                            model='additive', 
                            period=period,
                            extrapolate_trend='freq'
                        )
                        
                        # Check seasonality strength
                        seasonal_strength = np.var(decomposition.seasonal) / np.var(target_series)
                        
                        if seasonal_strength > 0.1:  # Significant seasonality
                            seasonality_results["has_seasonality"] = True
                            seasonality_results["decomposition"] = {
                                "period": period,
                                "seasonal_strength": seasonal_strength,
                                "trend_strength": np.var(decomposition.trend.dropna()) / np.var(target_series)
                            }
                            break
                            
            except Exception as e:
                self.logger.logger.warning(f"Seasonal decomposition failed: {e}")
            
            # Trend analysis using Mann-Kendall test
            if MANNKENDALL_AVAILABLE:
                try:
                    mk_result = mk.original_test(target_series.dropna())
                    seasonality_results["trend_analysis"] = {
                        "mann_kendall_trend": mk_result.trend,
                        "mann_kendall_p_value": mk_result.p,
                        "has_significant_trend": mk_result.p < 0.05
                    }
                    seasonality_results["has_trend"] = mk_result.p < 0.05
                except Exception as e:
                    self.logger.logger.warning(f"Mann-Kendall test failed: {e}")
            
            # Frequency domain analysis
            if SCIPY_SIGNAL_AVAILABLE and len(target_series) > 50:
                try:
                    freqs, psd = periodogram(target_series.dropna())
                    
                    # Find dominant frequencies
                    peaks, _ = find_peaks(psd, height=np.mean(psd))
                    
                    if len(peaks) > 0:
                        dominant_freq = freqs[peaks[np.argmax(psd[peaks])]]
                        seasonality_results["frequency_analysis"] = {
                            "dominant_frequency": dominant_freq,
                            "dominant_period": 1/dominant_freq if dominant_freq > 0 else None,
                            "peak_count": len(peaks)
                        }
                        
                except Exception as e:
                    self.logger.logger.warning(f"Frequency analysis failed: {e}")
            
        except Exception as e:
            seasonality_results["error"] = f"Seasonality analysis failed: {e}"
        
        return seasonality_results
    
    def _analyze_frequency_and_regularity(self, 
                                        data: pd.DataFrame,
                                        datetime_info: Dict) -> Dict[str, Any]:
        """Analyze frequency and regularity of time series data"""
        
        if not datetime_info["detected_columns"]:
            return {"has_regular_frequency": False, "reason": "No datetime columns"}
        
        primary_datetime = datetime_info["primary_datetime_column"]
        
        try:
            datetime_col = pd.to_datetime(data[primary_datetime], errors='coerce')
            datetime_clean = datetime_col.dropna().sort_values()
            
            if len(datetime_clean) < 3:
                return {"has_regular_frequency": False, "reason": "Insufficient datetime data"}
            
            # Infer frequency using pandas
            try:
                freq = pd.infer_freq(datetime_clean)
                has_pandas_freq = freq is not None
            except:
                freq = None
                has_pandas_freq = False
            
            # Manual frequency analysis
            time_diffs = datetime_clean.diff().dropna()
            mode_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            
            # Calculate regularity metrics
            cv = (time_diffs.dt.total_seconds().std() / 
                  time_diffs.dt.total_seconds().mean() if time_diffs.dt.total_seconds().mean() > 0 else float('inf'))
            
            frequency_analysis = {
                "has_regular_frequency": has_pandas_freq or cv < 0.1,
                "inferred_frequency": freq,
                "mode_interval": mode_diff,
                "coefficient_of_variation": cv,
                "unique_intervals": len(time_diffs.unique()),
                "total_intervals": len(time_diffs),
                "regularity_score": max(0, 1 - cv)
            }
            
            return frequency_analysis
            
        except Exception as e:
            return {"has_regular_frequency": False, "error": f"Frequency analysis failed: {e}"}
    
    def _classify_data_type(self, 
                          datetime_info: Dict,
                          temporal_patterns: Dict,
                          ts_characteristics: Dict,
                          statistical_tests: Dict,
                          seasonality_analysis: Dict) -> Dict[str, Any]:
        """Classify data as time series or cross-sectional based on analysis"""
        
        # Scoring system for time series likelihood
        ts_score = 0.0
        evidence = []
        
        # DateTime column presence (strong indicator)
        if datetime_info["total_detected"] > 0:
            ts_score += 3.0
            evidence.append("Has datetime column(s)")
        
        # Temporal patterns
        if temporal_patterns.get("has_temporal_patterns", False):
            ts_score += 2.0
            evidence.append("Has temporal patterns")
            
            # Regular intervals
            regularity = temporal_patterns.get("regularity", {})
            if regularity.get("is_regular", False):
                ts_score += 1.5
                evidence.append("Regular time intervals")
        
        # Sequential characteristics
        if ts_characteristics.get("temporal_ordering_likelihood", 0) > 0.8:
            ts_score += 1.0
            evidence.append("Data appears temporally ordered")
        
        # Autocorrelation evidence
        autocorr_tests = statistical_tests.get("autocorrelation_tests", {})
        if autocorr_tests.get("ljung_box", {}).get("has_autocorrelation", False):
            ts_score += 2.0
            evidence.append("Significant autocorrelation detected")
        
        # Seasonality evidence
        if seasonality_analysis.get("has_seasonality", False):
            ts_score += 1.5
            evidence.append("Seasonal patterns detected")
        
        # Trend evidence
        if seasonality_analysis.get("has_trend", False):
            ts_score += 1.0
            evidence.append("Trend patterns detected")
        
        # Non-stationarity (common in time series)
        stationarity_tests = statistical_tests.get("stationarity_tests", {})
        if stationarity_tests.get("overall_assessment") == "non_stationary":
            ts_score += 0.5
            evidence.append("Non-stationary data")
        
        # Classification decision
        is_time_series = ts_score >= 3.0  # Threshold for time series classification
        
        confidence = min(ts_score / 8.0, 1.0)  # Normalize to 0-1 scale
        
        classification = {
            "is_time_series": is_time_series,
            "data_type": "time_series" if is_time_series else "cross_sectional",
            "confidence": confidence,
            "ts_score": ts_score,
            "evidence": evidence,
            "recommendation": self._get_modeling_recommendation(is_time_series, seasonality_analysis, statistical_tests)
        }
        
        return classification
    
    def _get_modeling_recommendation(self, 
                                   is_time_series: bool,
                                   seasonality_analysis: Dict,
                                   statistical_tests: Dict) -> Dict[str, Any]:
        """Generate modeling recommendations based on time series characteristics"""
        
        if not is_time_series:
            return {
                "model_type": "cross_sectional",
                "recommended_models": ["random_forest", "xgboost", "linear_regression", "svm"],
                "approach": "standard_ml"
            }
        
        # Time series specific recommendations
        recommendations = {
            "model_type": "time_series",
            "recommended_models": [],
            "approach": "time_series_modeling"
        }
        
        has_seasonality = seasonality_analysis.get("has_seasonality", False)
        has_trend = seasonality_analysis.get("has_trend", False)
        stationarity = statistical_tests.get("stationarity_tests", {}).get("overall_assessment")
        
        # Model recommendations based on characteristics
        if has_seasonality:
            recommendations["recommended_models"].extend(["prophet", "seasonal_arima", "seasonal_decompose"])
        
        if has_trend or stationarity == "non_stationary":
            recommendations["recommended_models"].extend(["arima", "prophet"])
        
        if stationarity == "stationary":
            recommendations["recommended_models"].extend(["ar", "ma", "arma"])
        
        # Always include robust options
        recommendations["recommended_models"].extend(["prophet", "lstm", "gru"])
        
        # Remove duplicates and prioritize
        recommendations["recommended_models"] = list(dict.fromkeys(recommendations["recommended_models"]))
        
        return recommendations
    
    def _generate_recommendations(self, classification: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        if classification["is_time_series"]:
            recommendations.append("Data identified as time series - use temporal modeling approaches")
            
            model_rec = classification.get("recommendation", {})
            if model_rec.get("recommended_models"):
                recommendations.append(
                    f"Recommended models: {', '.join(model_rec['recommended_models'][:3])}"
                )
            
            # Analysis-specific recommendations
            if self.analysis_results.get("seasonality_analysis", {}).get("has_seasonality"):
                recommendations.append("Seasonality detected - consider seasonal models or decomposition")
            
            if self.analysis_results.get("statistical_tests", {}).get("stationarity_tests", {}).get("overall_assessment") == "non_stationary":
                recommendations.append("Data appears non-stationary - consider differencing or trend modeling")
            
            freq_analysis = self.analysis_results.get("frequency_analysis", {})
            if not freq_analysis.get("has_regular_frequency", True):
                recommendations.append("Irregular time intervals detected - consider resampling or interpolation")
                
        else:
            recommendations.append("Data identified as cross-sectional - use standard ML approaches")
            recommendations.append("Consider features like random forest, gradient boosting, or neural networks")
        
        # Data quality recommendations
        temporal_patterns = self.analysis_results.get("temporal_patterns", {})
        if temporal_patterns.get("gaps", {}).get("has_gaps", False):
            recommendations.append("Temporal gaps detected - consider handling missing periods")
        
        return recommendations
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the temporal analysis"""
        
        if not self.analysis_results:
            return {"error": "No analysis results available. Run analyze_temporal_structure() first."}
        
        classification = self.analysis_results.get("classification", {})
        
        summary = {
            "data_type": classification.get("data_type", "unknown"),
            "is_time_series": classification.get("is_time_series", False),
            "confidence": classification.get("confidence", 0.0),
            "datetime_columns": self.datetime_columns,
            "key_findings": [],
            "recommendations": classification.get("recommendation", {})
        }
        
        # Add key findings
        if self.analysis_results.get("seasonality_analysis", {}).get("has_seasonality"):
            summary["key_findings"].append("Seasonality detected")
        
        if self.analysis_results.get("seasonality_analysis", {}).get("has_trend"):
            summary["key_findings"].append("Trend patterns found")
        
        autocorr = self.analysis_results.get("statistical_tests", {}).get("autocorrelation_tests", {})
        if autocorr.get("ljung_box", {}).get("has_autocorrelation"):
            summary["key_findings"].append("Significant autocorrelation")
        
        freq_analysis = self.analysis_results.get("frequency_analysis", {})
        if freq_analysis.get("has_regular_frequency"):
            summary["key_findings"].append("Regular time intervals")
        
        return summary