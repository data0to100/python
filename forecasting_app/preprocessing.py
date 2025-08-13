from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    handle_missing: bool = True
    handle_outliers: bool = True
    drop_duplicates: bool = True
    outlier_method: str = "iqr"  # iqr or zscore
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.isna().mean().sort_values(ascending=False).to_frame(name="missing_ratio")
    return summary


def auto_impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows")
    return df


def cap_outliers(df: pd.DataFrame, method: str = "iqr", zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    if method == "zscore":
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            z = (df[col] - mean) / std
            df[col] = np.where(z > zscore_threshold, mean + zscore_threshold * std,
                               np.where(z < -zscore_threshold, mean - zscore_threshold * std, df[col]))
    else:  # IQR method
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            df[col] = df[col].clip(lower, upper)

    return df


def preprocess_df(df: pd.DataFrame, config: Optional[PreprocessConfig] = None) -> pd.DataFrame:
    if config is None:
        config = PreprocessConfig()

    logger.info("Starting preprocessing pipeline")
    processed = df.copy()

    if config.drop_duplicates:
        processed = remove_duplicates(processed)

    if config.handle_missing:
        processed = auto_impute_missing_values(processed)

    if config.handle_outliers:
        processed = cap_outliers(processed, method=config.outlier_method, zscore_threshold=config.zscore_threshold, iqr_multiplier=config.iqr_multiplier)

    return processed


def ensure_datetime_index(df: pd.DataFrame, datetime_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    if datetime_col and datetime_col in df.columns and not is_datetime64_any_dtype(df[datetime_col]):
        try:
            df = df.copy()
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce", infer_datetime_format=True)
        except Exception:
            pass

    if datetime_col and datetime_col in df.columns:
        df = df.sort_values(datetime_col)
        return df.set_index(datetime_col), datetime_col

    return df, None


def create_lag_features(ts_df: pd.DataFrame, target: str, lags: List[int]) -> pd.DataFrame:
    df = ts_df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    df = df.dropna()
    return df


def create_rolling_features(ts_df: pd.DataFrame, target: str, windows: List[int]) -> pd.DataFrame:
    df = ts_df.copy()
    for window in windows:
        df[f"{target}_rollmean_{window}"] = df[target].rolling(window).mean()
        df[f"{target}_rollstd_{window}"] = df[target].rolling(window).std()
    df = df.dropna()
    return df


def one_hot_encode(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in exclude]
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def train_test_split_timeseries(df: pd.DataFrame, target: str, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        return df, df.iloc[0:0]
    return df.iloc[:-test_size], df.iloc[-test_size:]