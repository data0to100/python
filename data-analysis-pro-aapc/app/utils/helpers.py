from typing import List, Tuple
import pandas as pd


def detect_feature_and_target_columns(df: pd.DataFrame, target_column: str) -> Tuple[List[str], str]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    feature_columns = [c for c in df.columns if c != target_column]
    return feature_columns, target_column


def infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    target_series = df[target_column]
    unique_values = target_series.dropna().unique()
    if target_series.dtype == "O" or len(unique_values) <= 20 and pd.api.types.is_integer_dtype(target_series):
        return "classification"
    if pd.api.types.is_bool_dtype(target_series):
        return "classification"
    return "regression"