from typing import List, Tuple
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocess_pipeline(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for col in feature_columns:
        if is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    return transformer, numeric_features, categorical_features