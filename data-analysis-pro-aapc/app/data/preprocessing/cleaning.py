import pandas as pd
from pandas.api.types import is_numeric_dtype


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if is_numeric_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        else:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "")
    return cleaned