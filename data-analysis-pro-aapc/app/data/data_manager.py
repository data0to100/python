from typing import Any, Dict, List
import pandas as pd
from .preprocessing.cleaning import basic_clean


class DataManager:
    def __init__(self) -> None:
        pass

    def from_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame.from_records(records)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return basic_clean(df)