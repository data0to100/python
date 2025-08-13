from __future__ import annotations
import pandas as pd


class AIService:
    def __init__(self) -> None:
        pass

    def generate_eda_report(self, df: pd.DataFrame) -> dict:
        desc = df.describe(include="all").to_dict()
        nulls = df.isnull().sum().to_dict()
        return {"describe": desc, "null_counts": nulls}