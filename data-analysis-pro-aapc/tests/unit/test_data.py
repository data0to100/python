from app.data.data_manager import DataManager
import pandas as pd


def test_data_manager_clean_fills_nulls():
    dm = DataManager()
    df = pd.DataFrame({
        "a": [1.0, None, 3.0],
        "b": ["x", None, "z"],
    })
    cleaned = dm.clean(df)
    assert cleaned.isnull().sum().sum() == 0