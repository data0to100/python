from app.models.model_registry import list_models
from app.models.power_tool_model import PowerToolModel
import pandas as pd


def test_model_registry_non_empty():
    models = list_models()
    assert "linear_regression" in models
    assert "random_forest_regressor" in models


def test_power_tool_fit_predict_regression():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [2, 1, 2, 1, 2, 1],
        "y": [2.0, 3.9, 6.1, 7.8, 10.1, 12.2],
    })
    pt = PowerToolModel()
    result = pt.fit(df, target_column="y")
    assert result.best_model is not None
    preds = pt.predict(df)
    assert len(preds) == len(df)