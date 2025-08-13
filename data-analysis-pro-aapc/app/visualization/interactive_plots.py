from typing import Any, Dict
import pandas as pd
from .plot_factory import histogram, correlation_heatmap
from .plot_utils import fig_to_json


def build_histogram_json(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    fig = histogram(df, column)
    return fig_to_json(fig)


def build_corr_json(df: pd.DataFrame) -> Dict[str, Any]:
    fig = correlation_heatmap(df)
    return fig_to_json(fig)