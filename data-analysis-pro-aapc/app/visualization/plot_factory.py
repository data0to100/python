from typing import List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def histogram(df: pd.DataFrame, column: str) -> go.Figure:
    return px.histogram(df, x=column)


def correlation_heatmap(df: pd.DataFrame, columns: List[str] | None = None) -> go.Figure:
    subset = df[columns] if columns else df.select_dtypes(include=["number"]).copy()
    corr = subset.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
    return fig