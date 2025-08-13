from typing import Any
import plotly.graph_objects as go


def fig_to_json(fig: go.Figure) -> dict[str, Any]:
    return fig.to_plotly_json()