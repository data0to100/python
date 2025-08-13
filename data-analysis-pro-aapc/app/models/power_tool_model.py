from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

from .base_model import SklearnModel
from .linear_models import linear_regression_model, logistic_regression_model
from .tree_models import random_forest_classifier, random_forest_regressor
from .ensemble_models import gb_classifier, gb_regressor
from ..data.preprocessing.feature_engineering import build_preprocess_pipeline
from ..utils.helpers import infer_task_type, detect_feature_and_target_columns


@dataclass
class PowerToolResult:
    best_model: SklearnModel
    metrics: Dict[str, float]
    candidates_scores: Dict[str, float]


class PowerToolModel:
    def __init__(self) -> None:
        self.best_model: SklearnModel | None = None
        self.preprocess: Any | None = None
        self.feature_columns: List[str] = []
        self.task_type: str | None = None

    def _get_candidates(self, task_type: str) -> List[SklearnModel]:
        if task_type == "classification":
            return [
                logistic_regression_model(),
                random_forest_classifier(),
                gb_classifier(),
            ]
        return [
            linear_regression_model(),
            random_forest_regressor(),
            gb_regressor(),
        ]

    def fit(self, df: pd.DataFrame, target_column: str) -> PowerToolResult:
        self.feature_columns, target = detect_feature_and_target_columns(df, target_column)
        self.task_type = infer_task_type(df, target)
        self.preprocess, _, _ = build_preprocess_pipeline(df, self.feature_columns)

        X = df[self.feature_columns]
        y = df[target]

        candidates = self._get_candidates(self.task_type)

        best_score = -np.inf
        best: SklearnModel | None = None
        scores_map: Dict[str, float] = {}

        for candidate in candidates:
            pipeline = Pipeline(steps=[
                ("prep", self.preprocess),
                ("model", candidate.estimator),
            ])
            try:
                if self.task_type == "classification":
                    cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring="f1_weighted")
                    score = float(np.mean(cv_scores))
                else:
                    cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring="r2")
                    score = float(np.mean(cv_scores))
            except Exception:
                score = float("-inf")

            scores_map[candidate.name] = score
            if score > best_score:
                best_score = score
                best = candidate

        if best is None or not np.isfinite(best_score):
            # Fallback: pick the first candidate and proceed without CV
            best = candidates[0]

        # Fit the best pipeline on full data
        final_pipeline = Pipeline(steps=[("prep", self.preprocess), ("model", best.estimator)])
        final_pipeline.fit(X, y)
        self.best_model = SklearnModel(name=f"power_tool:{best.name}", estimator=final_pipeline)

        # Compute metrics on training set (for demo; normally use holdout)
        preds = final_pipeline.predict(X)
        if self.task_type == "classification":
            metrics = {
                "f1_weighted": float(f1_score(y, preds, average="weighted")),
                "accuracy": float(accuracy_score(y, preds)),
            }
        else:
            rmse = float(np.sqrt(mean_squared_error(y, preds)))
            metrics = {
                "r2": float(r2_score(y, preds)),
                "rmse": rmse,
            }

        return PowerToolResult(best_model=self.best_model, metrics=metrics, candidates_scores=scores_map)

    def predict(self, df: pd.DataFrame) -> List[float | int]:
        if not self.best_model:
            raise RuntimeError("PowerToolModel not fitted")
        X = df[self.feature_columns]
        preds = self.best_model.predict(X)
        return preds.tolist()  # type: ignore[return-value]