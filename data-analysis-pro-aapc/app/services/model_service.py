from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import pandas as pd

from ..config.settings import settings
from ..models.base_model import BaseModel, SklearnModel
from ..models.model_registry import MODEL_REGISTRY, list_models
from ..models.power_tool_model import PowerToolModel
from ..data.data_manager import DataManager
from ..utils.helpers import detect_feature_and_target_columns, infer_task_type


@dataclass
class TrainResult:
    model_name: str
    metrics: Dict[str, float]
    saved_path: str


class ModelService:
    def __init__(self) -> None:
        self.data_manager = DataManager()
        self.models_dir = settings.model_artifacts_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.loaded_models: dict[str, BaseModel] = {}

    def available_models(self) -> List[str]:
        return list_models() + ["power_tool"]

    def _artifact_path(self, name: str) -> str:
        safe = name.replace("/", "_")
        return os.path.join(self.models_dir, f"{safe}.joblib")

    def train(self, records: List[dict], target_column: str, model_name: Optional[str] = None) -> TrainResult:
        df = self.data_manager.from_records(records)
        df = self.data_manager.clean(df)

        chosen = model_name or settings.default_model
        if chosen == "power_tool":
            pt = PowerToolModel()
            result = pt.fit(df, target_column)
            model_to_save: BaseModel = result.best_model
            metrics = result.metrics
            model_name_to_use = model_to_save.name
        else:
            if chosen not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {chosen}")
            features, target = detect_feature_and_target_columns(df, target_column)
            task_type = infer_task_type(df, target)

            from ..data.preprocessing.feature_engineering import build_preprocess_pipeline
            from sklearn.pipeline import Pipeline

            preprocess, _, _ = build_preprocess_pipeline(df, features)
            model = MODEL_REGISTRY[chosen]()

            X = df[features]
            y = df[target]

            pipeline = Pipeline(steps=[("prep", preprocess), ("model", model.estimator)])
            pipeline.fit(X, y)
            model_to_save = SklearnModel(name=chosen, estimator=pipeline)

            # naive metrics on train for demo
            preds = pipeline.predict(X)
            if task_type == "classification":
                from sklearn.metrics import f1_score, accuracy_score

                metrics = {
                    "f1_weighted": float(f1_score(y, preds, average="weighted")),
                    "accuracy": float(accuracy_score(y, preds)),
                }
            else:
                from sklearn.metrics import r2_score, mean_squared_error
                import numpy as np

                metrics = {
                    "r2": float(r2_score(y, preds)),
                    "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                }
            model_name_to_use = chosen

        path = self._artifact_path(model_name_to_use)
        model_to_save.save(path)
        self.loaded_models[model_name_to_use] = model_to_save

        return TrainResult(model_name=model_name_to_use, metrics=metrics, saved_path=path)

    def load(self, model_name: str) -> BaseModel:
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        path = self._artifact_path(model_name)
        model = SklearnModel.load(path)
        self.loaded_models[model_name] = model
        return model

    def predict(self, model_name: str, records: List[dict]) -> List[Any]:
        model = self.load(model_name)
        df = self.data_manager.from_records(records)
        df = self.data_manager.clean(df)
        preds = model.predict(df)  # type: ignore[arg-type]
        # Normalize to JSON-serializable types
        try:
            import numpy as np

            if isinstance(preds, np.ndarray):
                return preds.tolist()  # type: ignore[return-value]
            if hasattr(preds, "tolist"):
                return preds.tolist()  # type: ignore[return-value]
            if not isinstance(preds, list):
                return [preds]
            converted: List[Any] = []
            for val in preds:
                if isinstance(val, np.generic):
                    converted.append(val.item())
                else:
                    converted.append(val)
            return converted
        except Exception:
            try:
                return list(preds)  # type: ignore[list-item]
            except Exception:
                return [preds]