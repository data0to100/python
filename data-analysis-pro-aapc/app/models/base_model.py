from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import joblib
import os


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y) -> None:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def get_underlying(self):  # pragma: no cover
        raise NotImplementedError

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        return joblib.load(path)


@dataclass
class SklearnModel(BaseModel):
    name: str
    estimator: Any
    metadata: Optional[Dict[str, Any]] = None

    def fit(self, X, y) -> None:
        self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def get_underlying(self):
        return self.estimator