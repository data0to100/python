from typing import Callable, Dict
from .base_model import SklearnModel
from .linear_models import linear_regression_model, logistic_regression_model
from .tree_models import random_forest_classifier, random_forest_regressor
from .ensemble_models import gb_classifier, gb_regressor


ModelBuilder = Callable[[], SklearnModel]


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "linear_regression": linear_regression_model,
    "logistic_regression": logistic_regression_model,
    "random_forest_classifier": random_forest_classifier,
    "random_forest_regressor": random_forest_regressor,
    "gb_classifier": gb_classifier,
    "gb_regressor": gb_regressor,
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())