from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from .base_model import SklearnModel


def gb_classifier() -> SklearnModel:
    return SklearnModel(name="gb_classifier", estimator=GradientBoostingClassifier(random_state=42))


def gb_regressor() -> SklearnModel:
    return SklearnModel(name="gb_regressor", estimator=GradientBoostingRegressor(random_state=42))