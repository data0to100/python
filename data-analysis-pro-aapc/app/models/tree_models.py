from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base_model import SklearnModel


def random_forest_classifier() -> SklearnModel:
    return SklearnModel(name="random_forest_classifier", estimator=RandomForestClassifier(n_estimators=200, random_state=42))


def random_forest_regressor() -> SklearnModel:
    return SklearnModel(name="random_forest_regressor", estimator=RandomForestRegressor(n_estimators=200, random_state=42))