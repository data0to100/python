from sklearn.neural_network import MLPClassifier, MLPRegressor
from .base_model import SklearnModel


def mlp_classifier() -> SklearnModel:
    return SklearnModel(name="mlp_classifier", estimator=MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))


def mlp_regressor() -> SklearnModel:
    return SklearnModel(name="mlp_regressor", estimator=MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))