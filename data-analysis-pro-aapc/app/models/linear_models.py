from sklearn.linear_model import LinearRegression, LogisticRegression
from .base_model import SklearnModel


def linear_regression_model() -> SklearnModel:
    return SklearnModel(name="linear_regression", estimator=LinearRegression())


def logistic_regression_model() -> SklearnModel:
    return SklearnModel(name="logistic_regression", estimator=LogisticRegression(max_iter=1000))