from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


# ---------------------- Time Series Forecasting ----------------------

def forecast_auto_arima(ts: pd.Series, periods: int, seasonal: bool = False, m: int = 1) -> Tuple[pd.Series, Any]:
    try:
        import pmdarima as pm
    except Exception as e:
        raise RuntimeError("pmdarima is required for auto ARIMA. Please install it.") from e

    logger.info("Fitting auto_arima model")
    model = pm.auto_arima(ts, seasonal=seasonal, m=m, error_action='ignore', suppress_warnings=True, stepwise=True)
    forecast_index = pd.date_range(start=ts.index[-1] + (ts.index[-1] - ts.index[-2]), periods=periods, freq=pd.infer_freq(ts.index) or 'D')
    forecast_values = model.predict(n_periods=periods)
    forecast = pd.Series(forecast_values, index=forecast_index, name="forecast")
    return forecast, model


def forecast_prophet(ts: pd.Series, periods: int, freq: str = 'D') -> Tuple[pd.DataFrame, Any]:
    try:
        from prophet import Prophet
    except Exception as e:
        raise RuntimeError("prophet is required for Prophet forecasting. Please install it.") from e

    logger.info("Fitting Prophet model")
    df = pd.DataFrame({"ds": ts.index, "y": ts.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast_df = model.predict(future)
    return forecast_df, model


def forecast_lstm(ts: pd.Series, periods: int, lookback: int = 24, epochs: int = 20, batch_size: int = 32) -> Tuple[pd.Series, Any]:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as e:
        raise RuntimeError("tensorflow is required for LSTM. Please install it.") from e

    from sklearn.preprocessing import MinMaxScaler

    logger.info("Training LSTM model")

    values = ts.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    def create_dataset(series: np.ndarray, lb: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(lb, len(series)):
            X.append(series[i-lb:i, 0])
            y.append(series[i, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(values_scaled, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    last_window = values_scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(periods):
        pred = model.predict(last_window, verbose=0)[0, 0]
        preds.append(pred)
        # roll window
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = pred

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    forecast_index = pd.date_range(start=ts.index[-1] + (ts.index[-1] - ts.index[-2]), periods=periods, freq=pd.infer_freq(ts.index) or 'D')
    forecast = pd.Series(preds_inv, index=forecast_index, name="forecast")
    return forecast, model


# ---------------------- Predictive Modeling ----------------------

def build_preprocess_pipeline(X: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def train_classification(X: pd.DataFrame, y: pd.Series, model_name: str = 'RandomForest', tune: bool = False, n_trials: int = 20):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocess_pipeline(X)

    if model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
    elif model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError("Unsupported classification model")

    if tune:
        import optuna

        def objective(trial: optuna.trial.Trial):
            if model_name == 'LogisticRegression':
                C = trial.suggest_float('C', 1e-3, 10.0, log=True)
                solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
                model = Pipeline(steps=[('pre', preprocessor), ('clf', __import__('sklearn').linear_model.LogisticRegression(max_iter=1000, C=C, solver=solver))])
            else:
                n_estimators = trial.suggest_int('n_estimators', 100, 600)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                model = Pipeline(steps=[('pre', preprocessor), ('clf', __import__('sklearn').ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))])

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            score = f1_score(y_valid, preds, average='weighted')
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(f"Best params: {best_params}")

    model = Pipeline(steps=[('pre', preprocessor), ('clf', clf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1_weighted': float(f1_score(y_test, preds, average='weighted')),
    }
    return model, metrics


def train_regression(X: pd.DataFrame, y: pd.Series, model_name: str = 'XGBoost', tune: bool = False, n_trials: int = 20):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocess_pipeline(X)

    if model_name == 'XGBoost':
        try:
            from xgboost import XGBRegressor
            reg = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
        except Exception as e:
            raise RuntimeError("xgboost is required for XGBoost regression. Please install it.") from e
    elif model_name == 'LightGBM':
        try:
            from lightgbm import LGBMRegressor
            reg = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)
        except Exception as e:
            raise RuntimeError("lightgbm is required for LightGBM regression. Please install it.") from e
    else:
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=400, random_state=42)

    if tune:
        import optuna

        def objective(trial: optuna.trial.Trial):
            if model_name == 'XGBoost':
                from xgboost import XGBRegressor
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': 42,
                }
                model = Pipeline(steps=[('pre', preprocessor), ('reg', XGBRegressor(**params))])
            elif model_name == 'LightGBM':
                from lightgbm import LGBMRegressor
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': 42,
                }
                model = Pipeline(steps=[('pre', preprocessor), ('reg', LGBMRegressor(**params))])
            else:
                from sklearn.ensemble import RandomForestRegressor
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'random_state': 42,
                }
                model = Pipeline(steps=[('pre', preprocessor), ('reg', RandomForestRegressor(**params))])

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            score = -np.mean(np.abs(preds - y_valid))  # maximize negative MAE
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Best params: {study.best_params}")

    model = Pipeline(steps=[('pre', preprocessor), ('reg', reg)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        'r2': float(__import__('sklearn').metrics.r2_score(y_test, preds)),
        'mae': float(__import__('sklearn').metrics.mean_absolute_error(y_test, preds)),
        'rmse': float(np.sqrt(__import__('sklearn').metrics.mean_squared_error(y_test, preds))),
    }
    return model, metrics


# ---------------------- AutoML ----------------------

def train_automl_pycaret(df: pd.DataFrame, target: str, task: str = 'regression', fold: int = 5):
    try:
        from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, predict_model as cls_predict, finalize_model as cls_finalize
        from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, predict_model as reg_predict, finalize_model as reg_finalize
    except Exception as e:
        raise RuntimeError("pycaret is required for AutoML. Please install it.") from e

    if task == 'classification':
        s = cls_setup(data=df, target=target, session_id=42, silent=True, verbose=False)
        best = cls_compare()
        best_final = cls_finalize(best)
        leaderboard = cls_pull()
        return best_final, leaderboard
    else:
        s = reg_setup(data=df, target=target, session_id=42, silent=True, verbose=False)
        best = reg_compare()
        best_final = reg_finalize(best)
        leaderboard = reg_pull()
        return best_final, leaderboard


# ---------------------- Explainability ----------------------

def shap_summary_plot(model, X: pd.DataFrame):
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("shap (and matplotlib) are required for SHAP explainability.") from e

    explainer = shap.Explainer(model.predict, X) if hasattr(model, 'predict') else shap.Explainer(model, X)
    shap_values = explainer(X)
    # Draw plot on current Matplotlib figure and return it
    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    return fig


def lime_local_explanation(model, X: pd.DataFrame, instance_index: int = 0):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as e:
        raise RuntimeError("lime is required for LIME explainability.") from e

    is_classification = hasattr(model, 'predict_proba')
    mode = 'classification' if is_classification else 'regression'
    predict_fn = model.predict_proba if is_classification else model.predict

    explainer = LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        verbose=False,
        mode=mode
    )
    instance = X.iloc[instance_index].values
    exp = explainer.explain_instance(instance, predict_fn, num_features=min(10, X.shape[1]))
    return exp.as_list()


# ---------------------- Time series diagnostics ----------------------

def stl_decompose(ts: pd.Series, period: Optional[int] = None):
    from statsmodels.tsa.seasonal import STL
    if period is None:
        period = max(2, min(24, int(np.round(len(ts) / 10)) ))
    stl = STL(ts, period=period)
    res = stl.fit()
    return res


def adf_test(ts: pd.Series) -> Dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(ts.dropna(), autolag='AIC')
    output = {
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'n_lags': int(result[2]),
        'n_obs': int(result[3])
    }
    return output