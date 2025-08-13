# Streamlit Forecasting & Predictive Analytics App

## Quickstart

### Local

1. Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r forecasting_app/requirements.txt
```

2. Run the app:
```bash
cd forecasting_app
streamlit run app.py
```

### Docker

```bash
cd forecasting_app
docker build -t forecasting-app .
docker run --rm -p 8501:8501 forecasting-app
```

Open http://localhost:8501.

## Features
- Data ingestion from CSV/Excel upload or SQL Server via pyodbc/SQLAlchemy
- Preprocessing: missing values, outliers, duplicates, datetime parsing
- Feature engineering: lag and rolling features, one-hot encoding
- EDA: ydata-profiling and Sweetviz reports; Plotly charts
- Time series forecasting: Auto ARIMA, Prophet, LSTM
- Predictive modeling: classification and regression; XGBoost, LightGBM
- Hyperparameter tuning with Optuna
- Explainability: SHAP and LIME
- Downloads: CSV forecasts, PDF reports

## Notes
- Prophet relies on `cmdstanpy` which may download toolchains on first run.
- SQL Server connectivity requires `msodbcsql18` and appropriate network access.