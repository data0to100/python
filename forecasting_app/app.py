from __future__ import annotations

import io
import json
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html

from utils import get_logger, bytes_to_download_button_bytes
from data_loader import load_file, list_excel_sheets, load_sql_query, SqlConnectionParams, try_parse_datetime
from preprocessing import (
    PreprocessConfig, preprocess_df, ensure_datetime_index,
    create_lag_features, create_rolling_features, one_hot_encode, train_test_split_timeseries
)
from modeling import (
    forecast_auto_arima, forecast_prophet, forecast_lstm,
    train_classification, train_regression, train_automl_pycaret,
    shap_summary_plot, lime_local_explanation, stl_decompose, adf_test
)

logger = get_logger(__name__)

st.set_page_config(page_title="Forecasting & Predictive Analytics", layout="wide")


@st.cache_data(show_spinner=False)
def cache_df(df: pd.DataFrame):
    return df.copy()


def sidebar_navigation() -> str:
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", [
        "Data", "EDA", "Forecast", "Predict", "Explain", "Downloads"
    ])


def upload_or_sql_ui() -> Optional[pd.DataFrame]:
    st.subheader("Data Ingestion")
    source = st.radio("Select data source", ["File Upload", "SQL Server"], horizontal=True)

    if source == "File Upload":
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        df = None
        if file is not None:
            sheet_name = None
            if file.name.lower().endswith((".xlsx", ".xls")):
                sheets = list_excel_sheets(file)
                if sheets:
                    sheet_name = st.selectbox("Excel sheet", sheets)
            with st.spinner("Loading file..."):
                df = load_file(file, sheet_name=sheet_name)
        return df
    else:
        with st.expander("SQL Server Connection", expanded=True):
            server = st.text_input("Server", value="localhost")
            database = st.text_input("Database")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            driver = st.text_input("ODBC Driver", value="ODBC Driver 18 for SQL Server")
            query = st.text_area("SQL Query", height=150)
            if st.button("Run Query"):
                try:
                    params = SqlConnectionParams(server=server, database=database, username=username, password=password, driver=driver)
                    df = load_sql_query(params, query)
                    return df
                except Exception as e:
                    st.error(f"SQL query failed: {e}")
    return None


def page_data() -> Optional[pd.DataFrame]:
    df = upload_or_sql_ui()
    if df is None:
        st.info("Upload a dataset or run a SQL query to proceed.")
        return None

    st.write("Preview:")
    st.dataframe(df.head(100))

    st.markdown("---")

    # Datetime detection
    st.subheader("Datetime Parsing (optional for time series)")
    guess_df, guessed_col = try_parse_datetime(df)
    datetime_col = st.selectbox("Datetime column (optional)", ["<None>"] + df.columns.tolist(), index=(df.columns.tolist().index(guessed_col) + 1) if guessed_col in df.columns else 0)
    datetime_col = None if datetime_col == "<None>" else datetime_col

    # Preprocessing options
    st.subheader("Preprocessing")
    config = PreprocessConfig(
        handle_missing=st.checkbox("Auto-impute missing values", value=True),
        handle_outliers=st.checkbox("Cap outliers", value=True),
        drop_duplicates=st.checkbox("Drop duplicate rows", value=True),
        outlier_method=st.selectbox("Outlier method", ["iqr", "zscore"], index=0),
        zscore_threshold=st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1),
        iqr_multiplier=st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
    )

    processed = preprocess_df(df, config)

    # Optional feature engineering for time series
    if datetime_col:
        processed, _ = ensure_datetime_index(processed, datetime_col)
        if isinstance(processed.index, pd.DatetimeIndex):
            st.subheader("Feature Engineering (Time Series)")
            numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_fe = st.selectbox("Target for lag/rolling features", numeric_cols)
                with st.expander("Lag features"):
                    lags = st.text_input("Lags (comma-separated)", value="1,2,3")
                    if lags.strip():
                        try:
                            lag_list = [int(x.strip()) for x in lags.split(',') if x.strip().isdigit()]
                            processed = create_lag_features(processed, target_fe, lag_list)
                        except Exception as e:
                            st.warning(f"Invalid lags: {e}")
                with st.expander("Rolling features"):
                    windows = st.text_input("Windows (comma-separated)", value="3,7")
                    if windows.strip():
                        try:
                            win_list = [int(x.strip()) for x in windows.split(',') if x.strip().isdigit()]
                            processed = create_rolling_features(processed, target_fe, win_list)
                        except Exception as e:
                            st.warning(f"Invalid windows: {e}")

    st.success(f"Processed dataframe shape: {processed.shape}")
    st.dataframe(processed.head(100))

    st.session_state["data_raw"] = cache_df(df)
    st.session_state["data_processed"] = cache_df(processed)
    st.session_state["datetime_col"] = datetime_col
    return processed


def page_eda():
    df = st.session_state.get("data_processed")
    if df is None:
        st.warning("Go to Data tab first to load and preprocess data.")
        return

    st.subheader("Exploratory Data Analysis")

    # Profile report
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate ydata-profiling report"):
            try:
                from ydata_profiling import ProfileReport
                profile = ProfileReport(df, title="Profiling Report", explorative=True)
                html_str = profile.to_html()
                html(html_str, height=900, scrolling=True)
            except Exception as e:
                st.error(f"Profiling failed: {e}")

    with col2:
        if st.button("Generate Sweetviz report"):
            try:
                import sweetviz as sv
                report = sv.analyze(df)
                save_path = "/workspace/forecasting_app/assets/sweetviz_report.html"
                report.show_html(filepath=save_path, open_browser=False)
                with open(save_path, "r", encoding="utf-8") as f:
                    html(f.read(), height=900, scrolling=True)
            except Exception as e:
                st.error(f"Sweetviz failed: {e}")

    st.markdown("---")

    st.subheader("Visualizations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        target = st.selectbox("Select a numeric column to plot time series (if datetime index present)", ["<None>"] + numeric_cols)
        if target != "<None>":
            if isinstance(df.index, pd.DatetimeIndex):
                fig = px.line(df, x=df.index, y=target, title=f"Time Series - {target}")
            else:
                fig = px.line(df.reset_index(), x=df.reset_index().index, y=target, title=f"Trend - {target}")
            st.plotly_chart(fig, use_container_width=True)

    # Distribution and correlation
    col1, col2 = st.columns(2)
    with col1:
        col = st.selectbox("Distribution column", numeric_cols)
        fig = px.histogram(df, x=col, nbins=50, title=f"Distribution - {col}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Seasonality, trend, anomalies
    st.subheader("Seasonality, Trend and Anomalies")
    if isinstance(df.index, pd.DatetimeIndex) and numeric_cols:
        ts_col = st.selectbox("Target for decomposition", numeric_cols)
        from modeling import stl_decompose, adf_test
        try:
            res = stl_decompose(df[ts_col].dropna())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[ts_col], name='Observed'))
            fig.add_trace(go.Scatter(x=df.index, y=res.trend, name='Trend'))
            fig.add_trace(go.Scatter(x=df.index, y=res.seasonal, name='Seasonal'))
            fig.add_trace(go.Scatter(x=df.index, y=res.resid, name='Residual'))
            fig.update_layout(title='STL Decomposition', legend_orientation='h')
            st.plotly_chart(fig, use_container_width=True)

            adf = adf_test(df[ts_col])
            st.info(f"ADF p-value: {adf['p_value']:.4f} | Stationary: {'Yes' if adf['p_value'] < 0.05 else 'No'}")
        except Exception as e:
            st.error(f"STL/ADF failed: {e}")
    else:
        st.info("Provide a datetime index and a numeric target to analyze seasonality and trend.")


def page_forecast():
    df = st.session_state.get("data_processed")
    datetime_col = st.session_state.get("datetime_col")
    if df is None:
        st.warning("Go to Data tab first to load and preprocess data.")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        st.warning("Set a datetime column in Data tab to use forecasting.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.selectbox("Target variable", numeric_cols)
    periods = st.number_input("Forecast horizon (periods)", min_value=1, max_value=1000, value=24)

    model_choice = st.selectbox("Model", ["Auto ARIMA", "Prophet", "LSTM"])

    if st.button("Run Forecast"):
        try:
            if model_choice == "Auto ARIMA":
                seasonal = st.checkbox("Seasonal", value=False)
                m = st.number_input("Seasonal period (m)", min_value=1, max_value=366, value=1)
                fc, model = forecast_auto_arima(df[target].dropna(), periods=periods, seasonal=seasonal, m=m)
                fc_df = pd.DataFrame({"ds": fc.index, "forecast": fc.values})
            elif model_choice == "Prophet":
                freq = st.text_input("Frequency (pandas offset alias)", value="D")
                forecast_df, model = forecast_prophet(df[target].dropna(), periods=periods, freq=freq)
                fc = forecast_df.set_index("ds")["yhat"].tail(periods)
                fc_df = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).rename(columns={"ds": "ds", "yhat": "forecast"})
            else:
                lookback = st.slider("LSTM lookback window", 5, 120, 24)
                epochs = st.slider("Epochs", 5, 100, 20)
                batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
                fc, model = forecast_lstm(df[target].dropna(), periods=periods, lookback=lookback, epochs=epochs, batch_size=batch_size)
                fc_df = pd.DataFrame({"ds": fc.index, "forecast": fc.values})

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[target], name='Actual'))
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values if hasattr(fc, 'values') else fc, name='Forecast'))
            fig.update_layout(title=f"Forecast - {target}")
            st.plotly_chart(fig, use_container_width=True)

            # Business insight
            hist_mean = df[target].tail(periods).mean() if len(df) >= periods else df[target].mean()
            future_mean = np.mean(fc.values if hasattr(fc, 'values') else fc)
            change_pct = 100.0 * (future_mean - hist_mean) / (hist_mean + 1e-9)
            direction = "increase" if change_pct >= 0 else "drop"
            st.success(f"Insight: {target} is projected to {direction} by {abs(change_pct):.2f}% over the next {periods} periods.")

            st.session_state["forecast_df"] = fc_df
        except Exception as e:
            st.error(f"Forecasting failed: {e}")


def page_predict():
    df = st.session_state.get("data_processed")
    if df is None:
        st.warning("Go to Data tab first to load and preprocess data.")
        return

    st.subheader("Predictive Modeling")

    target = st.selectbox("Target column", df.columns.tolist())
    task_auto = "classification" if df[target].nunique() <= max(2, int(0.05 * len(df))) and not np.issubdtype(df[target].dtype, np.number) else "regression" if np.issubdtype(df[target].dtype, np.number) else "classification"
    task = st.radio("Task", ["classification", "regression", "AutoML"], index=0 if task_auto == "classification" else 1)

    if task == "AutoML":
        if st.button("Run AutoML (PyCaret)"):
            try:
                best_model, leaderboard = train_automl_pycaret(df, target, task='classification' if not np.issubdtype(df[target].dtype, np.number) else 'regression')
                st.write("Leaderboard:")
                st.dataframe(leaderboard)
                st.success("AutoML completed. Best model trained.")
                st.session_state["trained_model"] = best_model
            except Exception as e:
                st.error(f"AutoML failed: {e}")
        return

    X = df.drop(columns=[target])
    y = df[target]

    if task == "classification":
        model_name = st.selectbox("Model", ["RandomForest", "LogisticRegression"]) 
        tune = st.checkbox("Hyperparameter tuning (Optuna)", value=False)
        if st.button("Train model"):
            try:
                model, metrics = train_classification(X, y, model_name=model_name, tune=tune)
                st.json(metrics)
                st.success("Model trained successfully")
                st.session_state["trained_model"] = model
                st.session_state["trained_X"] = X
            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        model_name = st.selectbox("Model", ["XGBoost", "LightGBM", "RandomForest"]) 
        tune = st.checkbox("Hyperparameter tuning (Optuna)", value=False)
        if st.button("Train model"):
            try:
                model, metrics = train_regression(X, y, model_name=model_name, tune=tune)
                st.json(metrics)
                st.success("Model trained successfully")
                st.session_state["trained_model"] = model
                st.session_state["trained_X"] = X
            except Exception as e:
                st.error(f"Training failed: {e}")


def page_explain():
    model = st.session_state.get("trained_model")
    X = st.session_state.get("trained_X")

    if model is None or X is None:
        st.warning("Train a model in Predict tab first.")
        return

    st.subheader("Model Explainability")

    explain_choice = st.selectbox("Explainability method", ["SHAP", "LIME"])

    if explain_choice == "SHAP":
        try:
            fig = shap_summary_plot(model, X.sample(min(100, len(X)), random_state=42))
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP failed: {e}")
    else:
        idx = st.number_input("Row index to explain", min_value=0, max_value=len(X)-1, value=0)
        try:
            exp = lime_local_explanation(model, X, instance_index=int(idx))
            st.write("Local explanation (feature, contribution):")
            st.write(exp)
        except Exception as e:
            st.error(f"LIME failed: {e}")


def page_downloads():
    st.subheader("Downloads")

    # Forecast CSV
    fc_df = st.session_state.get("forecast_df")
    if fc_df is not None:
        st.write("Forecast preview:")
        st.dataframe(fc_df.head(100))
        st.download_button(
            label="Download forecast CSV",
            data=bytes_to_download_button_bytes(fc_df, index=False),
            file_name="forecast.csv",
            mime="text/csv"
        )

    if st.button("Generate PDF report"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from datetime import datetime

            path = "/workspace/forecasting_app/outputs/report.pdf"
            c = canvas.Canvas(path, pagesize=A4)
            width, height = A4

            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, height - 40, "Forecasting & Predictive Analytics Report")
            c.setFont("Helvetica", 10)
            c.drawString(40, height - 60, f"Generated: {datetime.now().isoformat()} ")

            y = height - 90
            df = st.session_state.get("data_processed")
            if df is not None:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Data Summary")
                y -= 15
                c.setFont("Helvetica", 10)
                c.drawString(40, y, f"Shape: {df.shape}")
                y -= 15

            if fc_df is not None:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Forecast Summary")
                y -= 15
                c.setFont("Helvetica", 10)
                c.drawString(40, y, f"Horizon: {len(fc_df)} periods")
                y -= 15

            c.showPage()
            c.save()
            with open(path, "rb") as f:
                st.download_button("Download PDF report", data=f.read(), file_name="report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF report generation failed: {e}")


# ---------------------- Main ----------------------

page = sidebar_navigation()

if page == "Data":
    page_data()
elif page == "EDA":
    page_eda()
elif page == "Forecast":
    page_forecast()
elif page == "Predict":
    page_predict()
elif page == "Explain":
    page_explain()
else:
    page_downloads()