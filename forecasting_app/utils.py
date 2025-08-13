import logging
import os
from typing import Optional

import streamlit as st


def ensure_directories_exist() -> None:
    directories = ["/workspace/forecasting_app/logs", 
                   "/workspace/forecasting_app/assets", 
                   "/workspace/forecasting_app/outputs", 
                   "/workspace/forecasting_app/tmp"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    ensure_directories_exist()

    logger = logging.getLogger(name if name else "forecasting_app")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(ch_formatter)

    # File handler
    fh = logging.FileHandler("/workspace/forecasting_app/logs/app.log")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


@st.cache_data(show_spinner=False)
def cache_data_frame(key: str, data):
    return data


def is_time_series(df, datetime_col: Optional[str]) -> bool:
    if datetime_col is None:
        return False
    try:
        df_sorted = df.sort_values(datetime_col)
        return df_sorted[datetime_col].is_monotonic_increasing
    except Exception:
        return False


def bytes_to_download_button_bytes(df, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")