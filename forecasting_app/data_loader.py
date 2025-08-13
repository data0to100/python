from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import io

import pandas as pd
import streamlit as st

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class SqlConnectionParams:
    server: str
    database: str
    username: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    encrypt: str = "yes"
    trust_server_certificate: str = "yes"


@st.cache_data(show_spinner=True)
def load_file(file: st.runtime.uploaded_file_manager.UploadedFile, 
              sheet_name: Optional[str] = None, 
              encoding: Optional[str] = None) -> pd.DataFrame:
    if file is None:
        raise ValueError("No file uploaded")

    name = file.name.lower()
    logger.info(f"Loading file: {name}")

    try:
        if name.endswith(".csv"):
            if encoding:
                df = pd.read_csv(file, encoding=encoding)
            else:
                # try utf-8 first
                try:
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    # fallback to latin-1
                    file.seek(0)
                    df = pd.read_csv(file, encoding="latin-1")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file, sheet_name=sheet_name)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    except Exception as e:
        logger.exception("Failed to load file")
        raise e

    logger.info(f"Loaded dataframe with shape {df.shape}")
    return df


@st.cache_data(show_spinner=True)
def list_excel_sheets(file: st.runtime.uploaded_file_manager.UploadedFile) -> List[str]:
    try:
        excel = pd.ExcelFile(file)
        return excel.sheet_names
    except Exception:
        return []


@st.cache_data(show_spinner=True)
def load_sql_query(params: SqlConnectionParams, query: str) -> pd.DataFrame:
    import urllib
    from sqlalchemy import create_engine

    if not query.strip():
        raise ValueError("SQL query is empty")

    conn_str = (
        f"DRIVER={{{{{{params.driver}}}}}};"
        f"SERVER={params.server};"
        f"DATABASE={params.database};"
        f"UID={params.username};PWD={params.password};"
        f"Encrypt={params.encrypt};TrustServerCertificate={params.trust_server_certificate};"
    )

    logger.info("Creating SQLAlchemy engine for SQL Server via pyodbc")

    quoted = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    logger.info(f"SQL query returned shape {df.shape}")
    return df


def try_parse_datetime(df: pd.DataFrame, prefer_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    if df.empty:
        return df, None

    candidate_cols = []
    if prefer_col and prefer_col in df.columns:
        candidate_cols.append(prefer_col)
    candidate_cols.extend([c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()])
    candidate_cols.extend([c for c in df.select_dtypes(include=["object"]).columns])

    parsed_col: Optional[str] = None

    for col in dict.fromkeys(candidate_cols):
        try:
            parsed = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
            df = df.copy()
            df[col] = parsed
            parsed_col = col
            logger.info(f"Parsed datetime column: {col}")
            break
        except Exception:
            continue

    return df, parsed_col