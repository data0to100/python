import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional


@dataclass
class SQLServerSettings:
    server: str = os.getenv("MSSQL_SERVER", "localhost")
    database: str = os.getenv("MSSQL_DATABASE", "master")
    username: Optional[str] = os.getenv("MSSQL_USERNAME")
    password: Optional[str] = os.getenv("MSSQL_PASSWORD")
    port: int = int(os.getenv("MSSQL_PORT", "1433"))

    driver: str = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
    encrypt: bool = os.getenv("MSSQL_ENCRYPT", "true").lower() in {"1", "true", "yes", "on"}
    trust_server_cert: bool = os.getenv("MSSQL_TRUST_SERVER_CERT", "false").lower() in {"1", "true", "yes", "on"}
    application_name: str = os.getenv("MSSQL_APP_NAME", "data-analysis-pro-aapc")

    pool_size: int = int(os.getenv("MSSQL_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("MSSQL_MAX_OVERFLOW", "10"))
    pool_timeout: int = int(os.getenv("MSSQL_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("MSSQL_POOL_RECYCLE", "1800"))

    echo: bool = os.getenv("MSSQL_ECHO", "false").lower() in {"1", "true", "yes", "on"}
    login_timeout: int = int(os.getenv("MSSQL_LOGIN_TIMEOUT", "30"))
    query_timeout: Optional[int] = (
        int(os.getenv("MSSQL_QUERY_TIMEOUT")) if os.getenv("MSSQL_QUERY_TIMEOUT") else None
    )
    autocommit: bool = os.getenv("MSSQL_AUTOCOMMIT", "false").lower() in {"1", "true", "yes", "on"}

    # Comma-separated list of key=value pairs appended to the ODBC connection string
    odbc_extra_params: Optional[str] = os.getenv("MSSQL_ODBC_EXTRA_PARAMS")

    # For dynamic query safety
    allowed_tables: Optional[str] = os.getenv("MSSQL_ALLOWED_TABLES")
    allowed_columns: Optional[str] = os.getenv("MSSQL_ALLOWED_COLUMNS")

    def parsed_allowed_tables(self) -> Optional[set[str]]:
        if not self.allowed_tables:
            return None
        return {t.strip() for t in self.allowed_tables.split(",") if t.strip()}

    def parsed_allowed_columns(self) -> Optional[Dict[str, set[str]]]:
        if not self.allowed_columns:
            return None
        table_to_columns: Dict[str, set[str]] = {}
        for table_spec in self.allowed_columns.split(";"):
            if not table_spec.strip():
                continue
            if ":" not in table_spec:
                continue
            table_name, cols = table_spec.split(":", 1)
            column_set = {c.strip() for c in cols.split(",") if c.strip()}
            table_to_columns[table_name.strip()] = column_set
        return table_to_columns


@lru_cache(maxsize=1)
def get_sqlserver_settings() -> SQLServerSettings:
    return SQLServerSettings()