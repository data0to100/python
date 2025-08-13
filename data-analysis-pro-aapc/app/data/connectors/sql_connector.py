from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError

from app.config.logging_config import get_logger
from app.config.settings import SQLServerSettings, get_sqlserver_settings
from app.data.connectors.base_connector import (
    BaseConnector,
    DatabaseConnectionError,
    DatabaseQueryError,
)


class SQLServerConnector(BaseConnector):
    def __init__(
        self,
        settings: Optional[SQLServerSettings] = None,
        logger: Optional[logging.Logger] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        self.settings = settings or get_sqlserver_settings()
        self.logger = logger or get_logger(self.__class__.__name__)
        self._engine: Optional[Engine] = engine

    # ---- Public API ----
    def connect(self) -> None:
        if self._engine is not None:
            return
        try:
            self._engine = self._create_engine()
            # Validate connection with a ping
            self.ping()
            self.logger.info("SQLServer connection pool initialized")
        except Exception as exc:  # noqa: BLE001
            self._engine = None
            self.logger.exception("Failed to initialize SQLServer engine: %s", type(exc).__name__)
            raise DatabaseConnectionError("Failed to connect to SQL Server") from exc

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self.logger.info("SQLServer connection pool disposed")

    def execute_non_query(self, query: Any, params: Optional[Dict[str, Any]] = None) -> int:
        statement = self._to_text_clause(query)
        try:
            with self._ensure_connection().begin() as connection:
                self._log_query(statement, params)
                result: Result = connection.execute(statement, params or {})
                rowcount = result.rowcount if result.rowcount is not None else 0
                return int(rowcount)
        except (OperationalError, DBAPIError, SQLAlchemyError) as exc:
            self._handle_query_exception(exc, statement)

    def execute_many(self, query: Any, param_list: Iterable[Dict[str, Any]]) -> int:
        statement = self._to_text_clause(query)
        params_as_list = list(param_list)
        try:
            with self._ensure_connection().begin() as connection:
                self._log_query(statement, {"__many__": len(params_as_list)})
                result: Result = connection.execute(statement, params_as_list)
                rowcount = result.rowcount if result.rowcount is not None else 0
                return int(rowcount)
        except (OperationalError, DBAPIError, SQLAlchemyError) as exc:
            self._handle_query_exception(exc, statement)

    def fetch_all(self, query: Any, params: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        statement = self._to_text_clause(query)
        try:
            with self._ensure_connection().connect() as connection:
                self._log_query(statement, params)
                result: Result = connection.execute(statement, params or {})
                rows = result.mappings().all()
                return [dict(row) for row in rows]
        except (OperationalError, DBAPIError, SQLAlchemyError) as exc:
            self._handle_query_exception(exc, statement)

    def fetch_one(self, query: Any, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        statement = self._to_text_clause(query)
        try:
            with self._ensure_connection().connect() as connection:
                self._log_query(statement, params)
                result: Result = connection.execute(statement, params or {})
                row = result.mappings().first()
                return dict(row) if row is not None else None
        except (OperationalError, DBAPIError, SQLAlchemyError) as exc:
            self._handle_query_exception(exc, statement)

    @contextmanager
    def transaction(self):
        try:
            with self._ensure_connection().begin() as connection:
                yield connection
        except (OperationalError, DBAPIError, SQLAlchemyError) as exc:
            self.logger.exception("Transaction failed: %s", type(exc).__name__)
            raise DatabaseQueryError("Transaction failed") from exc

    # ---- Health ----
    def ping(self) -> None:
        self._retry_operation(self._ping_once, retries=3, base_delay_seconds=0.5)

    # ---- Dynamic query helpers ----
    def build_select_query(
        self,
        table_name: str,
        columns: Sequence[str],
        filters: Optional[Mapping[str, Any]] = None,
        order_by: Optional[Sequence[str]] = None,
        allowed_tables: Optional[set[str]] = None,
        allowed_columns: Optional[Mapping[str, set[str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._validate_identifiers(table_name, columns, filters, order_by, allowed_tables, allowed_columns)

        selectable_columns = [sa.column(column_name) for column_name in columns]
        table = sa.table(table_name)
        stmt = sa.select(*selectable_columns).select_from(table)

        params: Dict[str, Any] = {}
        if filters:
            for key, value in filters.items():
                bind_name = f"param_{key}"
                stmt = stmt.where(sa.column(key) == sa.bindparam(bind_name))
                params[bind_name] = value

        if order_by:
            order_clauses = []
            for raw in order_by:
                is_desc = raw.lower().endswith(" desc") or raw.startswith("-")
                clean = raw.replace(" desc", "").replace(" DESC", "").lstrip("-")
                col_expr = sa.column(clean)
                order_clauses.append(col_expr.desc() if is_desc else col_expr.asc())
            stmt = stmt.order_by(*order_clauses)

        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)

        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        return str(compiled), params

    def build_insert_query(
        self,
        table_name: str,
        values: Mapping[str, Any],
        allowed_tables: Optional[set[str]] = None,
        allowed_columns: Optional[Mapping[str, set[str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        columns = list(values.keys())
        self._validate_identifiers(table_name, columns, None, None, allowed_tables, allowed_columns)

        table = sa.table(table_name, *[sa.column(c) for c in columns])
        bind_map = {c: sa.bindparam(f"param_{c}") for c in columns}
        stmt = sa.insert(table).values(**bind_map)
        params = {f"param_{c}": values[c] for c in columns}
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        return str(compiled), params

    def build_update_query(
        self,
        table_name: str,
        values: Mapping[str, Any],
        filters: Mapping[str, Any],
        allowed_tables: Optional[set[str]] = None,
        allowed_columns: Optional[Mapping[str, set[str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        value_columns = list(values.keys())
        filter_columns = list(filters.keys())
        all_columns = value_columns + filter_columns
        self._validate_identifiers(table_name, all_columns, None, None, allowed_tables, allowed_columns)

        # Provide both value and filter columns on the table for compilation
        table = sa.table(
            table_name,
            *[sa.column(c) for c in set(all_columns)],
        )
        value_bind_map = {c: sa.bindparam(f"val_{c}") for c in value_columns}
        stmt = sa.update(table).values(**value_bind_map)

        params: Dict[str, Any] = {f"val_{c}": values[c] for c in value_columns}
        for key, value in filters.items():
            bind_name = f"f_{key}"
            stmt = stmt.where(sa.column(key) == sa.bindparam(bind_name))
            params[bind_name] = value

        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        return str(compiled), params

    def build_delete_query(
        self,
        table_name: str,
        filters: Mapping[str, Any],
        allowed_tables: Optional[set[str]] = None,
        allowed_columns: Optional[Mapping[str, set[str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        filter_columns = list(filters.keys())
        self._validate_identifiers(table_name, filter_columns, None, None, allowed_tables, allowed_columns)

        table = sa.table(table_name, *[sa.column(c) for c in filter_columns])
        stmt = sa.delete(table)
        params: Dict[str, Any] = {}
        for key, value in filters.items():
            bind_name = f"f_{key}"
            stmt = stmt.where(sa.column(key) == sa.bindparam(bind_name))
            params[bind_name] = value

        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        return str(compiled), params

    # ---- Internals ----
    def _create_engine(self) -> Engine:
        url = self._build_sqla_url(self.settings)
        engine = sa.create_engine(
            url,
            pool_size=self.settings.pool_size,
            max_overflow=self.settings.max_overflow,
            pool_timeout=self.settings.pool_timeout,
            pool_recycle=self.settings.pool_recycle,
            pool_pre_ping=True,
            echo=self.settings.echo,
            future=True,
            fast_executemany=True,
        )
        return engine

    def _build_sqla_url(self, s: SQLServerSettings) -> sa.engine.URL:
        # We pass sensitive values separately to URL to avoid logging them
        query: Dict[str, str] = {
            "driver": s.driver,
            "Encrypt": "yes" if s.encrypt else "no",
            "TrustServerCertificate": "yes" if s.trust_server_cert else "no",
            "ApplicationIntent": "ReadWrite",
            "MultiSubnetFailover": "yes",
            "App": s.application_name,
            "LoginTimeout": str(s.login_timeout),
        }
        if s.query_timeout is not None:
            query["Timeout"] = str(s.query_timeout)
        if s.odbc_extra_params:
            for part in s.odbc_extra_params.split(","):
                if not part.strip():
                    continue
                if "=" in part:
                    k, v = part.split("=", 1)
                    query[k.strip()] = v.strip()

        return sa.engine.URL.create(
            drivername="mssql+pyodbc",
            username=s.username,
            password=s.password,
            host=s.server,
            port=s.port,
            database=s.database,
            query=query,
        )

    def _ensure_connection(self) -> Engine:
        if self._engine is None:
            self.connect()
        assert self._engine is not None
        return self._engine

    def _to_text_clause(self, query: Any) -> sa.sql.elements.TextClause:
        if isinstance(query, str):
            return text(query)
        if isinstance(query, sa.sql.elements.TextClause):
            return query
        try:
            compiled = str(query.compile(compile_kwargs={"literal_binds": False}))
            return text(compiled)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Unsupported query type; expected str, TextClause, or SQLAlchemy selectable") from exc

    def _log_query(self, statement: sa.sql.elements.TextClause, params: Optional[Dict[str, Any]]) -> None:
        try:
            param_info = None
            if params is None:
                param_info = "{}"
            elif isinstance(params, dict):
                redacted = {k: self._redact_value(v) for k, v in params.items()}
                param_info = str(redacted)
            else:
                param_info = f"<params type={type(params).__name__}>"
            self.logger.debug("Executing SQL: %s | params=%s", statement.text, param_info)
        except Exception:  # noqa: BLE001
            # Logging must never break queries
            pass

    def _handle_query_exception(self, exc: Exception, statement: sa.sql.elements.TextClause) -> None:
        self.logger.exception("SQL execution failed: %s | sql=%s", type(exc).__name__, statement.text)
        if isinstance(exc, (OperationalError, DBAPIError, SQLAlchemyError)):
            raise DatabaseQueryError("SQL execution failed") from exc
        raise

    def _retry_operation(self, func, retries: int, base_delay_seconds: float) -> None:
        attempt = 0
        while True:
            try:
                func()
                return
            except (OperationalError, DBAPIError) as exc:
                attempt += 1
                if attempt > retries:
                    self.logger.error("Operation failed after %d retries: %s", retries, type(exc).__name__)
                    raise DatabaseConnectionError("Operation failed after retries") from exc
                delay = base_delay_seconds * (2 ** (attempt - 1))
                self.logger.warning("Transient DB error, retrying in %.2fs (attempt %d/%d)", delay, attempt, retries)
                time.sleep(delay)

    def _ping_once(self) -> None:
        engine = self._ensure_connection()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def _redact_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        value_str = str(value)
        if len(value_str) > 64:
            return value_str[:61] + "..."
        return value_str

    def _validate_identifiers(
        self,
        table_name: str,
        columns: Sequence[str],
        filters: Optional[Mapping[str, Any]],
        order_by: Optional[Sequence[str]],
        allowed_tables: Optional[set[str]],
        allowed_columns: Optional[Mapping[str, set[str]]],
    ) -> None:
        allowed_tables = (
            allowed_tables
            or self.settings.parsed_allowed_tables()
            or set()
        )
        allowed_columns = (
            allowed_columns
            or self.settings.parsed_allowed_columns()
            or {}
        )

        if allowed_tables and table_name not in allowed_tables:
            raise ValueError(f"Table not allowed: {table_name}")
        if allowed_columns:
            allowed_for_table = allowed_columns.get(table_name)
            if allowed_for_table is not None:
                for c in columns:
                    if c not in allowed_for_table:
                        raise ValueError(f"Column not allowed for {table_name}: {c}")
                if filters:
                    for key in filters.keys():
                        if key not in allowed_for_table:
                            raise ValueError(f"Filter column not allowed for {table_name}: {key}")
                if order_by:
                    for raw in order_by:
                        clean = raw.replace(" desc", "").replace(" DESC", "").lstrip("-")
                        if clean not in allowed_for_table:
                            raise ValueError(f"Order-by column not allowed for {table_name}: {clean}")