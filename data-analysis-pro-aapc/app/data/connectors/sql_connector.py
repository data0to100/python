from typing import Any
from sqlalchemy import create_engine, text


class SQLConnector:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url)

    def fetch(self, sql: str, params: dict[str, Any] | None = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows = [dict(r._mapping) for r in result]
        return rows