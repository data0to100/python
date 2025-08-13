from __future__ import annotations
from typing import Any, List


class DataService:
    def __init__(self) -> None:
        pass

    def fetch_from_sql(self, database_url: str, sql: str, params: dict | None = None) -> List[dict[str, Any]]:
        from ..data.connectors.sql_connector import SQLConnector

        connector = SQLConnector(database_url)
        return connector.fetch(sql, params)

    def fetch_hubspot_contacts(self, api_key: str | None = None) -> dict:
        from ..data.connectors.hubspot_connector import HubSpotConnector

        connector = HubSpotConnector(api_key)
        return connector.fetch_contacts()