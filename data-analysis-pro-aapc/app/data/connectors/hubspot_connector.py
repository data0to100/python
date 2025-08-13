import os
from typing import Any
import requests


class HubSpotConnector:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("HUBSPOT_API_KEY")
        if not self.api_key:
            raise ValueError("HubSpot API key not provided")
        self.base_url = "https://api.hubapi.com"

    def fetch_contacts(self, limit: int = 10) -> Any:
        url = f"{self.base_url}/crm/v3/objects/contacts"
        params = {"hapikey": self.api_key, "limit": limit}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()