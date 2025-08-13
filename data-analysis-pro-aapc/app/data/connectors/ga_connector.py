class GAConnector:
    def __init__(self, credentials_path: str | None = None) -> None:
        self.credentials_path = credentials_path

    def fetch_example(self) -> dict:
        # Placeholder for GA data fetch
        return {"message": "GA fetch not implemented in this demo"}