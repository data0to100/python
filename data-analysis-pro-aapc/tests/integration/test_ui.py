from fastapi.testclient import TestClient
from app.main import app


def test_list_models():
    client = TestClient(app)
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data and isinstance(data["models"], list) and len(data["models"]) > 0