from fastapi.testclient import TestClient
from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_train_and_predict_endpoints():
    client = TestClient(app)

    train_payload = {
        "data": [
            {"a": 1, "b": 2, "target": 3},
            {"a": 2, "b": 1, "target": 3},
            {"a": 3, "b": 4, "target": 7}
        ],
        "target_column": "target",
        "model_name": "power_tool"
    }

    resp = client.post("/train", json=train_payload)
    assert resp.status_code == 200, resp.text
    model_name = resp.json()["model_name"]

    pred_payload = {
        "model_name": model_name,
        "data": [{"a": 5, "b": 6}]
    }
    resp_pred = client.post("/predict", json=pred_payload)
    assert resp_pred.status_code == 200, resp_pred.text
    assert "predictions" in resp_pred.json()