from app.services.model_service import ModelService


def test_model_service_train_and_predict():
    service = ModelService()
    records = [
        {"x1": 1, "x2": 2, "y": 3},
        {"x1": 2, "x2": 1, "y": 3},
        {"x1": 3, "x2": 4, "y": 7},
        {"x1": 4, "x2": 3, "y": 7},
    ]
    res = service.train(records, target_column="y", model_name="power_tool")
    assert res.model_name.startswith("power_tool:")
    preds = service.predict(res.model_name, [{"x1": 5, "x2": 6}])
    assert len(preds) == 1