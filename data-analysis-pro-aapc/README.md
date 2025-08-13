# Data Analysis Pro (AAPC)

A full-stack data analysis service with a FastAPI backend, smart preprocessing, visualization utilities, and an AutoML-inspired "PowerToolModel" that selects the best model for your dataset.

## Features
- REST API for training and prediction
- Automatic task inference (classification vs regression)
- Preprocessing with ColumnTransformer
- Model registry (linear, tree-based, neural, ensembles)
- PowerToolModel for model selection via cross-validation
- Plotly-based visualization utilities

## Quickstart

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Train a model (classification example):

```bash
curl -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [
      {"feat1": 0.1, "feat2": 1, "target": 0},
      {"feat1": 0.9, "feat2": 0, "target": 1}
    ],
    "target_column": "target",
    "model_name": "power_tool"
  }'
```

4. Predict using the saved model:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "power_tool",
    "data": [
      {"feat1": 0.5, "feat2": 1}
    ]
  }'
```

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -t data-analysis-pro-aapc .
docker run -p 8000:8000 --env-file .env data-analysis-pro-aapc
```