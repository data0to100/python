from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config.logging_config import setup_logging
from .services.model_service import ModelService
from .services.ai_service import AIService
from .utils.validators import validate_records


setup_logging()
app = FastAPI(title="Data Analysis Pro (AAPC)")
model_service = ModelService()
ai_service = AIService()


class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str
    model_name: Optional[str] = None


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_name: str


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def models() -> Dict[str, List[str]]:
    return {"models": model_service.available_models()}


@app.post("/train")
async def train(req: TrainRequest) -> Dict[str, Any]:
    try:
        validate_records(req.data)
        result = model_service.train(req.data, target_column=req.target_column, model_name=req.model_name)
        return {
            "model_name": result.model_name,
            "metrics": result.metrics,
            "artifact_path": result.saved_path,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        validate_records(req.data)
        preds = model_service.predict(req.model_name, req.data)
        return {"predictions": preds}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Train first.")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))