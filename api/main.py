import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .ml_core import FEATURES, LABELS, UNITS, rain_shap

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "ml_model" / "rain_rf_smote_pipeline.joblib"))
META_PATH = Path(os.getenv("META_PATH", BASE_DIR / "ml_model" / "model_meta.json"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", BASE_DIR / "ml_model" / "model_metrics.json"))
API_KEY = os.getenv("RAIN_API_KEY", "dev-secret-key")   # set a real one in production

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(f"Model file missing or empty: {MODEL_PATH}. Run `python train.py`.")
    model = joblib.load(MODEL_PATH)
    state["model"] = model
    state["explainer"] = shap.TreeExplainer(model.named_steps["rf"])
    state["threshold"] = (
        json.loads(META_PATH.read_text())["threshold"] if META_PATH.exists() else 0.5
    )
    state["metrics"] = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    yield
    state.clear()


app = FastAPI(
    title="Rain Prediction API",
    description="Random Forest + SMOTE rain classifier with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------- auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(api_key_header)):
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


# ---------------------------------------------------------------- schemas
class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=10, le=35, description="Temperature in °C")
    humidity: float = Field(..., ge=30, le=100, description="Relative humidity in %")
    wind_speed: float = Field(..., ge=0, le=20, description="Wind speed in km/h")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover in %")
    pressure: float = Field(..., ge=980, le=1050, description="Pressure in hPa")

    model_config = {
        "json_schema_extra": {
            "example": {"temperature": 18, "humidity": 88, "wind_speed": 8,
                        "cloud_cover": 80, "pressure": 1010}
        }
    }


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    unit: str
    impact: float = Field(..., description="Effect on rain chance, in percentage points")


class PredictionOut(BaseModel):
    probability: float
    will_rain: bool
    threshold: float
    risk_level: str
    base_value: float = Field(..., description="Model's average rain chance (%) before looking at inputs")
    contributions: list[Contribution]


class BatchIn(BaseModel):
    items: list[WeatherInput] = Field(..., min_length=1, max_length=100)


class BatchItem(BaseModel):
    probability: float
    will_rain: bool
    risk_level: str


class BatchOut(BaseModel):
    results: list[BatchItem]


# ---------------------------------------------------------------- helpers
def to_frame(items: list[WeatherInput]) -> pd.DataFrame:
    rows = [[i.temperature, i.humidity, i.wind_speed, i.cloud_cover, i.pressure] for i in items]
    return pd.DataFrame(rows, columns=FEATURES)


def risk_level(prob: float, threshold: float) -> str:
    if prob >= 0.6:
        return "High"
    if prob >= threshold:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------- routes
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model_loaded": "model" in state, "threshold": state.get("threshold")}


@app.post("/predict", response_model=PredictionOut, tags=["prediction"],
          dependencies=[Depends(require_key)])
def predict(payload: WeatherInput):
    """Predict rain and explain the result with SHAP."""
    df = to_frame([payload])
    prob = float(state["model"].predict_proba(df)[0, 1])
    threshold = float(state["threshold"])

    vals, base = rain_shap(state["explainer"], df)
    contribs = [
        Contribution(
            feature=f,
            label=LABELS[f],
            value=float(df.iloc[0][f]),
            unit=UNITS[f],
            impact=round(float(vals[0, i]) * 100, 2),
        )
        for i, f in enumerate(FEATURES)
    ]
    contribs.sort(key=lambda c: abs(c.impact), reverse=True)

    return PredictionOut(
        probability=round(prob, 4),
        will_rain=prob >= threshold,
        threshold=threshold,
        risk_level=risk_level(prob, threshold),
        base_value=round(base * 100, 2),
        contributions=contribs,
    )


@app.post("/predict/batch", response_model=BatchOut, tags=["prediction"],
          dependencies=[Depends(require_key)])
def predict_batch(payload: BatchIn):
    """Predict for up to 100 rows at once (no SHAP, for speed)."""
    probs = state["model"].predict_proba(to_frame(payload.items))[:, 1]
    t = float(state["threshold"])
    return BatchOut(results=[
        BatchItem(probability=round(float(p), 4), will_rain=bool(p >= t), risk_level=risk_level(float(p), t))
        for p in probs
    ])


@app.get("/model/info", tags=["model"], dependencies=[Depends(require_key)])
def model_info():
    """Test-set metrics, curves, and feature importance saved by train.py."""
    if not state.get("metrics"):
        raise HTTPException(404, "model_metrics.json not found. Run `python train.py`.")
    return state["metrics"]