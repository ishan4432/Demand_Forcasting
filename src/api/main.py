"""FastAPI serving layer — the trained model behind a web API.

Why an API at all?
    A model file on your laptop helps nobody. Wrapping it in an HTTP API
    means any system (a planning tool, a dashboard, a script) can ask
    "what's tomorrow's demand?" with a simple web request.

The one design decision that keeps latency under 100 ms:
    The model is loaded into memory ONCE, at server startup (see lifespan
    below) — never per-request. A request then only does: parse JSON ->
    build features -> one LSTM forward pass (a few ms on CPU) -> respond.

Endpoints:
    GET  /health        - liveness check (used by load balancers)
    GET  /model/info    - which model version is serving, and its metrics
    POST /predict       - forecast the next N days
    GET  /drift         - run the KS drift check on current data

Run it locally:
    uvicorn src.api.main:app --reload
    then open http://127.0.0.1:8000/docs   (interactive API playground)
"""

import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import CONFIG, ARTIFACTS_DIR, RAW_DATA_PATH
from src.features.build_features import TARGET_COL, apply_scaler, build_features
from src.models.lstm_model import LSTMForecaster
from src.monitoring.drift import detect_drift

MODEL_PATH = ARTIFACTS_DIR / "model.pt"

# Populated once at startup; shared by all requests.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load model + data into memory."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model at {MODEL_PATH}. Run: python -m src.training.train"
        )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_cfg"]
    model = LSTMForecaster(
        n_features=len(checkpoint["feat_cols"]),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    state["model"] = model
    state["feat_scaler"] = checkpoint["feat_scaler"]
    state["target_scaler"] = checkpoint["target_scaler"]
    state["feat_cols"] = checkpoint["feat_cols"]
    state["lookback"] = cfg["lookback"]
    state["metrics"] = checkpoint["metrics"]
    state["history"] = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    yield
    state.clear()


app = FastAPI(
    title="Demand Forecasting API",
    description="LSTM demand forecasts served from an MLflow-tracked model.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    days: int = Field(default=1, ge=1, le=30, description="How many days ahead to forecast")


class PredictResponse(BaseModel):
    forecasts: list[dict]
    model_test_mape: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model/info")
def model_info():
    return {
        "model": "LSTMForecaster",
        "lookback_days": state["lookback"],
        "n_features": len(state["feat_cols"]),
        "metrics": state["metrics"],
    }


def _forecast_next(df: pd.DataFrame) -> float:
    """One forward pass: features -> scale -> window -> LSTM -> unscale."""
    # Only the tail of history matters: the biggest lag (364) + the LSTM
    # lookback (60) + rolling windows. Trimming keeps feature-building fast,
    # which is part of staying under the 100 ms latency budget.
    df = df.tail(800)
    feats_df = build_features(df, CONFIG.features.lags, CONFIG.features.rolling_windows)
    window = feats_df.tail(state["lookback"])
    X = apply_scaler(window, state["feat_cols"], state["feat_scaler"])
    with torch.no_grad():
        pred_scaled = state["model"](torch.from_numpy(X[None].astype(np.float32)))
    pred = state["target_scaler"].inverse_transform(pred_scaled.numpy().reshape(-1, 1))
    return float(max(pred[0, 0], 0.0))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Forecast the next N days.

    Multi-day forecasts use the RECURSIVE strategy: predict tomorrow, append
    that prediction to the history as if it happened, then predict the day
    after. Simple and standard — with the known tradeoff that errors compound
    as the horizon grows.
    """
    start = time.perf_counter()
    df = state["history"].copy()
    forecasts = []
    for _ in range(req.days):
        next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
        pred = _forecast_next(df)
        forecasts.append({"date": str(next_date.date()), "predicted_demand": round(pred, 2)})
        df = pd.concat(
            [df, pd.DataFrame([{"date": next_date, TARGET_COL: pred, "is_holiday": 0, "is_promo": 0}])],
            ignore_index=True,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    return PredictResponse(
        forecasts=forecasts,
        model_test_mape=round(state["metrics"]["test_mape"], 2),
        latency_ms=round(latency_ms, 1),
    )


@app.get("/drift")
def drift():
    cfg = CONFIG.drift
    if "history" not in state:
        raise HTTPException(503, "model/data not loaded")
    return detect_drift(state["history"], cfg.detection_days, cfg.alpha, cfg.year_offset)
