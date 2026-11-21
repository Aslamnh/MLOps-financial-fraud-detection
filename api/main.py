"""
API inferensi deteksi transaksi mencurigakan.

Saat start, model Production ditarik dari MLflow tracking server melalui jaringan
Docker (models:/<nama>@<alias>). Bila tracking server atau artefaknya tidak dapat
diakses, API memakai model hasil ekspor di LOCAL_MODEL_PATH sebagai cadangan.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api-service")

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detection-rf")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/models/production")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
FEATURE_COLUMNS = ["log_price", "log_volume", "high_low_spread"]

state = {
    "model": None,
    "source": None,
    "info": {},
    "loaded_at": None,
}


class Transaction(BaseModel):
    current_price: float = Field(gt=0, description="Harga terkini (USD)")
    total_volume: float = Field(ge=0, description="Volume 24 jam")
    high_24h: float = Field(gt=0)
    low_24h: float = Field(gt=0)

    @model_validator(mode="after")
    def check_range(self):
        if self.high_24h < self.low_24h:
            raise ValueError("high_24h tidak boleh lebih kecil dari low_24h")
        return self


class PredictRequest(BaseModel):
    transactions: List[Transaction] = Field(min_length=1, max_length=1000)


def build_features(transactions: List[Transaction]) -> pd.DataFrame:
    """Fitur yang sama dengan saat pelatihan."""
    df = pd.DataFrame([t.model_dump() for t in transactions]).astype("float64")
    features = pd.DataFrame(
        {
            "log_price": np.log1p(df["current_price"]),
            "log_volume": np.log1p(df["total_volume"]),
            "high_low_spread": (df["high_24h"] - df["low_24h"]) / df["current_price"],
        }
    )
    return features[FEATURE_COLUMNS]


def load_from_tracking_server() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(MODEL_URI)
    metrics = {}
    try:
        run = client.get_run(version.run_id)
        metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
    except Exception as exc:  # metrik hanya pelengkap
        log.warning("Metrik run tidak dapat dibaca: %s", exc)
    state.update(
        model=model,
        source="tracking-server",
        info={
            "model_name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": version.version,
            "run_id": version.run_id,
            "model_uri": MODEL_URI,
            "tracking_uri": TRACKING_URI,
            "metrics": metrics,
        },
        loaded_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )
    log.info("Model dimuat dari tracking server: %s (versi %s)", MODEL_URI, version.version)


def load_from_local() -> None:
    model = mlflow.sklearn.load_model(LOCAL_MODEL_PATH)
    state.update(
        model=model,
        source="local-fallback",
        info={"model_name": MODEL_NAME, "model_uri": LOCAL_MODEL_PATH},
        loaded_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )
    log.warning("Model dimuat dari cadangan lokal: %s", LOCAL_MODEL_PATH)


def load_model() -> None:
    try:
        load_from_tracking_server()
        return
    except Exception as exc:
        log.error("Gagal memuat dari tracking server (%s): %s", TRACKING_URI, exc)
    try:
        load_from_local()
    except Exception as exc:
        log.error("Gagal memuat model cadangan lokal: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    return {
        "status": "ok",
        "model_source": state["source"],
        "loaded_at": state["loaded_at"],
    }


@app.get("/model-info")
def model_info():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    return {"model_source": state["source"], "loaded_at": state["loaded_at"], **state["info"]}


@app.post("/reload")
def reload_model():
    """Muat ulang model (mis. setelah alias production dipindah di registry)."""
    load_model()
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model gagal dimuat")
    return {"model_source": state["source"], **state["info"]}


@app.post("/predict")
def predict(request: PredictRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    started = time.perf_counter()
    features = build_features(request.transactions)
    proba = model.predict_proba(features)
    classes = list(model.classes_)
    positive = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(features))
    elapsed_ms = (time.perf_counter() - started) * 1000
    results = [
        {
            "is_suspicious": int(p >= DECISION_THRESHOLD),
            "probability": round(float(p), 4),
        }
        for p in positive
    ]
    return {
        "model_version": state["info"].get("version"),
        "model_source": state["source"],
        "threshold": DECISION_THRESHOLD,
        "latency_ms": round(elapsed_ms, 2),
        "predictions": results,
    }
