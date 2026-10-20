"""FastAPI application exposing the TransferIQ model as a JSON API."""

from __future__ import annotations

import os
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_PATH
from src.models.predict import predict_transfer_fee

app = FastAPI(
    title="TransferIQ API",
    description="Estimate Premier League transfer fees from player-season statistics.",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRANSFERIQ_CORS_ORIGINS",
        "http://localhost:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class PlayerStats(BaseModel):
    """Validated player statistics accepted by the prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(gt=0, le=45, description="Player age in years")
    minutes: float = Field(ge=0, le=6000)
    goals_scored: float = Field(ge=0, le=100)
    assists: float = Field(ge=0, le=100)
    total_points: float = Field(ge=0, le=1000)
    goals_conceded: float = Field(ge=0, le=200)
    clean_sheets: float = Field(ge=0, le=50)
    bonus: float = Field(ge=0, le=100)
    bps: float = Field(ge=0, le=3000)
    influence: float = Field(ge=0, le=2000)
    creativity: float = Field(ge=0, le=2000)
    threat: float = Field(ge=0, le=2000)
    ict_index: float = Field(ge=0, le=600)
    red_cards: float = Field(ge=0, le=10)
    yellow_cards: float = Field(ge=0, le=30)
    selected_by_percent: float = Field(ge=0, le=100)
    now_cost: float = Field(ge=0, le=200)
    position: Literal["Goalkeeper", "Defender", "Midfielder", "Forward"]
    element_type: Literal["1", "2", "3", "4"]


class PredictionResponse(BaseModel):
    predicted_fee_eur: float
    predicted_fee_millions: float
    currency: str = "EUR"
    model_status: str = "loaded"
    disclaimer: str = (
        "Research estimate only. Transfer fees also depend on contracts, "
        "club finances, negotiations, injuries, and market conditions."
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Return service and model availability without exposing filesystem paths."""
    return {
        "status": "ok",
        "service": "TransferIQ API",
        "model_available": MODEL_PATH.exists(),
    }


@app.get("/api/v1/model")
def model_info() -> dict[str, str | bool]:
    """Expose transparent model metadata for the frontend."""
    return {
        "name": "TransferIQ Transfer Fee Estimator",
        "target": "transfer fee in euros",
        "model_available": MODEL_PATH.exists(),
        "dataset_note": "Trained on a limited historical transfer dataset.",
        "evaluation_note": (
            "Historical test performance is not a guarantee of future accuracy."
        ),
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(payload: PlayerStats) -> PredictionResponse:
    """Validate player statistics and return a model-backed estimate."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model is not installed on this API server. "
                "Run the training pipeline and ensure models/transfer_value_model.pkl "
                "is available before requesting predictions."
            ),
        )

    try:
        fee = predict_transfer_fee(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"The model could not process these statistics: {exc}",
        ) from exc
    except Exception as exc:
        # Keep implementation details out of public API responses.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check that the saved model matches the current feature pipeline.",
        ) from exc

    if not np.isfinite(fee):
        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid estimate.",
        )

    return PredictionResponse(
        predicted_fee_eur=float(fee),
        predicted_fee_millions=round(float(fee) / 1_000_000, 3),
    )
