"""FastAPI application for local PromptShield inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from promptshield.inference import (
    DEFAULT_THRESHOLD,
    MAX_PROMPT_LENGTH,
    DebertaInference,
)

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "deberta-v3-small"
)


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)

    @field_validator("prompt")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace characters")
        return value


def create_app(
    model_dir: Path = DEFAULT_MODEL_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "auto",
) -> FastAPI:
    service = DebertaInference(model_dir, threshold=threshold, device=device)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            service.load()
        except (
            FileNotFoundError,
            ImportError,
            RuntimeError,
            OSError,
            ValueError,
        ) as error:
            service.load_error = str(error)
        yield

    app = FastAPI(
        title="PromptShield Inference API",
        description="Local-only inference for the saved PromptShield classifier.",
        lifespan=lifespan,
    )
    app.state.inference = service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "loaded": service.loaded,
            "device": service.device,
            "model_id": "microsoft/deberta-v3-small",
            "load_error": service.load_error,
        }

    @app.post("/predict")
    def predict(request: PromptRequest) -> dict[str, Any]:
        if not service.loaded:
            raise HTTPException(
                status_code=503,
                detail="model is unavailable for inference",
            )
        try:
            result = service.predict(request.prompt)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "malicious_probability": result.malicious_probability,
            "decision": result.decision,
            "threshold": result.threshold,
            "threshold_status": (
                "configurable experimental threshold selected from validation "
                "experiments, not independent test results"
            ),
        }

    return app


app = create_app()
