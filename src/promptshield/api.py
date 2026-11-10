"""FastAPI application for local PromptShield inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from promptshield.gateway import (
    UNAVAILABLE_DETAIL,
    GatewayCounters,
    GatewayPolicy,
    evaluate,
    evaluate_hybrid,
)
from promptshield.inference import (
    DEFAULT_THRESHOLD,
    MAX_PROMPT_LENGTH,
    DebertaInference,
    HybridInference,
)
from promptshield.review_suggestions import matched_strong_marker_rules

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "deberta-v3-small"
)
DEFAULT_SVM_ARTIFACT = (
    Path(__file__).resolve().parents[2] / "models" / "linear-svm.joblib"
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
    gateway_policy: GatewayPolicy | None = None,
    gateway_mode: str = "deberta",
    svm_artifact: Path = DEFAULT_SVM_ARTIFACT,
) -> FastAPI:
    if gateway_mode not in {"deberta", "hybrid"}:
        raise ValueError("gateway_mode must be deberta or hybrid")
    if gateway_mode == "hybrid":
        gateway_service = HybridInference(
            model_dir,
            svm_artifact,
            threshold=threshold,
            device=device,
        )
        service = gateway_service.deberta
    else:
        service = DebertaInference(model_dir, threshold=threshold, device=device)
        gateway_service = service
    policy = gateway_policy or GatewayPolicy()
    counters = GatewayCounters()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            gateway_service.load()
        except (
            FileNotFoundError,
            ImportError,
            RuntimeError,
            OSError,
            ValueError,
        ) as error:
            gateway_service.load_error = str(error)
        yield

    app = FastAPI(
        title="PromptShield Inference API",
        description=(
            "Local-only inference for the saved PromptShield classifier, "
            "plus the band_v1 protective gateway endpoint."
        ),
        lifespan=lifespan,
    )
    app.state.inference = service
    app.state.gateway_policy = policy
    app.state.gateway_counters = counters

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "loaded": gateway_service.loaded,
            "device": gateway_service.device,
            "model_id": "microsoft/deberta-v3-small",
            "load_error": gateway_service.load_error,
            "gateway_mode": gateway_mode,
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
        except (RuntimeError, OSError) as error:
            service.load_error = type(error).__name__
            raise HTTPException(
                status_code=503,
                detail="model is unavailable for inference",
            ) from error
        return {
            "malicious_probability": result.malicious_probability,
            "decision": result.decision,
            "threshold": result.threshold,
            "threshold_status": (
                "configurable experimental threshold selected from validation "
                "experiments, not independent test results"
            ),
        }

    @app.get("/gateway/stats")
    def gateway_stats() -> dict[str, Any]:
        return counters.snapshot()

    @app.post("/gateway")
    def gateway(request: PromptRequest) -> dict[str, Any]:
        if not gateway_service.loaded:
            counters.record_unavailable()
            raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL)
        try:
            result = gateway_service.predict(request.prompt)
            marker_ids = matched_strong_marker_rules(request.prompt)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (RuntimeError, OSError) as error:
            gateway_service.load_error = type(error).__name__
            counters.record_unavailable()
            raise HTTPException(
                status_code=503, detail=UNAVAILABLE_DETAIL
            ) from error
        if gateway_mode == "hybrid":
            return evaluate_hybrid(
                policy,
                counters,
                result.deberta.malicious_probability,
                result.svm_decision,
                marker_ids,
                classifier_decision=result.deberta.decision,
                classifier_threshold=result.deberta.threshold,
            )
        return evaluate(
            policy,
            counters,
            result.malicious_probability,
            marker_ids,
            classifier_decision=result.decision,
            classifier_threshold=result.threshold,
        )

    return app


app = create_app()
