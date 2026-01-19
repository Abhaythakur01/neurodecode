"""
Health check endpoint for the BCI backend.

Provides a simple endpoint for monitoring service health
and readiness of the meta-learner.
"""

from datetime import datetime

from fastapi import APIRouter

from src.backend.models.schemas import HealthResponse
from src.backend.services.decoder_service import decoder_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns:
        HealthResponse with service status and meta-learner readiness
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        meta_learner_ready=decoder_service.is_ready,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Check if service is ready to handle requests.

    Returns:
        Dictionary with readiness status
    """
    is_ready = decoder_service.is_ready

    return {
        "ready": is_ready,
        "meta_learner_initialized": is_ready,
        "average_latency_ms": decoder_service.average_latency if is_ready else None,
    }
