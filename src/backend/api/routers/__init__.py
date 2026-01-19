"""API routers for the BCI backend."""

from src.backend.api.routers.health import router as health_router
from src.backend.api.routers.simulation import router as simulation_router
from src.backend.api.routers.websocket import router as websocket_router

__all__ = ["health_router", "simulation_router", "websocket_router"]
