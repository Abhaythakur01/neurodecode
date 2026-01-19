"""API module for the BCI backend."""

from src.backend.api.routers import health_router, simulation_router, websocket_router

__all__ = ["health_router", "simulation_router", "websocket_router"]
