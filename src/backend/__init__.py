"""
NeuroDecode BCI Backend.

FastAPI-based backend for real-time neural decoding with WebSocket support.
"""

# Lazy import to avoid circular dependencies
# Use: from src.backend import app
__all__ = ["app"]


def __getattr__(name: str):
    """Lazy import for app to avoid circular dependencies."""
    if name == "app":
        from src.backend.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
