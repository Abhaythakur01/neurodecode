"""
Backend configuration using Pydantic settings.

Environment variables can override defaults via .env file or system env.
"""

import json
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings

# Default CORS origins
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://neurodecode.vercel.app",
    "https://neurodecode-frontend.vercel.app",
]


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "NeuroDecode BCI Backend"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Server
    host: str = "0.0.0.0"  # nosec B104 - required for Docker/container deployment
    port: int = 8000

    # Database (optional, for future use)
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string",
    )

    # Redis (optional, for future use)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection string for caching",
    )

    # CORS - stored as string to avoid pydantic-settings JSON parsing issues
    cors_origins_str: str = Field(
        default="",
        description="Allowed CORS origins (comma-separated or JSON array)",
        alias="CORS_ORIGINS",
    )

    @computed_field
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as a list, parsing from string if needed."""
        if not self.cors_origins_str:
            return DEFAULT_CORS_ORIGINS

        value = self.cors_origins_str.strip()

        # Try JSON array first
        if value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Fall back to comma-separated
        origins = [o.strip() for o in value.split(",") if o.strip()]
        return origins if origins else DEFAULT_CORS_ORIGINS

    # BCI/Decoder Settings
    max_latency_ms: float = Field(
        default=50.0,
        description="Maximum allowed decoding latency in milliseconds",
    )
    default_bin_size_ms: int = Field(
        default=20,
        description="Default time bin size for neural data (milliseconds)",
    )
    n_neurons_default: int = Field(
        default=50,
        description="Default number of neurons for simulation",
    )

    # Meta-Learner Settings
    meta_learner_top_k: int = Field(
        default=3,
        description="Number of top decoders to select",
    )
    meta_learner_parallel: bool = Field(
        default=True,
        description="Run decoders in parallel",
    )

    # WebSocket Settings
    ws_heartbeat_interval: float = Field(
        default=10.0,
        description="WebSocket heartbeat interval in seconds",
    )

    # Simulation Settings
    simulation_rate_hz: float = Field(
        default=50.0,
        description="Simulation update rate in Hz (50 Hz = 20ms bins)",
    )
    calibration_samples: int = Field(
        default=500,
        description="Number of samples for initial calibration",
    )
    skip_auto_calibration: bool = Field(
        default=False,
        description="Skip auto-calibration on startup (for low-memory environments)",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
