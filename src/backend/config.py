"""
Backend configuration using Pydantic settings.

Environment variables can override defaults via .env file or system env.
"""

from functools import lru_cache
from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


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

    # CORS
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "https://neurodecode.vercel.app",
            "https://neurodecode-frontend.vercel.app",
        ],
        description="Allowed CORS origins",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            # Handle comma-separated string from environment variable
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

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
