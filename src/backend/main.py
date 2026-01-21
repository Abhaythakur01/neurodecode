"""
FastAPI application for the NeuroDecode BCI backend.

Entry point for the real-time neural decoding web service.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.routers import health_router, simulation_router, websocket_router
from src.backend.config import settings
from src.backend.services.data_simulator import simulation_runner
from src.backend.services.decoder_service import decoder_service

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes services on startup and cleans up on shutdown.
    """
    # Startup
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Max latency target: {settings.max_latency_ms}ms")

    # Pre-calibrate with synthetic data for immediate availability
    if settings.skip_auto_calibration:
        logger.info("Skipping auto-calibration (SKIP_AUTO_CALIBRATION=true)")
    else:
        try:
            logger.info("Generating calibration data...")
            X_cal, y_cal = simulation_runner.generate_calibration_data(settings.calibration_samples)
            logger.info(f"Calibration data generated: X={X_cal.shape}, y={y_cal.shape}")

            logger.info("Initializing meta-learner...")
            scores = decoder_service.initialize(X_cal, y_cal)
            logger.info(f"Meta-learner initialized. Decoder scores: {scores}")
        except Exception as e:
            logger.warning(f"Auto-calibration failed: {e}. Manual calibration required.")

    logger.info(f"{settings.app_name} started successfully")

    yield

    # Shutdown
    logger.info("Shutting down...")
    simulation_runner.stop()
    decoder_service.shutdown()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Real-time neural decoding with adaptive meta-learner",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(websocket_router)
app.include_router(simulation_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "websocket_url": "/ws/decode",
        "api_prefix": settings.api_v1_prefix,
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
