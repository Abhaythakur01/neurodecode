"""
Simulation control endpoints.

Provides REST API for starting/stopping neural data simulation
and managing calibration.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from src.backend.config import settings
from src.backend.models.schemas import (
    CalibrationRequest,
    CalibrationResponse,
    DecoderListResponse,
    SimulationConfig,
    SimulationStartRequest,
    SimulationStartResponse,
    SimulationState,
    SimulationStopResponse,
)
from src.backend.services.data_simulator import simulation_runner
from src.backend.services.decoder_service import decoder_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

# Track simulation state
_current_state = SimulationState.STOPPED


@router.post("/start", response_model=SimulationStartResponse)
async def start_simulation(
    request: SimulationStartRequest = None,
) -> SimulationStartResponse:
    """
    Start neural data simulation.

    Args:
        request: Optional simulation configuration

    Returns:
        Confirmation with active configuration
    """
    global _current_state

    if _current_state == SimulationState.RUNNING:
        raise HTTPException(status_code=400, detail="Simulation is already running")

    # Get config
    config = request.config if request and request.config else SimulationConfig()

    # Ensure decoder service is initialized
    if not decoder_service.is_ready:
        logger.info("Initializing decoder service before simulation...")
        X_cal, y_cal = simulation_runner.generate_calibration_data(settings.calibration_samples)
        decoder_service.initialize(X_cal, y_cal)

    # Configure simulator
    simulation_runner.configure(
        n_neurons=config.n_neurons,
        noise_level=config.noise_level,
        pattern=config.pattern,
        speed=config.speed,
    )

    # Note: actual start happens when WebSocket connects
    _current_state = SimulationState.RUNNING

    logger.info(f"Simulation configured: {config}")

    return SimulationStartResponse(
        status="started",
        message="Simulation started. Connect to WebSocket to receive data.",
        config=config,
    )


@router.post("/stop", response_model=SimulationStopResponse)
async def stop_simulation() -> SimulationStopResponse:
    """
    Stop neural data simulation.

    Returns:
        Statistics from the simulation session
    """
    global _current_state

    if _current_state == SimulationState.STOPPED:
        return SimulationStopResponse(
            status="already_stopped",
            message="Simulation was not running",
        )

    stats = simulation_runner.stop()
    _current_state = SimulationState.STOPPED

    return SimulationStopResponse(
        status="stopped",
        message="Simulation stopped",
        total_predictions=stats.get("total_samples", 0),
        average_latency_ms=decoder_service.average_latency,
    )


@router.get("/status")
async def get_simulation_status() -> dict:
    """
    Get current simulation status.

    Returns:
        Dictionary with simulation state and statistics
    """
    return {
        "state": _current_state.value,
        "is_running": simulation_runner.is_running,
        "sample_count": simulation_runner.sample_count,
        "decoder_ready": decoder_service.is_ready,
        "average_latency_ms": decoder_service.average_latency,
        "predictions_per_second": decoder_service.predictions_per_second,
    }


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_decoders(
    request: CalibrationRequest = None,
) -> CalibrationResponse:
    """
    Run calibration to train decoders on synthetic data.

    Args:
        request: Calibration parameters

    Returns:
        Calibration results with decoder scores
    """
    global _current_state

    if _current_state == SimulationState.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot calibrate while simulation is running",
        )

    _current_state = SimulationState.CALIBRATING

    try:
        n_samples = request.n_samples if request else settings.calibration_samples
        include_decoders = request.include_decoders if request else None

        logger.info(f"Starting calibration with {n_samples} samples...")

        # Generate calibration data
        start_time = time.perf_counter()
        X_cal, y_cal = simulation_runner.generate_calibration_data(n_samples)

        # Initialize/recalibrate decoder service
        if decoder_service.is_ready:
            scores = decoder_service.recalibrate(X_cal, y_cal, include_decoders)
        else:
            scores = decoder_service.initialize(X_cal, y_cal, include_decoders)

        calibration_time = (time.perf_counter() - start_time) * 1000

        _current_state = SimulationState.STOPPED

        return CalibrationResponse(
            status="completed",
            message=f"Calibration completed with {len(scores)} decoders",
            calibration_time_ms=calibration_time,
            decoder_scores=scores,
        )

    except Exception as e:
        _current_state = SimulationState.STOPPED
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decoders", response_model=DecoderListResponse)
async def list_decoders() -> DecoderListResponse:
    """
    List all decoders and their current states.

    Returns:
        List of decoder information
    """
    return DecoderListResponse(
        decoders=decoder_service.get_decoder_states(),
        meta_learner_state=decoder_service.get_meta_learner_state(),
    )


def get_simulation_state() -> SimulationState:
    """Get current simulation state (for use by other modules)."""
    return _current_state


def set_simulation_state(state: SimulationState):
    """Set simulation state (for use by other modules)."""
    global _current_state
    _current_state = state
