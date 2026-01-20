"""
WebSocket endpoint for real-time neural decoding.

Handles bidirectional communication for:
- Receiving neural data frames
- Sending prediction responses
- Broadcasting simulation updates
"""

import asyncio
import json
import logging
import time
from typing import Set

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.backend.config import settings
from src.backend.models.schemas import (
    ErrorMessage,
    HeartbeatMessage,
    MessageType,
    NeuralFrame,
    PredictionResponse,
)
from src.backend.services.data_simulator import simulation_runner
from src.backend.services.decoder_service import decoder_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections for broadcasting."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and register a new connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a connection."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


def process_neural_frame(frame: NeuralFrame) -> PredictionResponse:
    """
    Process a neural data frame and return prediction.

    Args:
        frame: Incoming neural data frame

    Returns:
        Prediction response with decoded position and metadata
    """
    # Convert to numpy
    firing_rates = np.array(frame.firing_rates)

    # Ensure 2D
    if firing_rates.ndim == 1:
        firing_rates = firing_rates.reshape(1, -1)

    # Run decoding
    result, latency_ms = decoder_service.decode(firing_rates)

    # Extract prediction (handle batch predictions)
    prediction = result.prediction
    if prediction.ndim > 1:
        prediction = prediction[-1]  # Take last sample

    # Extract uncertainty
    if result.uncertainty is not None:
        uncertainty = result.uncertainty
        if uncertainty.ndim > 1:
            uncertainty = uncertainty[-1]
        uncertainty = uncertainty.tolist()
    else:
        uncertainty = [0.1, 0.1]  # Default uncertainty

    # Build response
    response = PredictionResponse(
        type=MessageType.PREDICTION,
        timestamp=int(time.time() * 1000),
        prediction=prediction.tolist(),
        uncertainty=uncertainty,
        selected_decoders=result.selected_decoders,
        decoder_weights=result.decoder_weights,
        latency_ms=result.total_latency_ms,
    )

    return response


@router.websocket("/ws/decode")
async def websocket_decode(websocket: WebSocket):
    """
    WebSocket endpoint for real-time neural decoding.

    Accepts neural data frames and returns predictions.
    Also handles simulation mode where data is generated server-side.
    """
    await manager.connect(websocket)

    # Track state
    last_state_broadcast = 0.0
    state_broadcast_interval = 1.0
    simulation_running = False
    heartbeat_task = None

    async def send_heartbeat():
        """Send periodic heartbeats."""
        try:
            while True:
                await asyncio.sleep(settings.ws_heartbeat_interval)
                msg = HeartbeatMessage(
                    type=MessageType.HEARTBEAT,
                    timestamp=int(time.time() * 1000),
                )
                await websocket.send_json(msg.model_dump())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Heartbeat stopped: {e}")

    async def run_simulation():
        """Run simulation and send predictions."""
        nonlocal simulation_running
        interval = 1.0 / settings.simulation_rate_hz
        next_time = time.perf_counter()

        # Configure simulator if needed
        if simulation_runner._simulator is None:
            simulation_runner.configure()

        simulation_runner._simulator.reset()

        while simulation_running:
            try:
                # Generate sample
                firing_rates, position = simulation_runner._simulator.generate_sample(interval)
                timestamp = time.time() * 1000

                # Create frame and process
                frame = NeuralFrame(
                    type=MessageType.NEURAL_DATA,
                    timestamp=int(timestamp),
                    firing_rates=[firing_rates.tolist()],
                )
                response = process_neural_frame(frame)

                # Include decoder states periodically
                current_time = time.time()
                nonlocal last_state_broadcast
                if current_time - last_state_broadcast > state_broadcast_interval:
                    response.decoder_states = decoder_service.get_decoder_states()
                    last_state_broadcast = current_time

                await websocket.send_json(response.model_dump())

                # Sleep until next sample
                next_time += interval
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    next_time = time.perf_counter()

            except Exception as e:
                logger.error(f"Simulation error: {e}")
                simulation_running = False
                break

    simulation_task = None

    try:
        # Start heartbeat
        heartbeat_task = asyncio.create_task(send_heartbeat())

        # Main message loop
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.ws_heartbeat_interval * 3,
                )
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == MessageType.NEURAL_DATA.value:
                    # Process neural data from client
                    if not decoder_service.is_ready:
                        error = ErrorMessage(
                            type=MessageType.ERROR,
                            timestamp=int(time.time() * 1000),
                            error="Decoder not ready",
                            details="Run calibration first: POST /api/simulation/calibrate",
                        )
                        await websocket.send_json(error.model_dump())
                        continue

                    frame = NeuralFrame(**message)
                    response = process_neural_frame(frame)

                    # Include decoder states periodically
                    current_time = time.time()
                    if current_time - last_state_broadcast > state_broadcast_interval:
                        response.decoder_states = decoder_service.get_decoder_states()
                        last_state_broadcast = current_time

                    await websocket.send_json(response.model_dump())

                elif msg_type == MessageType.HEARTBEAT.value:
                    # Echo heartbeat
                    msg = HeartbeatMessage(
                        type=MessageType.HEARTBEAT,
                        timestamp=int(time.time() * 1000),
                    )
                    await websocket.send_json(msg.model_dump())

                elif msg_type == "start_simulation":
                    # Start server-side simulation
                    if simulation_running:
                        await websocket.send_json(
                            {"type": "status", "message": "Simulation already running"}
                        )
                        continue

                    if not decoder_service.is_ready:
                        # Auto-calibrate
                        logger.info("Auto-calibrating decoder...")
                        X_cal, y_cal = simulation_runner.generate_calibration_data(
                            settings.calibration_samples
                        )
                        decoder_service.initialize(X_cal, y_cal)

                    simulation_running = True
                    simulation_task = asyncio.create_task(run_simulation())

                    await websocket.send_json({"type": "status", "message": "Simulation started"})

                elif msg_type == "stop_simulation":
                    simulation_running = False
                    if simulation_task:
                        simulation_task.cancel()
                        try:
                            await simulation_task
                        except asyncio.CancelledError:
                            pass
                        simulation_task = None

                    await websocket.send_json({"type": "status", "message": "Simulation stopped"})

            except asyncio.TimeoutError:
                # Connection might be stale, send heartbeat
                try:
                    msg = HeartbeatMessage(
                        type=MessageType.HEARTBEAT,
                        timestamp=int(time.time() * 1000),
                    )
                    await websocket.send_json(msg.model_dump())
                except Exception:
                    break

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
                error = ErrorMessage(
                    type=MessageType.ERROR,
                    timestamp=int(time.time() * 1000),
                    error="Invalid JSON",
                    details=str(e),
                )
                await websocket.send_json(error.model_dump())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            error = ErrorMessage(
                type=MessageType.ERROR,
                timestamp=int(time.time() * 1000),
                error="Internal error",
                details=str(e),
            )
            await websocket.send_json(error.model_dump())
        except Exception:
            pass

    finally:
        # Cleanup
        simulation_running = False
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        if simulation_task:
            simulation_task.cancel()
            try:
                await simulation_task
            except asyncio.CancelledError:
                pass
        await manager.disconnect(websocket)


@router.get("/ws/status")
async def websocket_status() -> dict:
    """Get WebSocket connection status."""
    return {
        "active_connections": manager.connection_count,
        "decoder_ready": decoder_service.is_ready,
    }
