"""
Lightweight demo backend for NeuroDecode BCI.

This is a minimal version without heavy ML dependencies,
suitable for portfolio deployment on free-tier platforms.
Returns simulated data that demonstrates the API structure.
"""

import asyncio
import json
import logging
import math
import os
import random  # nosec B311 - used for demo data simulation, not security
import time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import List, Optional, Set

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# CORS origins
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://neurodecode.vercel.app",
    "https://neurodecode-frontend.vercel.app",
]


def get_cors_origins() -> List[str]:
    """Get CORS origins from environment or defaults."""
    env_origins = os.environ.get("CORS_ORIGINS", "")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return DEFAULT_CORS_ORIGINS


# ============================================================================
# Schemas (Pydantic models)
# ============================================================================


class MessageType(str, Enum):
    NEURAL_DATA = "neural_data"
    PREDICTION = "prediction"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    STATUS = "status"


class SimulationState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    CALIBRATING = "calibrating"


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    meta_learner_ready: bool
    demo_mode: bool = True


class DecoderInfo(BaseModel):
    name: str
    type: str
    is_active: bool
    score: float
    latency_ms: float
    weight: float


class SimulationConfig(BaseModel):
    n_neurons: int = 50
    noise_level: float = 0.1
    pattern: str = "circle"
    speed: float = 1.0


class SimulationStartRequest(BaseModel):
    config: Optional[SimulationConfig] = None


class SimulationStartResponse(BaseModel):
    status: str
    message: str
    config: SimulationConfig


class SimulationStopResponse(BaseModel):
    status: str
    message: str
    total_predictions: int = 0
    average_latency_ms: float = 0.0


class CalibrationRequest(BaseModel):
    n_samples: int = 500
    include_decoders: Optional[List[str]] = None


class CalibrationResponse(BaseModel):
    status: str
    message: str
    calibration_time_ms: float
    decoder_scores: dict


class DecoderListResponse(BaseModel):
    decoders: List[DecoderInfo]
    meta_learner_state: dict


# ============================================================================
# Demo Data Generator
# ============================================================================


class DemoSimulator:
    """Generates realistic-looking demo data."""

    # Decoder configurations for demo
    DECODERS = [
        ("KalmanFilter", "classic", 0.85),
        ("WienerFilter", "classic", 0.78),
        ("LDA", "classic", 0.72),
        ("SVM_RBF", "ml", 0.81),
        ("RandomForest", "ml", 0.79),
        ("XGBoost", "ml", 0.83),
        ("GaussianProcess", "ml", 0.80),
        ("LSTM", "deep_learning", 0.88),
        ("Transformer", "deep_learning", 0.91),
        ("TCN", "deep_learning", 0.86),
        ("VAE", "deep_learning", 0.82),
        ("HMM", "classic", 0.75),
        ("MetaLearner", "ensemble", 0.93),
    ]

    def __init__(self):
        self.t = 0.0
        self.is_calibrated = False
        self.prediction_count = 0

    def generate_position(self, dt: float = 0.02) -> tuple:
        """Generate smooth circular/figure-8 trajectory."""
        self.t += dt

        # Figure-8 pattern
        x = math.sin(self.t * 0.5) * 0.8
        y = math.sin(self.t) * 0.4

        # Add small noise
        x += random.gauss(0, 0.02)
        y += random.gauss(0, 0.02)

        return (x, y)

    def generate_prediction_response(self) -> dict:
        """Generate a mock prediction response."""
        self.prediction_count += 1
        pos = self.generate_position()

        # Select top 3 decoders randomly weighted by score
        active_decoders = random.sample(self.DECODERS[:11], 3)
        weights = [d[2] for d in active_decoders]
        total = sum(weights)
        weights = [w / total for w in weights]

        return {
            "type": "prediction",
            "timestamp": int(time.time() * 1000),
            "prediction": list(pos),
            "uncertainty": [random.uniform(0.05, 0.15), random.uniform(0.05, 0.15)],
            "selected_decoders": [d[0] for d in active_decoders],
            "decoder_weights": weights,
            "latency_ms": random.uniform(8, 25),
        }

    def get_decoder_states(self) -> List[dict]:
        """Get mock decoder states."""
        states = []
        for name, dtype, base_score in self.DECODERS:
            states.append(
                {
                    "name": name,
                    "type": dtype,
                    "is_active": random.random() > 0.3,
                    "score": base_score + random.uniform(-0.05, 0.05),
                    "latency_ms": random.uniform(5, 30),
                    "weight": random.uniform(0.1, 0.5),
                }
            )
        return states

    def get_calibration_scores(self) -> dict:
        """Get mock calibration scores."""
        self.is_calibrated = True
        return {name: score + random.uniform(-0.03, 0.03) for name, _, score in self.DECODERS}


# Global simulator
simulator = DemoSimulator()

# Simulation state
simulation_state = SimulationState.STOPPED


# ============================================================================
# WebSocket Manager
# ============================================================================


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()


# ============================================================================
# Routers
# ============================================================================

health_router = APIRouter(tags=["health"])
simulation_router = APIRouter(prefix="/api/simulation", tags=["simulation"])
websocket_router = APIRouter(tags=["websocket"])


# Health endpoints
@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        meta_learner_ready=simulator.is_calibrated,
        demo_mode=True,
    )


@health_router.get("/ready")
async def readiness_check():
    return {
        "ready": True,
        "meta_learner_initialized": simulator.is_calibrated,
        "average_latency_ms": 15.0,
        "demo_mode": True,
    }


# Simulation endpoints
@simulation_router.post("/start", response_model=SimulationStartResponse)
async def start_simulation(request: SimulationStartRequest = None):
    global simulation_state
    config = request.config if request and request.config else SimulationConfig()
    simulation_state = SimulationState.RUNNING

    if not simulator.is_calibrated:
        simulator.is_calibrated = True

    return SimulationStartResponse(
        status="started",
        message="Demo simulation started. Connect to WebSocket to receive data.",
        config=config,
    )


@simulation_router.post("/stop", response_model=SimulationStopResponse)
async def stop_simulation():
    global simulation_state
    count = simulator.prediction_count
    simulation_state = SimulationState.STOPPED

    return SimulationStopResponse(
        status="stopped",
        message="Demo simulation stopped",
        total_predictions=count,
        average_latency_ms=15.0,
    )


@simulation_router.get("/status")
async def get_simulation_status():
    return {
        "state": simulation_state.value,
        "is_running": simulation_state == SimulationState.RUNNING,
        "sample_count": simulator.prediction_count,
        "decoder_ready": simulator.is_calibrated,
        "average_latency_ms": 15.0,
        "predictions_per_second": 50.0,
        "demo_mode": True,
    }


@simulation_router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_decoders(request: CalibrationRequest = None):
    global simulation_state
    simulation_state = SimulationState.CALIBRATING

    # Simulate calibration delay
    await asyncio.sleep(0.5)

    scores = simulator.get_calibration_scores()
    simulation_state = SimulationState.STOPPED

    return CalibrationResponse(
        status="completed",
        message=f"Demo calibration completed with {len(scores)} decoders",
        calibration_time_ms=500.0,
        decoder_scores=scores,
    )


@simulation_router.get("/decoders", response_model=DecoderListResponse)
async def list_decoders():
    return DecoderListResponse(
        decoders=[DecoderInfo(**d) for d in simulator.get_decoder_states()],
        meta_learner_state={
            "top_k": 3,
            "ensemble_method": "weighted_average",
            "adaptation_rate": 0.1,
        },
    )


# WebSocket endpoint
@websocket_router.websocket("/ws/decode")
async def websocket_decode(websocket: WebSocket):
    await manager.connect(websocket)

    simulation_running = False
    heartbeat_interval = 10.0
    simulation_rate = 50.0  # Hz

    async def send_heartbeat():
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "timestamp": int(time.time() * 1000),
                    }
                )
        except asyncio.CancelledError:
            pass  # Expected on shutdown
        except Exception:
            pass  # nosec B110 - intentional silent fail for background heartbeat

    async def run_simulation():
        nonlocal simulation_running
        interval = 1.0 / simulation_rate
        last_state_time = 0.0

        while simulation_running:
            try:
                response = simulator.generate_prediction_response()

                # Add decoder states periodically
                current_time = time.time()
                if current_time - last_state_time > 1.0:
                    response["decoder_states"] = simulator.get_decoder_states()
                    last_state_time = current_time

                await websocket.send_json(response)
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Simulation error: {e}")
                simulation_running = False
                break

    heartbeat_task = None
    simulation_task = None

    try:
        heartbeat_task = asyncio.create_task(send_heartbeat())

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=heartbeat_interval * 3,
                )
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "start_simulation":
                    if not simulation_running:
                        if not simulator.is_calibrated:
                            simulator.is_calibrated = True

                        simulation_running = True
                        simulation_task = asyncio.create_task(run_simulation())
                        await websocket.send_json(
                            {
                                "type": "status",
                                "message": "Demo simulation started",
                            }
                        )

                elif msg_type == "stop_simulation":
                    simulation_running = False
                    if simulation_task:
                        simulation_task.cancel()
                        try:
                            await simulation_task
                        except asyncio.CancelledError:
                            pass
                        simulation_task = None

                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Demo simulation stopped",
                        }
                    )

                elif msg_type == "heartbeat":
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "timestamp": int(time.time() * 1000),
                        }
                    )

            except asyncio.TimeoutError:
                try:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                except Exception:
                    break

            except json.JSONDecodeError as e:
                await websocket.send_json(
                    {
                        "type": "error",
                        "timestamp": int(time.time() * 1000),
                        "error": "Invalid JSON",
                        "details": str(e),
                    }
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        simulation_running = False
        if heartbeat_task:
            heartbeat_task.cancel()
        if simulation_task:
            simulation_task.cancel()
        await manager.disconnect(websocket)


@websocket_router.get("/ws/status")
async def websocket_status():
    return {
        "active_connections": manager.connection_count,
        "decoder_ready": simulator.is_calibrated,
        "demo_mode": True,
    }


# ============================================================================
# Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NeuroDecode Demo Backend...")
    logger.info("Demo mode: Heavy ML libraries not loaded")
    simulator.is_calibrated = True  # Pre-calibrated for demo
    logger.info("Demo backend started successfully")
    yield
    logger.info("Shutting down demo backend...")


app = FastAPI(
    title="NeuroDecode BCI Demo",
    description="Lightweight demo backend for portfolio deployment",
    version="1.0.0-demo",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
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
    return {
        "name": "NeuroDecode BCI Demo",
        "version": "1.0.0-demo",
        "docs_url": "/docs",
        "health_url": "/health",
        "websocket_url": "/ws/decode",
        "demo_mode": True,
        "note": "This is a lightweight demo. Full ML backend available in development mode.",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104 - required for container
