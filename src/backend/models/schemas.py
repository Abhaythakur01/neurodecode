"""
Pydantic models for API request/response schemas.

These models define the WebSocket message formats and REST API payloads.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# === Enums ===


class MessageType(str, Enum):
    """WebSocket message types."""

    NEURAL_DATA = "neural_data"
    PREDICTION = "prediction"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CALIBRATION = "calibration"


class SimulationState(str, Enum):
    """Simulation states."""

    STOPPED = "stopped"
    RUNNING = "running"
    CALIBRATING = "calibrating"
    PAUSED = "paused"


class DecoderStateEnum(str, Enum):
    """Decoder state enumeration."""

    ACTIVE = "active"
    STANDBY = "standby"
    DEGRADED = "degraded"
    DISABLED = "disabled"


# === WebSocket Messages ===


class NeuralFrame(BaseModel):
    """
    Incoming neural data frame from client or simulation.

    Attributes:
        type: Message type (always 'neural_data')
        timestamp: Unix timestamp in milliseconds
        firing_rates: 2D array of shape (n_samples, n_neurons)
    """

    type: MessageType = MessageType.NEURAL_DATA
    timestamp: int = Field(description="Unix timestamp in milliseconds")
    firing_rates: List[List[float]] = Field(
        description="Firing rates array of shape (n_samples, n_neurons)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "neural_data",
                "timestamp": 1705678901234,
                "firing_rates": [[0.5, 0.3, 0.8, 0.2]],
            }
        }
    }


class DecoderInfo(BaseModel):
    """Information about a single decoder."""

    name: str
    state: DecoderStateEnum
    weight: float = Field(ge=0.0, le=1.0)
    r2_score: float = Field(description="Recent R-squared score")
    latency_ms: float = Field(description="Recent average latency")
    uncertainty: Optional[float] = None


class PredictionResponse(BaseModel):
    """
    Outgoing prediction response to client.

    Attributes:
        type: Message type (always 'prediction')
        timestamp: Unix timestamp in milliseconds
        prediction: Decoded position [x, y]
        uncertainty: Uncertainty estimate [x_std, y_std]
        selected_decoders: List of decoders used for this prediction
        decoder_weights: Weight assigned to each decoder
        latency_ms: Total decoding latency
    """

    type: MessageType = MessageType.PREDICTION
    timestamp: int = Field(description="Unix timestamp in milliseconds")
    prediction: List[float] = Field(
        description="Decoded position [x, y]", min_length=2, max_length=2
    )
    uncertainty: List[float] = Field(
        description="Uncertainty [x_std, y_std]", min_length=2, max_length=2
    )
    selected_decoders: List[str] = Field(description="Names of decoders used")
    decoder_weights: Dict[str, float] = Field(description="Weight for each selected decoder")
    latency_ms: float = Field(description="Total decoding latency in ms")
    decoder_states: Optional[List[DecoderInfo]] = Field(
        default=None, description="Full decoder state info (sent periodically)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "prediction",
                "timestamp": 1705678901256,
                "prediction": [0.23, -0.15],
                "uncertainty": [0.05, 0.04],
                "selected_decoders": ["Kalman", "SVM"],
                "decoder_weights": {"Kalman": 0.6, "SVM": 0.4},
                "latency_ms": 22.5,
            }
        }
    }


class StatusMessage(BaseModel):
    """Status update message."""

    type: MessageType = MessageType.STATUS
    timestamp: int
    simulation_state: SimulationState
    connected_clients: int = 0
    predictions_per_second: float = 0.0
    average_latency_ms: float = 0.0


class ErrorMessage(BaseModel):
    """Error message."""

    type: MessageType = MessageType.ERROR
    timestamp: int
    error: str
    details: Optional[str] = None


class HeartbeatMessage(BaseModel):
    """Heartbeat message for connection keep-alive."""

    type: MessageType = MessageType.HEARTBEAT
    timestamp: int


# === REST API Models ===


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    meta_learner_ready: bool = False


class SimulationConfig(BaseModel):
    """Configuration for simulation."""

    pattern: str = Field(
        default="circular",
        description="Movement pattern: 'circular', 'reaching', 'random'",
    )
    speed: float = Field(default=1.0, ge=0.1, le=5.0, description="Movement speed multiplier")
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0, description="Neural noise level")
    n_neurons: int = Field(default=50, ge=10, le=200, description="Number of neurons")


class SimulationStartRequest(BaseModel):
    """Request to start simulation."""

    config: Optional[SimulationConfig] = None


class SimulationStartResponse(BaseModel):
    """Response after starting simulation."""

    status: str
    message: str
    config: SimulationConfig


class SimulationStopResponse(BaseModel):
    """Response after stopping simulation."""

    status: str
    message: str
    total_predictions: int = 0
    average_latency_ms: float = 0.0


class DecoderListResponse(BaseModel):
    """List of all decoders and their states."""

    decoders: List[DecoderInfo]
    meta_learner_state: Dict[str, Any]


class CalibrationRequest(BaseModel):
    """Request for recalibration."""

    n_samples: int = Field(default=500, ge=100, le=5000)
    include_decoders: Optional[List[str]] = None


class CalibrationResponse(BaseModel):
    """Response after calibration."""

    status: str
    message: str
    calibration_time_ms: float
    decoder_scores: Dict[str, float]
