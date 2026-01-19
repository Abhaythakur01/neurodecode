/**
 * TypeScript type definitions for the NeuroDecode BCI frontend.
 *
 * These types mirror the Pydantic schemas from the backend.
 */

// === Message Types ===

export type MessageType =
  | 'neural_data'
  | 'prediction'
  | 'status'
  | 'error'
  | 'heartbeat'
  | 'calibration';

export type SimulationState =
  | 'stopped'
  | 'running'
  | 'calibrating'
  | 'paused';

export type DecoderState =
  | 'active'
  | 'standby'
  | 'degraded'
  | 'disabled';

// === WebSocket Messages ===

export interface NeuralFrame {
  type: 'neural_data';
  timestamp: number;
  firing_rates: number[][];
}

export interface DecoderInfo {
  name: string;
  state: DecoderState;
  weight: number;
  r2_score: number;
  latency_ms: number;
  uncertainty?: number;
}

export interface PredictionResponse {
  type: 'prediction';
  timestamp: number;
  prediction: [number, number];
  uncertainty: [number, number];
  selected_decoders: string[];
  decoder_weights: Record<string, number>;
  latency_ms: number;
  decoder_states?: DecoderInfo[];
}

export interface StatusMessage {
  type: 'status';
  timestamp: number;
  simulation_state: SimulationState;
  connected_clients: number;
  predictions_per_second: number;
  average_latency_ms: number;
}

export interface ErrorMessage {
  type: 'error';
  timestamp: number;
  error: string;
  details?: string;
}

export interface HeartbeatMessage {
  type: 'heartbeat';
  timestamp: number;
}

export type WebSocketMessage =
  | PredictionResponse
  | StatusMessage
  | ErrorMessage
  | HeartbeatMessage;

// === Application State ===

export interface Point {
  x: number;
  y: number;
}

export interface TrajectoryPoint extends Point {
  timestamp: number;
  uncertainty?: [number, number];
}

export interface PerformanceMetrics {
  r2Scores: number[];
  latencies: number[];
  timestamps: number[];
}

export interface ConnectionState {
  isConnected: boolean;
  isConnecting: boolean;
  lastPingTime?: number;
  error?: string;
}

export interface SimulationConfig {
  pattern: 'circular' | 'reaching' | 'random' | 'figure_eight';
  speed: number;
  noise_level: number;
  n_neurons: number;
}

// === API Response Types ===

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  meta_learner_ready: boolean;
}

export interface SimulationStartResponse {
  status: string;
  message: string;
  config: SimulationConfig;
}

export interface SimulationStopResponse {
  status: string;
  message: string;
  total_predictions: number;
  average_latency_ms: number;
}

export interface CalibrationResponse {
  status: string;
  message: string;
  calibration_time_ms: number;
  decoder_scores: Record<string, number>;
}

// === Chart Data Types ===

export interface ChartData {
  x: number[];
  y: number[];
}

export interface DecoderChartData {
  name: string;
  weights: number[];
  scores: number[];
  timestamps: number[];
}
