/**
 * App - Main application component for NeuroDecode BCI.
 *
 * Integrates all visualization components and manages application state.
 */

import React, { useCallback, useEffect, useState } from 'react';
import CursorCanvas from './components/CursorCanvas';
import PerformanceChart from './components/PerformanceChart';
import DecoderPanel from './components/DecoderPanel';
import Controls from './components/Controls';
import { useWebSocket } from './hooks/useWebSocket';
import type {
  DecoderInfo,
  PerformanceMetrics,
  PredictionResponse,
  SimulationConfig,
  TrajectoryPoint,
} from './types';

const MAX_TRAJECTORY_LENGTH = 200;
const MAX_METRICS_LENGTH = 500;

const DEFAULT_CONFIG: SimulationConfig = {
  pattern: 'circular',
  speed: 1.0,
  noise_level: 0.1,
  n_neurons: 50,
};

function App() {
  // Cursor state
  const [position, setPosition] = useState<[number, number]>([0, 0]);
  const [uncertainty, setUncertainty] = useState<[number, number]>([0.1, 0.1]);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);

  // Decoder state
  const [decoders, setDecoders] = useState<DecoderInfo[]>([]);
  const [selectedDecoders, setSelectedDecoders] = useState<string[]>([]);
  const [decoderWeights, setDecoderWeights] = useState<Record<string, number>>({});

  // Performance metrics
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    r2Scores: [],
    latencies: [],
    timestamps: [],
  });

  // Simulation state
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [config, setConfig] = useState<SimulationConfig>(DEFAULT_CONFIG);

  // Handle incoming predictions
  const handlePrediction = useCallback((prediction: PredictionResponse) => {
    // Update position and uncertainty
    setPosition(prediction.prediction);
    setUncertainty(prediction.uncertainty);

    // Add to trajectory
    setTrajectory((prev) => {
      const newPoint: TrajectoryPoint = {
        x: prediction.prediction[0],
        y: prediction.prediction[1],
        timestamp: prediction.timestamp,
        uncertainty: prediction.uncertainty,
      };
      const updated = [...prev, newPoint];
      return updated.slice(-MAX_TRAJECTORY_LENGTH);
    });

    // Update decoder info
    setSelectedDecoders(prediction.selected_decoders);
    setDecoderWeights(prediction.decoder_weights);

    // Update metrics
    setMetrics((prev) => {
      // Calculate average R² from decoder weights (approximation)
      const avgWeight = Object.values(prediction.decoder_weights).reduce(
        (a, b) => a + b,
        0
      );

      const newR2 = [...prev.r2Scores, avgWeight > 0 ? avgWeight : 0.5];
      const newLatencies = [...prev.latencies, prediction.latency_ms];
      const newTimestamps = [...prev.timestamps, prediction.timestamp];

      return {
        r2Scores: newR2.slice(-MAX_METRICS_LENGTH),
        latencies: newLatencies.slice(-MAX_METRICS_LENGTH),
        timestamps: newTimestamps.slice(-MAX_METRICS_LENGTH),
      };
    });
  }, []);

  // Handle decoder state updates
  const handleDecoderStates = useCallback((states: DecoderInfo[]) => {
    setDecoders(states);
  }, []);

  // Handle errors
  const handleError = useCallback((error: string) => {
    console.error('WebSocket error:', error);
  }, []);

  // WebSocket hook
  const {
    connectionState,
    connect,
    disconnect,
    startSimulation,
    stopSimulation,
  } = useWebSocket({
    onPrediction: handlePrediction,
    onDecoderStates: handleDecoderStates,
    onError: handleError,
    autoConnect: false,
  });

  // Handle simulation start/stop
  const handleStartSimulation = useCallback(() => {
    startSimulation();
    setIsSimulationRunning(true);
  }, [startSimulation]);

  const handleStopSimulation = useCallback(() => {
    stopSimulation();
    setIsSimulationRunning(false);
  }, [stopSimulation]);

  // Handle calibration
  const handleCalibrate = useCallback(async () => {
    try {
      const response = await fetch('/api/simulation/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: 500 }),
      });
      const data = await response.json();
      console.log('Calibration result:', data);
    } catch (err) {
      console.error('Calibration failed:', err);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          if (isSimulationRunning) {
            handleStopSimulation();
          } else if (connectionState.isConnected) {
            handleStartSimulation();
          }
          break;
        case 'KeyC':
          if (connectionState.isConnected && !isSimulationRunning) {
            handleCalibrate();
          }
          break;
        case 'KeyR':
          if (!connectionState.isConnected) {
            connect();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    connectionState.isConnected,
    isSimulationRunning,
    handleStartSimulation,
    handleStopSimulation,
    handleCalibrate,
    connect,
  ]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroDecode BCI</h1>
        <span className="subtitle">Real-time Neural Decoding Interface</span>
      </header>

      <main className="app-main">
        <div className="left-panel">
          <CursorCanvas
            position={position}
            uncertainty={uncertainty}
            trajectory={trajectory}
            maxTrailLength={100}
            width={500}
            height={500}
          />
        </div>

        <div className="right-panel">
          <Controls
            connectionState={connectionState}
            isSimulationRunning={isSimulationRunning}
            onConnect={connect}
            onDisconnect={disconnect}
            onStartSimulation={handleStartSimulation}
            onStopSimulation={handleStopSimulation}
            onCalibrate={handleCalibrate}
            config={config}
            onConfigChange={setConfig}
          />

          <PerformanceChart
            metrics={metrics}
            maxPoints={100}
            width={450}
            height={250}
            latencyThreshold={50}
          />

          <DecoderPanel
            decoders={decoders}
            selectedDecoders={selectedDecoders}
            decoderWeights={decoderWeights}
          />
        </div>
      </main>

      <style>{`
        .app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background: #0a0a0f;
        }

        .app-header {
          padding: 15px 30px;
          background: rgba(20, 20, 30, 0.8);
          border-bottom: 1px solid #222;
          display: flex;
          align-items: baseline;
          gap: 15px;
        }

        .app-header h1 {
          margin: 0;
          font-size: 24px;
          font-weight: 600;
          color: #00ff88;
          letter-spacing: -0.5px;
        }

        .subtitle {
          font-size: 14px;
          color: #666;
        }

        .app-main {
          flex: 1;
          display: flex;
          padding: 20px;
          gap: 20px;
        }

        .left-panel {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .right-panel {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 15px;
          max-width: 500px;
        }

        @media (max-width: 1100px) {
          .app-main {
            flex-direction: column;
            align-items: center;
          }

          .right-panel {
            width: 100%;
            max-width: 500px;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
