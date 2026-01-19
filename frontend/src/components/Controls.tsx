/**
 * Controls - Session control panel for the BCI interface.
 *
 * Provides buttons to start/stop simulation, calibrate, and
 * displays connection status.
 */

import React, { useState } from 'react';
import type { ConnectionState, SimulationConfig } from '../types';

interface ControlsProps {
  /** WebSocket connection state */
  connectionState: ConnectionState;
  /** Whether simulation is currently running */
  isSimulationRunning: boolean;
  /** Connect to WebSocket */
  onConnect: () => void;
  /** Disconnect from WebSocket */
  onDisconnect: () => void;
  /** Start simulation */
  onStartSimulation: () => void;
  /** Stop simulation */
  onStopSimulation: () => void;
  /** Run calibration */
  onCalibrate?: () => Promise<void>;
  /** Current simulation config */
  config?: SimulationConfig;
  /** Update simulation config */
  onConfigChange?: (config: SimulationConfig) => void;
}

export const Controls: React.FC<ControlsProps> = ({
  connectionState,
  isSimulationRunning,
  onConnect,
  onDisconnect,
  onStartSimulation,
  onStopSimulation,
  onCalibrate,
  config,
  onConfigChange,
}) => {
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [showConfig, setShowConfig] = useState(false);

  const handleCalibrate = async () => {
    if (!onCalibrate) return;
    setIsCalibrating(true);
    try {
      await onCalibrate();
    } finally {
      setIsCalibrating(false);
    }
  };

  const { isConnected, isConnecting, error } = connectionState;

  return (
    <div className="controls-panel">
      {/* Connection Status */}
      <div className="connection-status">
        <div className="status-indicator-container">
          <span
            className={`status-dot ${
              isConnected ? 'connected' : isConnecting ? 'connecting' : 'disconnected'
            }`}
          />
          <span className="status-text">
            {isConnected
              ? 'Connected'
              : isConnecting
              ? 'Connecting...'
              : 'Disconnected'}
          </span>
        </div>
        {error && <span className="error-text">{error}</span>}
      </div>

      {/* Main Controls */}
      <div className="control-buttons">
        {!isConnected ? (
          <button
            className="control-btn connect"
            onClick={onConnect}
            disabled={isConnecting}
          >
            {isConnecting ? 'Connecting...' : 'Connect'}
          </button>
        ) : (
          <button className="control-btn disconnect" onClick={onDisconnect}>
            Disconnect
          </button>
        )}

        <button
          className="control-btn calibrate"
          onClick={handleCalibrate}
          disabled={!isConnected || isCalibrating || isSimulationRunning}
        >
          {isCalibrating ? 'Calibrating...' : 'Calibrate'}
        </button>

        {!isSimulationRunning ? (
          <button
            className="control-btn start"
            onClick={onStartSimulation}
            disabled={!isConnected}
          >
            Start Simulation
          </button>
        ) : (
          <button className="control-btn stop" onClick={onStopSimulation}>
            Stop Simulation
          </button>
        )}

        <button
          className="control-btn config"
          onClick={() => setShowConfig(!showConfig)}
        >
          {showConfig ? 'Hide Config' : 'Config'}
        </button>
      </div>

      {/* Configuration Panel */}
      {showConfig && config && onConfigChange && (
        <div className="config-panel">
          <h4>Simulation Settings</h4>

          <div className="config-field">
            <label>Pattern</label>
            <select
              value={config.pattern}
              onChange={(e) =>
                onConfigChange({
                  ...config,
                  pattern: e.target.value as SimulationConfig['pattern'],
                })
              }
            >
              <option value="circular">Circular</option>
              <option value="reaching">Reaching</option>
              <option value="random">Random</option>
              <option value="figure_eight">Figure Eight</option>
            </select>
          </div>

          <div className="config-field">
            <label>Speed: {config.speed.toFixed(1)}x</label>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={config.speed}
              onChange={(e) =>
                onConfigChange({
                  ...config,
                  speed: parseFloat(e.target.value),
                })
              }
            />
          </div>

          <div className="config-field">
            <label>Noise: {(config.noise_level * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.05"
              value={config.noise_level}
              onChange={(e) =>
                onConfigChange({
                  ...config,
                  noise_level: parseFloat(e.target.value),
                })
              }
            />
          </div>

          <div className="config-field">
            <label>Neurons: {config.n_neurons}</label>
            <input
              type="range"
              min="20"
              max="100"
              step="10"
              value={config.n_neurons}
              onChange={(e) =>
                onConfigChange({
                  ...config,
                  n_neurons: parseInt(e.target.value),
                })
              }
            />
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts Help */}
      <div className="shortcuts-help">
        <span>Space: Start/Stop | C: Calibrate | R: Reconnect</span>
      </div>

      <style>{`
        .controls-panel {
          background: rgba(20, 20, 30, 0.5);
          border-radius: 8px;
          padding: 15px;
          border: 1px solid #333;
        }

        .connection-status {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 15px;
          padding-bottom: 10px;
          border-bottom: 1px solid #333;
        }

        .status-indicator-container {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          transition: all 0.3s ease;
        }

        .status-dot.connected {
          background: #00ff88;
          box-shadow: 0 0 8px #00ff88;
        }

        .status-dot.connecting {
          background: #ffc107;
          animation: pulse 1s infinite;
        }

        .status-dot.disconnected {
          background: #ff6b6b;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .status-text {
          font-size: 13px;
          color: #e0e0e0;
        }

        .error-text {
          font-size: 11px;
          color: #ff6b6b;
        }

        .control-buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 10px;
        }

        .control-btn {
          padding: 8px 16px;
          border: none;
          border-radius: 6px;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          flex: 1;
          min-width: 100px;
        }

        .control-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .control-btn.connect {
          background: #00ff88;
          color: #000;
        }

        .control-btn.connect:hover:not(:disabled) {
          background: #00cc6a;
        }

        .control-btn.disconnect {
          background: #444;
          color: #fff;
        }

        .control-btn.disconnect:hover:not(:disabled) {
          background: #555;
        }

        .control-btn.calibrate {
          background: #7c3aed;
          color: #fff;
        }

        .control-btn.calibrate:hover:not(:disabled) {
          background: #6d28d9;
        }

        .control-btn.start {
          background: #00ff88;
          color: #000;
        }

        .control-btn.start:hover:not(:disabled) {
          background: #00cc6a;
        }

        .control-btn.stop {
          background: #ff6b6b;
          color: #fff;
        }

        .control-btn.stop:hover:not(:disabled) {
          background: #ff5252;
        }

        .control-btn.config {
          background: #333;
          color: #aaa;
        }

        .control-btn.config:hover:not(:disabled) {
          background: #444;
          color: #fff;
        }

        .config-panel {
          background: rgba(30, 30, 40, 0.8);
          border-radius: 6px;
          padding: 12px;
          margin-bottom: 10px;
        }

        .config-panel h4 {
          margin: 0 0 12px 0;
          font-size: 12px;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .config-field {
          margin-bottom: 10px;
        }

        .config-field label {
          display: block;
          font-size: 12px;
          color: #aaa;
          margin-bottom: 4px;
        }

        .config-field select,
        .config-field input[type="range"] {
          width: 100%;
        }

        .config-field select {
          background: #222;
          border: 1px solid #444;
          border-radius: 4px;
          color: #e0e0e0;
          padding: 6px;
          font-size: 12px;
        }

        .config-field input[type="range"] {
          -webkit-appearance: none;
          background: #333;
          height: 4px;
          border-radius: 2px;
        }

        .config-field input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px;
          height: 14px;
          background: #00ff88;
          border-radius: 50%;
          cursor: pointer;
        }

        .shortcuts-help {
          font-size: 10px;
          color: #555;
          text-align: center;
          margin-top: 10px;
        }
      `}</style>
    </div>
  );
};

export default Controls;
