/**
 * DecoderPanel - Displays decoder states, weights, and performance.
 *
 * Shows which decoders are active and their contribution to the ensemble.
 */

import React from 'react';
import type { DecoderInfo, DecoderState } from '../types';

interface DecoderPanelProps {
  /** List of decoder information */
  decoders: DecoderInfo[];
  /** Currently selected decoders for ensemble */
  selectedDecoders: string[];
  /** Weights assigned to each decoder */
  decoderWeights: Record<string, number>;
}

const STATE_COLORS: Record<DecoderState, string> = {
  active: '#00ff88',
  standby: '#888888',
  degraded: '#ff9f43',
  disabled: '#ff6b6b',
};

const STATE_LABELS: Record<DecoderState, string> = {
  active: 'Active',
  standby: 'Standby',
  degraded: 'Degraded',
  disabled: 'Disabled',
};

export const DecoderPanel: React.FC<DecoderPanelProps> = ({
  decoders,
  selectedDecoders,
  decoderWeights,
}) => {
  // Sort decoders by weight (most important first)
  const sortedDecoders = [...decoders].sort((a, b) => {
    const weightA = decoderWeights[a.name] ?? 0;
    const weightB = decoderWeights[b.name] ?? 0;
    return weightB - weightA;
  });

  return (
    <div className="decoder-panel">
      <div className="panel-header">
        <h3>Decoders</h3>
        <span className="decoder-count">
          {selectedDecoders.length} / {decoders.length} active
        </span>
      </div>

      <div className="decoder-list">
        {sortedDecoders.map((decoder) => {
          const isSelected = selectedDecoders.includes(decoder.name);
          const weight = decoderWeights[decoder.name] ?? 0;
          const stateColor = STATE_COLORS[decoder.state];

          return (
            <div
              key={decoder.name}
              className={`decoder-item ${isSelected ? 'selected' : ''}`}
            >
              <div className="decoder-header">
                <div className="decoder-name">
                  <span
                    className="state-indicator"
                    style={{ backgroundColor: stateColor }}
                  />
                  <span>{decoder.name}</span>
                </div>
                <span className="decoder-state" style={{ color: stateColor }}>
                  {STATE_LABELS[decoder.state]}
                </span>
              </div>

              <div className="decoder-metrics">
                <div className="metric">
                  <span className="metric-label">R²</span>
                  <span
                    className="metric-value"
                    style={{
                      color: decoder.r2_score > 0.7 ? '#00ff88' : '#ff6b6b',
                    }}
                  >
                    {decoder.r2_score.toFixed(3)}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Latency</span>
                  <span className="metric-value">
                    {decoder.latency_ms.toFixed(1)}ms
                  </span>
                </div>
                {isSelected && (
                  <div className="metric">
                    <span className="metric-label">Weight</span>
                    <span className="metric-value weight">
                      {(weight * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>

              {isSelected && (
                <div className="weight-bar-container">
                  <div
                    className="weight-bar"
                    style={{ width: `${weight * 100}%` }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {decoders.length === 0 && (
        <div className="no-decoders">
          No decoders available. Run calibration to initialize.
        </div>
      )}

      <style>{`
        .decoder-panel {
          background: rgba(20, 20, 30, 0.5);
          border-radius: 8px;
          padding: 15px;
          border: 1px solid #333;
          min-width: 280px;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
          padding-bottom: 10px;
          border-bottom: 1px solid #333;
        }

        .panel-header h3 {
          margin: 0;
          font-size: 14px;
          color: #e0e0e0;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .decoder-count {
          font-size: 12px;
          color: #888;
        }

        .decoder-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .decoder-item {
          background: rgba(30, 30, 40, 0.5);
          border-radius: 6px;
          padding: 10px;
          border: 1px solid #2a2a3a;
          transition: all 0.2s ease;
        }

        .decoder-item.selected {
          border-color: #00ff88;
          background: rgba(0, 255, 136, 0.05);
        }

        .decoder-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .decoder-name {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 500;
          color: #e0e0e0;
        }

        .state-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .decoder-state {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .decoder-metrics {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
        }

        .metric {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .metric-label {
          font-size: 10px;
          color: #666;
          text-transform: uppercase;
        }

        .metric-value {
          font-size: 13px;
          font-family: monospace;
          color: #aaa;
        }

        .metric-value.weight {
          color: #00ff88;
          font-weight: 600;
        }

        .weight-bar-container {
          margin-top: 8px;
          height: 4px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 2px;
          overflow: hidden;
        }

        .weight-bar {
          height: 100%;
          background: linear-gradient(90deg, #00ff88, #00cc6a);
          border-radius: 2px;
          transition: width 0.3s ease;
        }

        .no-decoders {
          text-align: center;
          color: #666;
          padding: 20px;
          font-size: 13px;
        }
      `}</style>
    </div>
  );
};

export default DecoderPanel;
