/**
 * DecoderComparison - Decoder comparison chart component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import type { DecoderComparison as DecoderComparisonType } from '../../types';

interface DecoderComparisonProps {
  data: DecoderComparisonType[];
}

export const DecoderComparison: React.FC<DecoderComparisonProps> = ({ data }) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  if (data.length === 0) {
    return (
      <div className="decoder-comparison empty">
        <p>No decoder data available</p>
        <style>{`
          .decoder-comparison.empty {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100px;
            color: ${theme.textMuted};
            font-size: 13px;
          }
        `}</style>
      </div>
    );
  }

  const maxWeight = Math.max(...data.map((d) => d.averageWeight));

  return (
    <div className="decoder-comparison">
      <h4>Decoder Performance</h4>
      <div className="comparison-list">
        {data.map((decoder) => (
          <div key={decoder.name} className="comparison-item">
            <div className="item-header">
              <span className="decoder-name">{decoder.name}</span>
              <span className="decoder-weight">
                {(decoder.averageWeight * 100).toFixed(1)}%
              </span>
            </div>

            <div className="weight-bar-container">
              <div
                className="weight-bar"
                style={{
                  width: `${(decoder.averageWeight / maxWeight) * 100}%`,
                }}
              />
            </div>

            <div className="item-stats">
              <span className="stat">
                Usage: <strong>{decoder.usagePercent.toFixed(0)}%</strong>
              </span>
              <span className="stat">
                R² contrib: <strong>{decoder.averageR2.toFixed(3)}</strong>
              </span>
            </div>
          </div>
        ))}
      </div>

      <style>{`
        .decoder-comparison {
          width: 100%;
        }

        .decoder-comparison h4 {
          margin: 0 0 16px 0;
          font-size: 13px;
          color: ${theme.text};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .comparison-list {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .comparison-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .item-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .decoder-name {
          font-size: 13px;
          font-weight: 500;
          color: ${theme.text};
        }

        .decoder-weight {
          font-size: 14px;
          font-family: monospace;
          font-weight: 600;
          color: ${theme.accent};
        }

        .weight-bar-container {
          height: 8px;
          background: rgba(255, 255, 255, 0.08);
          border-radius: 4px;
          overflow: hidden;
        }

        .weight-bar {
          height: 100%;
          background: linear-gradient(90deg, ${theme.accent}, ${theme.accent}80);
          border-radius: 4px;
          transition: width 0.3s ease;
        }

        .item-stats {
          display: flex;
          gap: 16px;
        }

        .item-stats .stat {
          font-size: 11px;
          color: ${theme.textMuted};
        }

        .item-stats strong {
          color: ${theme.text};
        }
      `}</style>
    </div>
  );
};

export default DecoderComparison;
