/**
 * StatisticsSummary - Key metrics cards component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import type { SessionStatistics } from '../../types';

interface StatisticsSummaryProps {
  statistics: SessionStatistics | null;
}

interface StatCardProps {
  label: string;
  value: string;
  subValue?: string;
  color?: string;
  trend?: 'up' | 'down' | 'neutral';
}

const StatCard: React.FC<StatCardProps> = ({ label, value, subValue, color, trend }) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={{ color: color || theme.text }}>
        {value}
        {trend && (
          <span className={`trend-indicator ${trend}`}>
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '–'}
          </span>
        )}
      </span>
      {subValue && <span className="stat-sub">{subValue}</span>}
      <style>{`
        .stat-card {
          display: flex;
          flex-direction: column;
          gap: 4px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
          border-radius: 8px;
        }

        .stat-label {
          font-size: 11px;
          color: ${theme.textMuted};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .stat-value {
          font-size: 20px;
          font-weight: 600;
          font-family: monospace;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .stat-sub {
          font-size: 11px;
          color: ${theme.textMuted};
        }

        .trend-indicator {
          font-size: 14px;
        }

        .trend-indicator.up { color: #00ff88; }
        .trend-indicator.down { color: #ff6b6b; }
        .trend-indicator.neutral { color: ${theme.textMuted}; }
      `}</style>
    </div>
  );
};

export const StatisticsSummary: React.FC<StatisticsSummaryProps> = ({ statistics }) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  if (!statistics) {
    return (
      <div className="statistics-summary empty">
        <p>No data available</p>
        <style>{`
          .statistics-summary.empty {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 120px;
            color: ${theme.textMuted};
            font-size: 13px;
          }
        `}</style>
      </div>
    );
  }

  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getR2Color = (r2: number): string => {
    if (r2 >= 0.8) return '#00ff88';
    if (r2 >= 0.6) return '#ffc107';
    return '#ff6b6b';
  };

  const getLatencyColor = (latency: number): string => {
    if (latency <= 30) return '#00ff88';
    if (latency <= 50) return '#ffc107';
    return '#ff6b6b';
  };

  return (
    <div className="statistics-summary">
      <div className="stats-grid">
        <StatCard
          label="Average R²"
          value={statistics.averageR2.toFixed(3)}
          subValue={`±${statistics.stdR2.toFixed(3)}`}
          color={getR2Color(statistics.averageR2)}
        />
        <StatCard
          label="R² Range"
          value={`${statistics.minR2.toFixed(2)} - ${statistics.maxR2.toFixed(2)}`}
        />
        <StatCard
          label="Avg Latency"
          value={`${statistics.averageLatency.toFixed(1)}ms`}
          subValue={`p95: ${statistics.p95Latency.toFixed(1)}ms`}
          color={getLatencyColor(statistics.averageLatency)}
        />
        <StatCard
          label="Latency Range"
          value={`${statistics.minLatency.toFixed(0)} - ${statistics.maxLatency.toFixed(0)}ms`}
        />
        <StatCard
          label="Total Frames"
          value={statistics.totalFrames.toLocaleString()}
        />
        <StatCard
          label="Duration"
          value={formatDuration(statistics.duration)}
        />
        <StatCard
          label="Avg Uncertainty"
          value={statistics.averageUncertainty.toFixed(4)}
        />
        <StatCard
          label="Frame Rate"
          value={`${(statistics.totalFrames / (statistics.duration / 1000)).toFixed(1)} fps`}
        />
      </div>

      <style>{`
        .statistics-summary {
          width: 100%;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 12px;
        }
      `}</style>
    </div>
  );
};

export default StatisticsSummary;
