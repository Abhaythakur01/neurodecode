/**
 * TrendAnalysis - Trend visualization component.
 */

import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSettingsStore } from '../../stores/settingsStore';
import type { TrendPoint } from '../../types';

interface TrendAnalysisProps {
  r2Trend: TrendPoint[];
  latencyTrend: TrendPoint[];
  width?: number;
  height?: number;
}

export const TrendAnalysis: React.FC<TrendAnalysisProps> = ({
  r2Trend,
  latencyTrend,
  width = 400,
  height = 200,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  const r2Data = useMemo(() => {
    const timestamps = r2Trend.map((t) => t.timestamp / 1000);
    return [
      {
        x: timestamps,
        y: r2Trend.map((t) => t.value),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'R²',
        line: { color: theme.accent, width: 1 },
        opacity: 0.5,
      },
      {
        x: timestamps,
        y: r2Trend.map((t) => t.movingAverage),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'R² (MA)',
        line: { color: theme.accent, width: 2 },
      },
    ];
  }, [r2Trend, theme]);

  const latencyData = useMemo(() => {
    const timestamps = latencyTrend.map((t) => t.timestamp / 1000);
    return [
      {
        x: timestamps,
        y: latencyTrend.map((t) => t.value),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Latency',
        line: { color: '#ff6b6b', width: 1 },
        opacity: 0.5,
      },
      {
        x: timestamps,
        y: latencyTrend.map((t) => t.movingAverage),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Latency (MA)',
        line: { color: '#ff6b6b', width: 2 },
      },
    ];
  }, [latencyTrend]);

  const commonLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(20, 20, 30, 0.5)',
    font: { color: theme.textMuted, size: 10 },
    margin: { l: 45, r: 15, t: 25, b: 30 },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      zerolinecolor: 'rgba(255, 255, 255, 0.1)',
      title: { text: 'Time (s)', font: { size: 10 } },
    },
    legend: {
      orientation: 'h' as const,
      x: 0,
      y: 1.12,
      font: { size: 9 },
    },
    showlegend: true,
  };

  if (r2Trend.length === 0 || latencyTrend.length === 0) {
    return (
      <div className="trend-analysis empty">
        <p>No trend data available</p>
        <style>{`
          .trend-analysis.empty {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 150px;
            color: ${theme.textMuted};
            font-size: 13px;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="trend-analysis">
      <div className="chart-container">
        <h5>R² Score Trend</h5>
        <Plot
          data={r2Data}
          layout={{
            ...commonLayout,
            height: height * 0.9,
            width,
            yaxis: {
              gridcolor: 'rgba(255, 255, 255, 0.05)',
              zerolinecolor: 'rgba(255, 255, 255, 0.1)',
              title: { text: 'R²', font: { size: 10 } },
              range: [0, 1],
            },
          }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>

      <div className="chart-container">
        <h5>Latency Trend</h5>
        <Plot
          data={latencyData}
          layout={{
            ...commonLayout,
            height: height * 0.9,
            width,
            yaxis: {
              gridcolor: 'rgba(255, 255, 255, 0.05)',
              zerolinecolor: 'rgba(255, 255, 255, 0.1)',
              title: { text: 'ms', font: { size: 10 } },
            },
          }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>

      <style>{`
        .trend-analysis {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .chart-container h5 {
          margin: 0 0 8px 0;
          font-size: 12px;
          color: ${theme.textMuted};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
      `}</style>
    </div>
  );
};

export default TrendAnalysis;
