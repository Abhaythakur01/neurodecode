/**
 * PerformanceChart - Real-time performance metrics visualization.
 *
 * Displays R² scores and latency over time using Plotly.
 */

import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { PerformanceMetrics } from '../types';

interface PerformanceChartProps {
  /** Performance metrics data */
  metrics: PerformanceMetrics;
  /** Maximum number of points to display */
  maxPoints?: number;
  /** Chart width */
  width?: number;
  /** Chart height */
  height?: number;
  /** Latency threshold in ms */
  latencyThreshold?: number;
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({
  metrics,
  maxPoints = 100,
  width = 450,
  height = 200,
  latencyThreshold = 50,
}) => {
  // Calculate time axis relative to now
  const timeData = useMemo(() => {
    const now = Date.now();
    const recentTimestamps = metrics.timestamps.slice(-maxPoints);
    return recentTimestamps.map((t) => (t - now) / 1000); // Seconds ago
  }, [metrics.timestamps, maxPoints]);

  // R² score chart data
  const r2Data = useMemo(() => {
    const recentScores = metrics.r2Scores.slice(-maxPoints);
    return {
      x: timeData,
      y: recentScores,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'R²',
      line: {
        color: '#00ff88',
        width: 2,
      },
      fill: 'tozeroy' as const,
      fillcolor: 'rgba(0, 255, 136, 0.1)',
    };
  }, [metrics.r2Scores, timeData, maxPoints]);

  // Latency chart data
  const latencyData = useMemo(() => {
    const recentLatencies = metrics.latencies.slice(-maxPoints);
    return {
      x: timeData,
      y: recentLatencies,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Latency',
      line: {
        color: '#ff6b6b',
        width: 2,
      },
    };
  }, [metrics.latencies, timeData, maxPoints]);

  // Threshold line for latency
  const thresholdLine = useMemo(
    () => ({
      x: [timeData[0] || -10, timeData[timeData.length - 1] || 0],
      y: [latencyThreshold, latencyThreshold],
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: `${latencyThreshold}ms threshold`,
      line: {
        color: 'rgba(255, 200, 100, 0.7)',
        width: 1,
        dash: 'dash' as const,
      },
    }),
    [timeData, latencyThreshold]
  );

  // Common layout settings
  const commonLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(20, 20, 30, 0.8)',
    font: { color: '#888', size: 10 },
    margin: { l: 50, r: 20, t: 30, b: 30 },
    xaxis: {
      gridcolor: 'rgba(100, 100, 120, 0.2)',
      zerolinecolor: 'rgba(100, 100, 120, 0.3)',
      title: { text: 'Time (s)', font: { size: 10 } },
      range: [-10, 0],
    },
    legend: {
      orientation: 'h' as const,
      x: 0,
      y: 1.15,
      font: { size: 10 },
    },
  };

  // Current values for display
  const currentR2 = metrics.r2Scores[metrics.r2Scores.length - 1] ?? 0;
  const currentLatency = metrics.latencies[metrics.latencies.length - 1] ?? 0;
  const avgLatency =
    metrics.latencies.length > 0
      ? metrics.latencies.reduce((a, b) => a + b, 0) / metrics.latencies.length
      : 0;

  return (
    <div className="performance-chart-container">
      {/* R² Score Chart */}
      <div className="chart-section">
        <div className="chart-header">
          <span className="chart-title">Decoding Accuracy (R²)</span>
          <span
            className="chart-value"
            style={{ color: currentR2 > 0.7 ? '#00ff88' : '#ff6b6b' }}
          >
            {currentR2.toFixed(3)}
          </span>
        </div>
        <Plot
          data={[r2Data]}
          layout={{
            ...commonLayout,
            height: height * 0.45,
            width,
            yaxis: {
              gridcolor: 'rgba(100, 100, 120, 0.2)',
              zerolinecolor: 'rgba(100, 100, 120, 0.3)',
              title: { text: 'R²', font: { size: 10 } },
              range: [0, 1],
            },
          }}
          config={{
            displayModeBar: false,
            staticPlot: false,
            responsive: true,
          }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Latency Chart */}
      <div className="chart-section">
        <div className="chart-header">
          <span className="chart-title">Latency</span>
          <span
            className="chart-value"
            style={{
              color: currentLatency < latencyThreshold ? '#00ff88' : '#ff6b6b',
            }}
          >
            {currentLatency.toFixed(1)}ms (avg: {avgLatency.toFixed(1)}ms)
          </span>
        </div>
        <Plot
          data={[latencyData, thresholdLine]}
          layout={{
            ...commonLayout,
            height: height * 0.45,
            width,
            yaxis: {
              gridcolor: 'rgba(100, 100, 120, 0.2)',
              zerolinecolor: 'rgba(100, 100, 120, 0.3)',
              title: { text: 'ms', font: { size: 10 } },
              range: [0, Math.max(100, currentLatency * 1.2)],
            },
          }}
          config={{
            displayModeBar: false,
            staticPlot: false,
            responsive: true,
          }}
          style={{ width: '100%' }}
        />
      </div>

      <style>{`
        .performance-chart-container {
          background: rgba(20, 20, 30, 0.5);
          border-radius: 8px;
          padding: 10px;
          border: 1px solid #333;
        }
        .chart-section {
          margin-bottom: 10px;
        }
        .chart-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0 10px;
          margin-bottom: 5px;
        }
        .chart-title {
          font-size: 12px;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .chart-value {
          font-size: 14px;
          font-weight: 600;
          font-family: monospace;
        }
      `}</style>
    </div>
  );
};

export default PerformanceChart;
