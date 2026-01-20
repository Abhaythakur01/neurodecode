/**
 * Statistics utilities for analytics calculations.
 */

import type { SessionStatistics, DecoderComparison, TrendPoint, RecordingFrame } from '../types';

export function calculateMean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

export function calculateStd(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = calculateMean(values);
  const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
  return Math.sqrt(calculateMean(squaredDiffs));
}

export function calculatePercentile(values: number[], percentile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil((percentile / 100) * sorted.length) - 1;
  return sorted[Math.max(0, index)];
}

export function calculateMovingAverage(values: number[], windowSize: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - windowSize + 1);
    const window = values.slice(start, i + 1);
    result.push(calculateMean(window));
  }
  return result;
}

export function calculateSessionStatistics(frames: RecordingFrame[]): SessionStatistics {
  if (frames.length === 0) {
    return {
      totalFrames: 0,
      duration: 0,
      averageR2: 0,
      minR2: 0,
      maxR2: 0,
      stdR2: 0,
      averageLatency: 0,
      minLatency: 0,
      maxLatency: 0,
      p95Latency: 0,
      averageUncertainty: 0,
    };
  }

  const r2Values = frames.map((f) => f.metrics.r2);
  const latencyValues = frames.map((f) => f.metrics.latency);
  const uncertaintyValues = frames.map((f) =>
    (Math.abs(f.uncertainty[0]) + Math.abs(f.uncertainty[1])) / 2
  );

  const duration = frames[frames.length - 1].timestamp - frames[0].timestamp;

  return {
    totalFrames: frames.length,
    duration,
    averageR2: calculateMean(r2Values),
    minR2: Math.min(...r2Values),
    maxR2: Math.max(...r2Values),
    stdR2: calculateStd(r2Values),
    averageLatency: calculateMean(latencyValues),
    minLatency: Math.min(...latencyValues),
    maxLatency: Math.max(...latencyValues),
    p95Latency: calculatePercentile(latencyValues, 95),
    averageUncertainty: calculateMean(uncertaintyValues),
  };
}

export function calculateDecoderComparison(frames: RecordingFrame[]): DecoderComparison[] {
  if (frames.length === 0) return [];

  const decoderStats: Record<
    string,
    { weights: number[]; r2Values: number[]; usageCount: number }
  > = {};

  frames.forEach((frame) => {
    frame.decoders.forEach((decoder) => {
      if (!decoderStats[decoder]) {
        decoderStats[decoder] = { weights: [], r2Values: [], usageCount: 0 };
      }
      const weight = frame.weights[decoder] || 0;
      decoderStats[decoder].weights.push(weight);
      decoderStats[decoder].r2Values.push(frame.metrics.r2 * weight);
      decoderStats[decoder].usageCount++;
    });
  });

  return Object.entries(decoderStats)
    .map(([name, stats]) => ({
      name,
      averageWeight: calculateMean(stats.weights),
      averageR2: stats.weights.length > 0
        ? calculateMean(stats.r2Values) / calculateMean(stats.weights)
        : 0,
      usagePercent: (stats.usageCount / frames.length) * 100,
    }))
    .sort((a, b) => b.averageWeight - a.averageWeight);
}

export function calculateTrend(
  frames: RecordingFrame[],
  getValue: (frame: RecordingFrame) => number,
  windowSize: number = 10
): TrendPoint[] {
  if (frames.length === 0) return [];

  const values = frames.map(getValue);
  const movingAvg = calculateMovingAverage(values, windowSize);

  return frames.map((frame, i) => ({
    timestamp: frame.timestamp,
    value: values[i],
    movingAverage: movingAvg[i],
  }));
}
