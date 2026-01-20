/**
 * useAnalytics - Hook for calculating analytics from session data.
 */

import { useMemo } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import {
  calculateSessionStatistics,
  calculateDecoderComparison,
  calculateTrend,
} from '../utils/statistics';
import type { SessionStatistics, DecoderComparison, TrendPoint, TimeRange } from '../types';

interface UseAnalyticsReturn {
  // Statistics
  statistics: SessionStatistics | null;
  decoderComparison: DecoderComparison[];
  r2Trend: TrendPoint[];
  latencyTrend: TrendPoint[];

  // Session info
  hasData: boolean;
  sessionId: string | null;
  sessionName: string | null;
  frameCount: number;
}

export function useAnalytics(timeRange?: TimeRange): UseAnalyticsReturn {
  const { activeSession } = useSessionStore();

  const filteredFrames = useMemo(() => {
    if (!activeSession) return [];
    if (!timeRange) return activeSession.frames;

    return activeSession.frames.filter(
      (f) => f.timestamp >= timeRange.start && f.timestamp <= timeRange.end
    );
  }, [activeSession, timeRange]);

  const statistics = useMemo(() => {
    if (filteredFrames.length === 0) return null;
    return calculateSessionStatistics(filteredFrames);
  }, [filteredFrames]);

  const decoderComparison = useMemo(() => {
    return calculateDecoderComparison(filteredFrames);
  }, [filteredFrames]);

  const r2Trend = useMemo(() => {
    return calculateTrend(filteredFrames, (f) => f.metrics.r2, 20);
  }, [filteredFrames]);

  const latencyTrend = useMemo(() => {
    return calculateTrend(filteredFrames, (f) => f.metrics.latency, 20);
  }, [filteredFrames]);

  return {
    statistics,
    decoderComparison,
    r2Trend,
    latencyTrend,
    hasData: filteredFrames.length > 0,
    sessionId: activeSession?.id || null,
    sessionName: activeSession?.name || null,
    frameCount: filteredFrames.length,
  };
}

export default useAnalytics;
