/**
 * useSessionRecording - Hook for managing session recording.
 */

import { useCallback, useRef, useEffect } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import type { RecordingFrame, SessionMetadata } from '../types';

interface UseSessionRecordingOptions {
  onRecordingStart?: () => void;
  onRecordingStop?: (sessionId: string) => void;
}

interface UseSessionRecordingReturn {
  isRecording: boolean;
  recordingDuration: number;
  frameCount: number;
  startRecording: () => void;
  stopRecording: () => void;
  cancelRecording: () => void;
  recordFrame: (data: Omit<RecordingFrame, 'timestamp'>) => void;
}

export function useSessionRecording(
  options: UseSessionRecordingOptions = {}
): UseSessionRecordingReturn {
  const { onRecordingStart, onRecordingStop } = options;

  const {
    recording,
    startRecording: storeStart,
    stopRecording: storeStop,
    addFrame,
    cancelRecording: storeCancel,
    saveSession,
  } = useSessionStore();

  const metricsAccumulator = useRef<{ r2: number[]; latency: number[] }>({
    r2: [],
    latency: [],
  });

  const decodersSet = useRef<Set<string>>(new Set());

  // Calculate recording duration
  const recordingDuration = recording.isRecording && recording.startTime
    ? Date.now() - recording.startTime
    : 0;

  const startRecording = useCallback(() => {
    metricsAccumulator.current = { r2: [], latency: [] };
    decodersSet.current.clear();
    storeStart();
    onRecordingStart?.();
  }, [storeStart, onRecordingStart]);

  const stopRecording = useCallback(() => {
    if (!recording.isRecording) return;

    // Calculate metadata from accumulated metrics
    const { r2, latency } = metricsAccumulator.current;
    const metadata: SessionMetadata = {
      averageR2: r2.length > 0 ? r2.reduce((a, b) => a + b, 0) / r2.length : 0,
      averageLatency: latency.length > 0 ? latency.reduce((a, b) => a + b, 0) / latency.length : 0,
      selectedDecoders: Array.from(decodersSet.current),
    };

    const session = storeStop(metadata);
    if (session) {
      saveSession(session);
      onRecordingStop?.(session.id);
    }
  }, [recording.isRecording, storeStop, saveSession, onRecordingStop]);

  const cancelRecording = useCallback(() => {
    metricsAccumulator.current = { r2: [], latency: [] };
    decodersSet.current.clear();
    storeCancel();
  }, [storeCancel]);

  const recordFrame = useCallback(
    (data: Omit<RecordingFrame, 'timestamp'>) => {
      if (!recording.isRecording) return;

      // Accumulate metrics for metadata
      metricsAccumulator.current.r2.push(data.metrics.r2);
      metricsAccumulator.current.latency.push(data.metrics.latency);

      // Track decoders used
      data.decoders.forEach((d) => decodersSet.current.add(d));

      addFrame(data);
    },
    [recording.isRecording, addFrame]
  );

  // Auto-stop on unmount if recording
  useEffect(() => {
    return () => {
      if (recording.isRecording) {
        stopRecording();
      }
    };
  }, []);

  return {
    isRecording: recording.isRecording,
    recordingDuration,
    frameCount: recording.frames.length,
    startRecording,
    stopRecording,
    cancelRecording,
    recordFrame,
  };
}

export default useSessionRecording;
