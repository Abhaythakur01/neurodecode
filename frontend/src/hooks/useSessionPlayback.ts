/**
 * useSessionPlayback - Hook for managing session playback state machine.
 */

import { useCallback, useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import type { RecordingFrame, PlaybackState } from '../types';

interface UseSessionPlaybackOptions {
  onFrame?: (frame: RecordingFrame) => void;
  onPlaybackEnd?: () => void;
}

interface UseSessionPlaybackReturn {
  // State
  isPlaying: boolean;
  isPaused: boolean;
  playbackState: PlaybackState;
  currentTime: number;
  duration: number;
  speed: number;
  currentFrame: RecordingFrame | null;
  progress: number;
  hasSession: boolean;
  sessionName: string | null;

  // Actions
  play: () => void;
  pause: () => void;
  stop: () => void;
  seek: (time: number) => void;
  seekToProgress: (progress: number) => void;
  setSpeed: (speed: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
}

export function useSessionPlayback(
  options: UseSessionPlaybackOptions = {}
): UseSessionPlaybackReturn {
  const { onFrame, onPlaybackEnd } = options;

  const {
    activeSession,
    playback,
    play: storePlay,
    pause: storePause,
    stop: storeStop,
    seek: storeSeek,
    setSpeed: storeSetSpeed,
    nextFrame,
  } = useSessionStore();

  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  const currentFrame =
    activeSession && playback.currentFrameIndex < activeSession.frames.length
      ? activeSession.frames[playback.currentFrameIndex]
      : null;

  const progress =
    playback.duration > 0 ? (playback.currentTime / playback.duration) * 100 : 0;

  // Animation loop for playback
  useEffect(() => {
    if (playback.state !== 'playing' || !activeSession) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }

    const frameInterval = 1000 / 60; // Target 60fps
    lastTimeRef.current = performance.now();

    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTimeRef.current;

      if (deltaTime >= frameInterval / playback.speed) {
        lastTimeRef.current = currentTime;

        const frame = nextFrame();
        if (frame) {
          onFrame?.(frame);
        } else {
          // Playback ended
          onPlaybackEnd?.();
          return;
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [playback.state, playback.speed, activeSession, nextFrame, onFrame, onPlaybackEnd]);

  const play = useCallback(() => {
    if (activeSession) {
      storePlay();
    }
  }, [activeSession, storePlay]);

  const seekToProgress = useCallback(
    (progressPercent: number) => {
      if (!activeSession) return;
      const time = (progressPercent / 100) * playback.duration;
      storeSeek(time);
    },
    [activeSession, playback.duration, storeSeek]
  );

  const stepForward = useCallback(() => {
    if (!activeSession) return;
    const nextIdx = Math.min(
      playback.currentFrameIndex + 1,
      activeSession.frames.length - 1
    );
    if (nextIdx < activeSession.frames.length) {
      storeSeek(activeSession.frames[nextIdx].timestamp);
    }
  }, [activeSession, playback.currentFrameIndex, storeSeek]);

  const stepBackward = useCallback(() => {
    if (!activeSession) return;
    const prevIdx = Math.max(playback.currentFrameIndex - 1, 0);
    storeSeek(activeSession.frames[prevIdx].timestamp);
  }, [activeSession, playback.currentFrameIndex, storeSeek]);

  return {
    // State
    isPlaying: playback.state === 'playing',
    isPaused: playback.state === 'paused',
    playbackState: playback.state,
    currentTime: playback.currentTime,
    duration: playback.duration,
    speed: playback.speed,
    currentFrame,
    progress,
    hasSession: activeSession !== null,
    sessionName: activeSession?.name || null,

    // Actions
    play,
    pause: storePause,
    stop: storeStop,
    seek: storeSeek,
    seekToProgress,
    setSpeed: storeSetSpeed,
    stepForward,
    stepBackward,
  };
}

export default useSessionPlayback;
