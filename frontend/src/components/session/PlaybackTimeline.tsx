/**
 * PlaybackTimeline - Seekable timeline component.
 */

import React, { useRef, useCallback } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface PlaybackTimelineProps {
  currentTime: number;
  duration: number;
  progress: number;
  onSeek: (time: number) => void;
  disabled?: boolean;
}

export const PlaybackTimeline: React.FC<PlaybackTimelineProps> = ({
  currentTime,
  duration,
  progress,
  onSeek,
  disabled = false,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);
  const timelineRef = useRef<HTMLDivElement>(null);

  const formatTime = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const millis = Math.floor((ms % 1000) / 10);
    return `${minutes}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(2, '0')}`;
  };

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (disabled || !timelineRef.current) return;

      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = Math.max(0, Math.min(1, x / rect.width));
      const time = percent * duration;
      onSeek(time);
    },
    [disabled, duration, onSeek]
  );

  const handleDrag = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (disabled || e.buttons !== 1 || !timelineRef.current) return;

      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = Math.max(0, Math.min(1, x / rect.width));
      const time = percent * duration;
      onSeek(time);
    },
    [disabled, duration, onSeek]
  );

  return (
    <div className={`playback-timeline ${disabled ? 'disabled' : ''}`}>
      <span className="time-current">{formatTime(currentTime)}</span>

      <div
        ref={timelineRef}
        className="timeline-track"
        onClick={handleClick}
        onMouseMove={handleDrag}
      >
        <div className="timeline-fill" style={{ width: `${progress}%` }} />
        <div className="timeline-thumb" style={{ left: `${progress}%` }} />
      </div>

      <span className="time-total">{formatTime(duration)}</span>

      <style>{`
        .playback-timeline {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 8px 0;
        }

        .playback-timeline.disabled {
          opacity: 0.5;
          pointer-events: none;
        }

        .time-current,
        .time-total {
          font-family: monospace;
          font-size: 12px;
          color: ${theme.textMuted};
          min-width: 70px;
        }

        .time-current {
          text-align: right;
        }

        .timeline-track {
          flex: 1;
          height: 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          position: relative;
          cursor: pointer;
        }

        .timeline-fill {
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          background: ${theme.accent};
          border-radius: 4px;
          pointer-events: none;
        }

        .timeline-thumb {
          position: absolute;
          top: 50%;
          width: 16px;
          height: 16px;
          background: #fff;
          border-radius: 50%;
          transform: translate(-50%, -50%);
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
          pointer-events: none;
          transition: transform 0.1s ease;
        }

        .timeline-track:hover .timeline-thumb {
          transform: translate(-50%, -50%) scale(1.2);
        }

        .timeline-track:active .timeline-thumb {
          transform: translate(-50%, -50%) scale(0.9);
        }
      `}</style>
    </div>
  );
};

export default PlaybackTimeline;
