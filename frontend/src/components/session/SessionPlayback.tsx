/**
 * SessionPlayback - Playback UI component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { PlaybackTimeline } from './PlaybackTimeline';
import { IconButton } from '../ui/IconButton';
import type { PlaybackState } from '../../types';

interface SessionPlaybackProps {
  // State
  isPlaying: boolean;
  playbackState: PlaybackState;
  currentTime: number;
  duration: number;
  progress: number;
  speed: number;
  sessionName: string | null;
  hasSession: boolean;

  // Actions
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onSeek: (time: number) => void;
  onSetSpeed: (speed: number) => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onClose: () => void;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2, 4];

export const SessionPlayback: React.FC<SessionPlaybackProps> = ({
  isPlaying,
  playbackState: _playbackState,
  currentTime,
  duration,
  progress,
  speed,
  sessionName,
  hasSession,
  onPlay,
  onPause,
  onStop,
  onSeek,
  onSetSpeed,
  onStepForward,
  onStepBackward,
  onClose,
}) => {
  // playbackState available for future use (e.g., showing state indicator)
  void _playbackState;
  const theme = useSettingsStore((state) => state.settings.theme);

  if (!hasSession) return null;

  return (
    <div className="session-playback">
      <div className="playback-header">
        <div className="session-info">
          <span className="playback-badge">PLAYBACK</span>
          <span className="session-name">{sessionName}</span>
        </div>
        <button className="close-btn" onClick={onClose} title="Exit playback">
          ×
        </button>
      </div>

      <PlaybackTimeline
        currentTime={currentTime}
        duration={duration}
        progress={progress}
        onSeek={onSeek}
      />

      <div className="playback-controls">
        <div className="transport-controls">
          <IconButton onClick={onStop} title="Stop" size="md">
            ■
          </IconButton>
          <IconButton onClick={onStepBackward} title="Previous frame" size="md">
            ⏮
          </IconButton>
          {isPlaying ? (
            <IconButton onClick={onPause} title="Pause" size="lg" variant="primary">
              ⏸
            </IconButton>
          ) : (
            <IconButton onClick={onPlay} title="Play" size="lg" variant="primary">
              ▶
            </IconButton>
          )}
          <IconButton onClick={onStepForward} title="Next frame" size="md">
            ⏭
          </IconButton>
        </div>

        <div className="speed-controls">
          <span className="speed-label">Speed:</span>
          <div className="speed-options">
            {SPEED_OPTIONS.map((s) => (
              <button
                key={s}
                className={`speed-btn ${speed === s ? 'active' : ''}`}
                onClick={() => onSetSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        .session-playback {
          display: flex;
          flex-direction: column;
          gap: 12px;
          padding: 16px;
          background: ${theme.panelBg};
          border: 1px solid ${theme.borderColor};
          border-radius: 12px;
        }

        .playback-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .session-info {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .playback-badge {
          font-size: 10px;
          font-weight: 600;
          padding: 4px 8px;
          background: rgba(77, 166, 255, 0.2);
          color: #4da6ff;
          border-radius: 4px;
          letter-spacing: 0.5px;
        }

        .session-name {
          font-size: 14px;
          color: ${theme.text};
          font-weight: 500;
        }

        .close-btn {
          width: 28px;
          height: 28px;
          border: none;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.08);
          color: ${theme.textMuted};
          font-size: 18px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .close-btn:hover {
          background: rgba(255, 107, 107, 0.2);
          color: #ff6b6b;
        }

        .playback-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .transport-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .speed-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .speed-label {
          font-size: 12px;
          color: ${theme.textMuted};
        }

        .speed-options {
          display: flex;
          gap: 4px;
        }

        .speed-btn {
          padding: 4px 8px;
          font-size: 11px;
          font-weight: 500;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 4px;
          color: ${theme.textMuted};
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .speed-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          color: ${theme.text};
        }

        .speed-btn.active {
          background: ${theme.accent}20;
          border-color: ${theme.accent};
          color: ${theme.accent};
        }
      `}</style>
    </div>
  );
};

export default SessionPlayback;
