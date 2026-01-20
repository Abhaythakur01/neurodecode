/**
 * SessionRecorder - Recording controls component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface SessionRecorderProps {
  isRecording: boolean;
  duration: number;
  frameCount: number;
  onStart: () => void;
  onStop: () => void;
  onCancel: () => void;
  disabled?: boolean;
}

export const SessionRecorder: React.FC<SessionRecorderProps> = ({
  isRecording,
  duration,
  frameCount,
  onStart,
  onStop,
  onCancel,
  disabled = false,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="session-recorder">
      <div className="recorder-controls">
        {!isRecording ? (
          <button
            className="record-btn"
            onClick={onStart}
            disabled={disabled}
            title="Start recording"
          >
            <span className="record-icon" />
            Record
          </button>
        ) : (
          <>
            <button
              className="stop-btn"
              onClick={onStop}
              title="Stop and save recording"
            >
              <span className="stop-icon" />
              Stop
            </button>
            <button
              className="cancel-btn"
              onClick={onCancel}
              title="Cancel recording"
            >
              Cancel
            </button>
          </>
        )}
      </div>

      {isRecording && (
        <div className="recording-status">
          <span className="recording-indicator">
            <span className="pulse" />
            REC
          </span>
          <span className="recording-time">{formatDuration(duration)}</span>
          <span className="frame-count">{frameCount} frames</span>
        </div>
      )}

      <style>{`
        .session-recorder {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .recorder-controls {
          display: flex;
          gap: 8px;
        }

        .record-btn,
        .stop-btn,
        .cancel-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          border: none;
          border-radius: 6px;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .record-btn {
          background: rgba(255, 107, 107, 0.15);
          color: #ff6b6b;
          border: 1px solid rgba(255, 107, 107, 0.3);
        }

        .record-btn:hover:not(:disabled) {
          background: rgba(255, 107, 107, 0.25);
        }

        .record-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .record-icon {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: #ff6b6b;
        }

        .stop-btn {
          background: #ff6b6b;
          color: #fff;
        }

        .stop-btn:hover {
          background: #ff5252;
        }

        .stop-icon {
          width: 10px;
          height: 10px;
          background: #fff;
          border-radius: 2px;
        }

        .cancel-btn {
          background: rgba(255, 255, 255, 0.08);
          color: ${theme.textMuted};
        }

        .cancel-btn:hover {
          background: rgba(255, 255, 255, 0.12);
          color: ${theme.text};
        }

        .recording-status {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .recording-indicator {
          display: flex;
          align-items: center;
          gap: 6px;
          color: #ff6b6b;
          font-weight: 600;
          font-size: 12px;
        }

        .pulse {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #ff6b6b;
          animation: pulse-animation 1s ease-in-out infinite;
        }

        @keyframes pulse-animation {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.8); }
        }

        .recording-time {
          font-family: monospace;
          font-size: 16px;
          color: ${theme.text};
        }

        .frame-count {
          font-size: 12px;
          color: ${theme.textMuted};
        }
      `}</style>
    </div>
  );
};

export default SessionRecorder;
