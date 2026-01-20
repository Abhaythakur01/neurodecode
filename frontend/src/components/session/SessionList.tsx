/**
 * SessionList - Saved sessions list component.
 */

import React, { useState } from 'react';
import { useSessionStore } from '../../stores/sessionStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { Modal } from '../ui/Modal';
import { ExportDialog } from './ExportDialog';
import type { SessionRecording } from '../../types';

interface SessionListProps {
  isOpen: boolean;
  onClose: () => void;
  onLoadSession: (sessionId: string) => void;
}

export const SessionList: React.FC<SessionListProps> = ({
  isOpen,
  onClose,
  onLoadSession,
}) => {
  const { sessions, deleteSession, renameSession, clearAllSessions } = useSessionStore();
  const theme = useSettingsStore((state) => state.settings.theme);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [exportSession, setExportSession] = useState<SessionRecording | null>(null);

  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (timestamp: number): string => {
    return new Date(timestamp).toLocaleString();
  };

  const handleStartEdit = (session: SessionRecording) => {
    setEditingId(session.id);
    setEditName(session.name);
  };

  const handleSaveEdit = (id: string) => {
    if (editName.trim()) {
      renameSession(id, editName.trim());
    }
    setEditingId(null);
  };

  const handleDelete = (id: string) => {
    if (window.confirm('Delete this recording?')) {
      deleteSession(id);
    }
  };

  const handleClearAll = () => {
    if (window.confirm('Delete all recordings? This cannot be undone.')) {
      clearAllSessions();
    }
  };

  const handleLoad = (id: string) => {
    onLoadSession(id);
    onClose();
  };

  return (
    <>
      <Modal isOpen={isOpen} onClose={onClose} title="Saved Recordings" width="600px">
        <div className="session-list">
          {sessions.length === 0 ? (
            <div className="empty-state">
              <p>No recordings yet</p>
              <p className="hint">Start a recording to save session data</p>
            </div>
          ) : (
            <>
              <div className="list-header">
                <span>{sessions.length} recording{sessions.length !== 1 ? 's' : ''}</span>
                <button className="clear-all-btn" onClick={handleClearAll}>
                  Clear All
                </button>
              </div>

              <div className="sessions">
                {sessions.map((session) => (
                  <div key={session.id} className="session-item">
                    <div className="session-info">
                      {editingId === session.id ? (
                        <input
                          className="edit-input"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onBlur={() => handleSaveEdit(session.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveEdit(session.id);
                            if (e.key === 'Escape') setEditingId(null);
                          }}
                          autoFocus
                        />
                      ) : (
                        <span
                          className="session-name"
                          onClick={() => handleStartEdit(session)}
                        >
                          {session.name}
                        </span>
                      )}
                      <div className="session-meta">
                        <span>{formatDate(session.createdAt)}</span>
                        <span className="dot">·</span>
                        <span>{formatDuration(session.duration)}</span>
                        <span className="dot">·</span>
                        <span>{session.frames.length} frames</span>
                      </div>
                      <div className="session-stats">
                        <span className="stat">
                          R²: <strong>{session.metadata.averageR2.toFixed(3)}</strong>
                        </span>
                        <span className="stat">
                          Latency: <strong>{session.metadata.averageLatency.toFixed(1)}ms</strong>
                        </span>
                      </div>
                    </div>

                    <div className="session-actions">
                      <button
                        className="action-btn play"
                        onClick={() => handleLoad(session.id)}
                        title="Load session"
                      >
                        ▶
                      </button>
                      <button
                        className="action-btn export"
                        onClick={() => setExportSession(session)}
                        title="Export"
                      >
                        ↓
                      </button>
                      <button
                        className="action-btn delete"
                        onClick={() => handleDelete(session.id)}
                        title="Delete"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <style>{`
          .session-list {
            min-height: 200px;
          }

          .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: ${theme.textMuted};
          }

          .empty-state p {
            margin: 0;
          }

          .empty-state .hint {
            font-size: 12px;
            margin-top: 8px;
            opacity: 0.7;
          }

          .list-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            font-size: 13px;
            color: ${theme.textMuted};
          }

          .clear-all-btn {
            background: transparent;
            border: none;
            color: #ff6b6b;
            font-size: 12px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
          }

          .clear-all-btn:hover {
            background: rgba(255, 107, 107, 0.1);
          }

          .sessions {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 400px;
            overflow-y: auto;
          }

          .session-item {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 12px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
          }

          .session-info {
            flex: 1;
          }

          .session-name {
            font-weight: 500;
            color: ${theme.text};
            cursor: pointer;
          }

          .session-name:hover {
            color: ${theme.accent};
          }

          .edit-input {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid ${theme.accent};
            border-radius: 4px;
            padding: 4px 8px;
            color: ${theme.text};
            font-size: 14px;
            width: 200px;
          }

          .session-meta {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 6px;
            font-size: 12px;
            color: ${theme.textMuted};
          }

          .dot {
            opacity: 0.5;
          }

          .session-stats {
            display: flex;
            gap: 16px;
            margin-top: 8px;
            font-size: 12px;
            color: ${theme.textMuted};
          }

          .session-stats strong {
            color: ${theme.accent};
          }

          .session-actions {
            display: flex;
            gap: 8px;
          }

          .action-btn {
            width: 32px;
            height: 32px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
          }

          .action-btn.play {
            background: rgba(0, 255, 136, 0.15);
            color: ${theme.accent};
          }

          .action-btn.play:hover {
            background: rgba(0, 255, 136, 0.25);
          }

          .action-btn.export {
            background: rgba(77, 166, 255, 0.15);
            color: #4da6ff;
          }

          .action-btn.export:hover {
            background: rgba(77, 166, 255, 0.25);
          }

          .action-btn.delete {
            background: rgba(255, 107, 107, 0.15);
            color: #ff6b6b;
          }

          .action-btn.delete:hover {
            background: rgba(255, 107, 107, 0.25);
          }
        `}</style>
      </Modal>

      {exportSession && (
        <ExportDialog
          session={exportSession}
          onClose={() => setExportSession(null)}
        />
      )}
    </>
  );
};

export default SessionList;
