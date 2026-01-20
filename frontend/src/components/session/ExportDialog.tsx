/**
 * ExportDialog - Export options dialog component.
 */

import React, { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Toggle } from '../ui/Toggle';
import { useSettingsStore } from '../../stores/settingsStore';
import { exportSession } from '../../utils/export';
import type { SessionRecording, ExportFormat, ExportOptions } from '../../types';

interface ExportDialogProps {
  session: SessionRecording;
  onClose: () => void;
}

export const ExportDialog: React.FC<ExportDialogProps> = ({ session, onClose }) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  const [format, setFormat] = useState<ExportFormat>('json');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeAllFrames, setIncludeAllFrames] = useState(true);

  const handleExport = () => {
    const options: ExportOptions = {
      format,
      includeMetadata,
      includeAllFrames,
    };

    exportSession(session, options);
    onClose();
  };

  const estimateFileSize = (): string => {
    const baseSize = includeMetadata ? 500 : 100;
    const frameSize = includeAllFrames ? session.frames.length * (format === 'json' ? 150 : 80) : 0;
    const total = baseSize + frameSize;

    if (total < 1024) return `~${total} B`;
    if (total < 1024 * 1024) return `~${(total / 1024).toFixed(1)} KB`;
    return `~${(total / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Modal isOpen={true} onClose={onClose} title="Export Recording" width="420px">
      <div className="export-dialog">
        <div className="session-preview">
          <h4>{session.name}</h4>
          <p>
            {session.frames.length} frames · {(session.duration / 1000).toFixed(1)}s
          </p>
        </div>

        <div className="format-selector">
          <h5>Format</h5>
          <div className="format-options">
            <button
              className={`format-option ${format === 'json' ? 'active' : ''}`}
              onClick={() => setFormat('json')}
            >
              <span className="format-icon">{ }</span>
              <span className="format-name">JSON</span>
              <span className="format-desc">Full data with structure</span>
            </button>
            <button
              className={`format-option ${format === 'csv' ? 'active' : ''}`}
              onClick={() => setFormat('csv')}
            >
              <span className="format-icon">▤</span>
              <span className="format-name">CSV</span>
              <span className="format-desc">Spreadsheet compatible</span>
            </button>
          </div>
        </div>

        <div className="export-options">
          <h5>Options</h5>
          <Toggle
            label="Include metadata"
            checked={includeMetadata}
            onChange={setIncludeMetadata}
          />
          <Toggle
            label="Include all frame data"
            checked={includeAllFrames}
            onChange={setIncludeAllFrames}
          />
        </div>

        <div className="export-info">
          <span>Estimated file size: {estimateFileSize()}</span>
        </div>

        <div className="export-actions">
          <button className="cancel-btn" onClick={onClose}>
            Cancel
          </button>
          <button className="export-btn" onClick={handleExport}>
            Export
          </button>
        </div>
      </div>

      <style>{`
        .export-dialog {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .session-preview {
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 8px;
        }

        .session-preview h4 {
          margin: 0 0 4px 0;
          font-size: 14px;
          color: ${theme.text};
        }

        .session-preview p {
          margin: 0;
          font-size: 12px;
          color: ${theme.textMuted};
        }

        .format-selector h5,
        .export-options h5 {
          margin: 0 0 12px 0;
          font-size: 12px;
          color: ${theme.textMuted};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .format-options {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }

        .format-option {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          padding: 16px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .format-option:hover {
          background: rgba(255, 255, 255, 0.06);
        }

        .format-option.active {
          border-color: ${theme.accent};
          background: rgba(0, 255, 136, 0.05);
        }

        .format-icon {
          font-size: 24px;
          opacity: 0.8;
        }

        .format-name {
          font-weight: 600;
          color: ${theme.text};
        }

        .format-desc {
          font-size: 11px;
          color: ${theme.textMuted};
        }

        .export-options {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .export-info {
          font-size: 12px;
          color: ${theme.textMuted};
          text-align: center;
          padding: 8px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 6px;
        }

        .export-actions {
          display: flex;
          gap: 12px;
          justify-content: flex-end;
        }

        .cancel-btn,
        .export-btn {
          padding: 10px 24px;
          border: none;
          border-radius: 8px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .cancel-btn {
          background: rgba(255, 255, 255, 0.08);
          color: ${theme.text};
        }

        .cancel-btn:hover {
          background: rgba(255, 255, 255, 0.12);
        }

        .export-btn {
          background: ${theme.accent};
          color: #000;
        }

        .export-btn:hover {
          opacity: 0.9;
        }
      `}</style>
    </Modal>
  );
};

export default ExportDialog;
