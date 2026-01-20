/**
 * LayoutSelector - Preset layout picker component.
 */

import React from 'react';
import { useLayoutStore } from '../../stores/layoutStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { layoutPresets } from '../../constants/layouts';
import type { LayoutPreset } from '../../types';

interface LayoutSelectorProps {
  onClose?: () => void;
}

export const LayoutSelector: React.FC<LayoutSelectorProps> = ({ onClose }) => {
  const { currentPreset, setPreset, isEditMode, toggleEditMode, resetToPreset } = useLayoutStore();
  const theme = useSettingsStore((state) => state.settings.theme);
  const setSettingsLayout = useSettingsStore((state) => state.setLayout);

  const handleSelectPreset = (preset: LayoutPreset) => {
    setPreset(preset);
    setSettingsLayout(preset);
    onClose?.();
  };

  const layoutDescriptions: Record<LayoutPreset, string> = {
    default: 'Balanced layout with cursor on left, controls on right',
    compact: 'Maximized canvas space with minimized side panel',
    analytics: 'Expanded charts and statistics for data analysis',
    recording: 'Timeline and recording controls prominently displayed',
  };

  return (
    <div className="layout-selector">
      <div className="layout-header">
        <h4>Dashboard Layout</h4>
        <div className="layout-actions">
          <button
            className={`edit-mode-btn ${isEditMode ? 'active' : ''}`}
            onClick={toggleEditMode}
          >
            {isEditMode ? 'Done Editing' : 'Edit Layout'}
          </button>
          {isEditMode && (
            <button className="reset-btn" onClick={resetToPreset}>
              Reset
            </button>
          )}
        </div>
      </div>

      <div className="layout-presets">
        {layoutPresets.map((preset) => (
          <button
            key={preset}
            className={`preset-option ${currentPreset === preset ? 'active' : ''}`}
            onClick={() => handleSelectPreset(preset)}
          >
            <div className="preset-icon">
              <LayoutPreview preset={preset} />
            </div>
            <div className="preset-info">
              <span className="preset-name">{preset}</span>
              <span className="preset-desc">{layoutDescriptions[preset]}</span>
            </div>
          </button>
        ))}
      </div>

      <style>{`
        .layout-selector {
          padding: 8px 0;
        }

        .layout-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .layout-header h4 {
          margin: 0;
          font-size: 14px;
          color: ${theme.text};
        }

        .layout-actions {
          display: flex;
          gap: 8px;
        }

        .edit-mode-btn,
        .reset-btn {
          padding: 6px 12px;
          font-size: 12px;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .edit-mode-btn {
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: ${theme.text};
        }

        .edit-mode-btn:hover {
          background: rgba(255, 255, 255, 0.12);
        }

        .edit-mode-btn.active {
          background: ${theme.accent};
          border-color: ${theme.accent};
          color: #000;
        }

        .reset-btn {
          background: transparent;
          border: 1px solid rgba(255, 107, 107, 0.3);
          color: #ff6b6b;
        }

        .reset-btn:hover {
          background: rgba(255, 107, 107, 0.1);
        }

        .layout-presets {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .preset-option {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s ease;
          text-align: left;
        }

        .preset-option:hover {
          background: rgba(255, 255, 255, 0.06);
          border-color: rgba(255, 255, 255, 0.12);
        }

        .preset-option.active {
          border-color: ${theme.accent};
          background: rgba(0, 255, 136, 0.05);
        }

        .preset-icon {
          width: 64px;
          height: 48px;
          flex-shrink: 0;
        }

        .preset-info {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .preset-name {
          font-size: 13px;
          font-weight: 600;
          color: ${theme.text};
          text-transform: capitalize;
        }

        .preset-desc {
          font-size: 11px;
          color: ${theme.textMuted};
          line-height: 1.4;
        }
      `}</style>
    </div>
  );
};

// Mini layout preview component
const LayoutPreview: React.FC<{ preset: LayoutPreset }> = ({ preset }) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  const previewStyles: Record<LayoutPreset, React.CSSProperties[]> = {
    default: [
      { left: 0, top: 0, width: '50%', height: '100%' },
      { left: '52%', top: 0, width: '48%', height: '33%' },
      { left: '52%', top: '35%', width: '48%', height: '40%' },
      { left: '52%', top: '77%', width: '48%', height: '23%' },
    ],
    compact: [
      { left: 0, top: 0, width: '66%', height: '100%' },
      { left: '68%', top: 0, width: '32%', height: '25%' },
      { left: '68%', top: '27%', width: '32%', height: '40%' },
      { left: '68%', top: '69%', width: '32%', height: '31%' },
    ],
    analytics: [
      { left: 0, top: 0, width: '40%', height: '65%' },
      { left: 0, top: '67%', width: '40%', height: '33%' },
      { left: '42%', top: 0, width: '58%', height: '50%' },
      { left: '42%', top: '52%', width: '30%', height: '48%' },
      { left: '74%', top: '52%', width: '26%', height: '48%' },
    ],
    recording: [
      { left: 0, top: 0, width: '50%', height: '80%' },
      { left: '52%', top: 0, width: '48%', height: '25%' },
      { left: '52%', top: '27%', width: '48%', height: '25%' },
      { left: '52%', top: '54%', width: '48%', height: '26%' },
      { left: 0, top: '82%', width: '100%', height: '18%' },
    ],
  };

  return (
    <svg viewBox="0 0 64 48" fill="none" style={{ width: '100%', height: '100%' }}>
      <rect
        x="0"
        y="0"
        width="64"
        height="48"
        fill="rgba(255,255,255,0.05)"
        rx="4"
      />
      {previewStyles[preset].map((style, i) => (
        <rect
          key={i}
          x={parseFloat(String(style.left)) * 0.64}
          y={parseFloat(String(style.top)) * 0.48}
          width={parseFloat(String(style.width)) * 0.64}
          height={parseFloat(String(style.height)) * 0.48}
          fill={i === 0 ? theme.accent + '40' : 'rgba(255,255,255,0.1)'}
          stroke={theme.borderColor}
          strokeWidth="0.5"
          rx="2"
        />
      ))}
    </svg>
  );
};

export default LayoutSelector;
