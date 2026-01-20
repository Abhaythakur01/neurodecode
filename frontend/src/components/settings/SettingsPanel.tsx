/**
 * SettingsPanel - Main settings modal component.
 */

import React, { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Tabs } from '../ui/Tabs';
import { ThemeSettings } from './ThemeSettings';
import { DisplaySettings } from './DisplaySettings';
import { useSettingsStore } from '../../stores/settingsStore';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const tabs = [
  { id: 'theme', label: 'Theme' },
  { id: 'display', label: 'Display' },
  { id: 'about', label: 'About' },
];

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState('theme');
  const { resetSettings, settings } = useSettingsStore();
  const theme = settings.theme;

  const handleReset = () => {
    if (window.confirm('Reset all settings to defaults?')) {
      resetSettings();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Settings" width="520px">
      <Tabs tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="settings-content">
        {activeTab === 'theme' && <ThemeSettings />}
        {activeTab === 'display' && <DisplaySettings />}
        {activeTab === 'about' && (
          <div className="about-section">
            <h4>NeuroDecode BCI</h4>
            <p>Real-time Neural Decoding Interface</p>
            <p className="version">Version 1.0.0</p>

            <div className="about-info">
              <h5>Keyboard Shortcuts</h5>
              <ul>
                <li><kbd>Space</kbd> - Start/Stop simulation</li>
                <li><kbd>C</kbd> - Calibrate decoders</li>
                <li><kbd>R</kbd> - Reconnect WebSocket</li>
                <li><kbd>Esc</kbd> - Close modals</li>
              </ul>
            </div>

            <button className="reset-button" onClick={handleReset}>
              Reset All Settings
            </button>
          </div>
        )}
      </div>

      <style>{`
        .settings-content {
          margin-top: 20px;
          min-height: 300px;
        }

        .about-section h4 {
          margin: 0 0 8px 0;
          font-size: 18px;
          color: ${theme.text};
        }

        .about-section p {
          margin: 0 0 4px 0;
          color: ${theme.textMuted};
          font-size: 13px;
        }

        .about-section .version {
          color: ${theme.accent};
          font-family: monospace;
        }

        .about-info {
          margin-top: 24px;
        }

        .about-info h5 {
          margin: 0 0 12px 0;
          font-size: 13px;
          color: ${theme.text};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .about-info ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .about-info li {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 0;
          font-size: 13px;
          color: ${theme.textMuted};
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .about-info kbd {
          background: rgba(255, 255, 255, 0.1);
          padding: 4px 8px;
          border-radius: 4px;
          font-family: monospace;
          font-size: 12px;
          color: ${theme.text};
          min-width: 50px;
          text-align: center;
        }

        .reset-button {
          margin-top: 24px;
          padding: 10px 20px;
          background: rgba(255, 107, 107, 0.15);
          border: 1px solid rgba(255, 107, 107, 0.3);
          border-radius: 8px;
          color: #ff6b6b;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .reset-button:hover {
          background: rgba(255, 107, 107, 0.25);
        }
      `}</style>
    </Modal>
  );
};

export default SettingsPanel;
