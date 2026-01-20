/**
 * Tabs - Tab navigation component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, onTabChange }) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  return (
    <div className="tabs-container">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.icon && <span className="tab-icon">{tab.icon}</span>}
          <span className="tab-label">{tab.label}</span>
        </button>
      ))}
      <style>{`
        .tabs-container {
          display: flex;
          gap: 4px;
          background: rgba(0, 0, 0, 0.2);
          padding: 4px;
          border-radius: 8px;
        }

        .tab-button {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 8px 16px;
          background: transparent;
          border: none;
          border-radius: 6px;
          font-size: 13px;
          font-weight: 500;
          color: ${theme.textMuted};
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .tab-button:hover {
          color: ${theme.text};
          background: rgba(255, 255, 255, 0.05);
        }

        .tab-button.active {
          background: rgba(255, 255, 255, 0.1);
          color: ${theme.accent};
        }

        .tab-icon {
          display: flex;
          align-items: center;
          font-size: 16px;
        }

        .tab-label {
          white-space: nowrap;
        }
      `}</style>
    </div>
  );
};

export default Tabs;
