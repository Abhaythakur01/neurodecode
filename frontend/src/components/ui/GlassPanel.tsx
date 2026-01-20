/**
 * GlassPanel - Reusable glassmorphism container component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface GlassPanelProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  noPadding?: boolean;
  header?: React.ReactNode;
}

export const GlassPanel: React.FC<GlassPanelProps> = ({
  children,
  className = '',
  style = {},
  noPadding = false,
  header,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  return (
    <div
      className={`glass-panel ${className}`}
      style={{
        background: theme.panelBg,
        backdropFilter: `blur(${theme.blur})`,
        WebkitBackdropFilter: `blur(${theme.blur})`,
        border: `1px solid ${theme.borderColor}`,
        ...style,
      }}
    >
      {header && <div className="glass-panel-header">{header}</div>}
      <div className={`glass-panel-content ${noPadding ? 'no-padding' : ''}`}>
        {children}
      </div>
      <style>{`
        .glass-panel {
          border-radius: 12px;
          overflow: hidden;
          transition: all 0.3s ease;
        }

        .glass-panel:hover {
          border-color: rgba(255, 255, 255, 0.12);
        }

        .glass-panel-header {
          padding: 12px 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          font-size: 13px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: ${theme.textMuted};
        }

        .glass-panel-content {
          padding: 16px;
        }

        .glass-panel-content.no-padding {
          padding: 0;
        }
      `}</style>
    </div>
  );
};

export default GlassPanel;
