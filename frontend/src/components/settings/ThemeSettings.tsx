/**
 * ThemeSettings - Theme customization component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { themes, themeNames, ThemeName } from '../../constants/themes';

export const ThemeSettings: React.FC = () => {
  const { settings, setTheme } = useSettingsStore();
  const currentTheme = settings.theme;

  // Find current theme name by comparing values
  const getCurrentThemeName = (): ThemeName => {
    for (const name of themeNames) {
      if (themes[name].accent === currentTheme.accent) {
        return name;
      }
    }
    return 'default';
  };

  return (
    <div className="theme-settings">
      <h4>Theme</h4>
      <div className="theme-grid">
        {themeNames.map((themeName) => {
          const theme = themes[themeName];
          const isActive = getCurrentThemeName() === themeName;

          return (
            <button
              key={themeName}
              className={`theme-option ${isActive ? 'active' : ''}`}
              onClick={() => setTheme(themeName)}
            >
              <div
                className="theme-preview"
                style={{
                  background: theme.panelBg,
                  borderColor: theme.borderColor,
                }}
              >
                <div
                  className="theme-accent"
                  style={{ background: theme.accent }}
                />
              </div>
              <span className="theme-name">{themeName}</span>
            </button>
          );
        })}
      </div>

      <div className="theme-preview-section">
        <h5>Preview</h5>
        <div
          className="preview-panel"
          style={{
            background: currentTheme.panelBg,
            border: `1px solid ${currentTheme.borderColor}`,
          }}
        >
          <div className="preview-header" style={{ color: currentTheme.text }}>
            Panel Header
          </div>
          <div className="preview-content" style={{ color: currentTheme.textMuted }}>
            Muted text content
          </div>
          <div className="preview-accent" style={{ color: currentTheme.accent }}>
            Accent color
          </div>
        </div>
      </div>

      <style>{`
        .theme-settings h4 {
          margin: 0 0 16px 0;
          font-size: 14px;
          color: ${currentTheme.text};
        }

        .theme-settings h5 {
          margin: 20px 0 12px 0;
          font-size: 12px;
          color: ${currentTheme.textMuted};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .theme-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }

        .theme-option {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .theme-option:hover {
          background: rgba(255, 255, 255, 0.06);
        }

        .theme-option.active {
          border-color: ${currentTheme.accent};
          background: rgba(0, 255, 136, 0.05);
        }

        .theme-preview {
          width: 100%;
          height: 48px;
          border-radius: 6px;
          border: 1px solid;
          display: flex;
          align-items: flex-end;
          justify-content: center;
          padding: 6px;
        }

        .theme-accent {
          width: 60%;
          height: 4px;
          border-radius: 2px;
        }

        .theme-name {
          font-size: 12px;
          color: ${currentTheme.text};
          text-transform: capitalize;
        }

        .preview-panel {
          padding: 12px;
          border-radius: 8px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .preview-header {
          font-weight: 600;
          font-size: 13px;
        }

        .preview-content {
          font-size: 12px;
        }

        .preview-accent {
          font-size: 12px;
          font-weight: 500;
        }
      `}</style>
    </div>
  );
};

export default ThemeSettings;
