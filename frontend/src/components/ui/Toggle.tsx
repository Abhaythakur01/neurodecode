/**
 * Toggle - Modern toggle switch component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}

export const Toggle: React.FC<ToggleProps> = ({
  checked,
  onChange,
  label,
  disabled = false,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  return (
    <label className={`toggle-container ${disabled ? 'disabled' : ''}`}>
      {label && <span className="toggle-label">{label}</span>}
      <div className="toggle-wrapper">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
        />
        <span className="toggle-track">
          <span className="toggle-thumb" />
        </span>
      </div>
      <style>{`
        .toggle-container {
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
          gap: 12px;
        }

        .toggle-container.disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .toggle-label {
          font-size: 13px;
          color: ${theme.text};
        }

        .toggle-wrapper {
          position: relative;
        }

        .toggle-wrapper input {
          position: absolute;
          opacity: 0;
          width: 0;
          height: 0;
        }

        .toggle-track {
          display: block;
          width: 44px;
          height: 24px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          position: relative;
          transition: background 0.2s ease;
        }

        .toggle-wrapper input:checked + .toggle-track {
          background: ${theme.accent};
        }

        .toggle-thumb {
          position: absolute;
          top: 2px;
          left: 2px;
          width: 20px;
          height: 20px;
          background: #fff;
          border-radius: 50%;
          transition: transform 0.2s ease;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .toggle-wrapper input:checked + .toggle-track .toggle-thumb {
          transform: translateX(20px);
        }

        .toggle-wrapper input:focus + .toggle-track {
          box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.3);
        }
      `}</style>
    </label>
  );
};

export default Toggle;
