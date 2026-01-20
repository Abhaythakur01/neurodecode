/**
 * Slider - Styled range input component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  disabled?: boolean;
}

export const Slider: React.FC<SliderProps> = ({
  value,
  onChange,
  min,
  max,
  step = 1,
  label,
  showValue = true,
  formatValue = (v) => v.toString(),
  disabled = false,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`slider-container ${disabled ? 'disabled' : ''}`}>
      {(label || showValue) && (
        <div className="slider-header">
          {label && <span className="slider-label">{label}</span>}
          {showValue && <span className="slider-value">{formatValue(value)}</span>}
        </div>
      )}
      <div className="slider-wrapper">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          style={{
            background: `linear-gradient(to right, ${theme.accent} ${percentage}%, rgba(255,255,255,0.1) ${percentage}%)`,
          }}
        />
      </div>
      <style>{`
        .slider-container {
          width: 100%;
        }

        .slider-container.disabled {
          opacity: 0.5;
        }

        .slider-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .slider-label {
          font-size: 13px;
          color: ${theme.text};
        }

        .slider-value {
          font-size: 12px;
          font-family: monospace;
          color: ${theme.accent};
          background: rgba(0, 255, 136, 0.1);
          padding: 2px 8px;
          border-radius: 4px;
        }

        .slider-wrapper {
          position: relative;
        }

        .slider-wrapper input[type="range"] {
          -webkit-appearance: none;
          width: 100%;
          height: 6px;
          border-radius: 3px;
          outline: none;
          cursor: pointer;
        }

        .slider-wrapper input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          background: #fff;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
          transition: transform 0.15s ease;
        }

        .slider-wrapper input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.1);
        }

        .slider-wrapper input[type="range"]::-moz-range-thumb {
          width: 18px;
          height: 18px;
          background: #fff;
          border-radius: 50%;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }

        .slider-wrapper input[type="range"]:disabled {
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default Slider;
