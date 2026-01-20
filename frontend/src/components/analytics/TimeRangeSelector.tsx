/**
 * TimeRangeSelector - Time range picker component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import type { TimeRange } from '../../types';

interface TimeRangeSelectorProps {
  duration: number;
  selectedRange: TimeRange | null;
  onRangeChange: (range: TimeRange | null) => void;
}

const PRESET_RANGES = [
  { label: 'All', getRange: null },
  { label: 'Last 30s', getRange: (d: number) => ({ start: Math.max(0, d - 30000), end: d, label: 'Last 30s' }) },
  { label: 'Last 1m', getRange: (d: number) => ({ start: Math.max(0, d - 60000), end: d, label: 'Last 1m' }) },
  { label: 'Last 5m', getRange: (d: number) => ({ start: Math.max(0, d - 300000), end: d, label: 'Last 5m' }) },
  { label: 'First half', getRange: (d: number) => ({ start: 0, end: d / 2, label: 'First half' }) },
  { label: 'Second half', getRange: (d: number) => ({ start: d / 2, end: d, label: 'Second half' }) },
];

export const TimeRangeSelector: React.FC<TimeRangeSelectorProps> = ({
  duration,
  selectedRange,
  onRangeChange,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  const isRangeSelected = (label: string): boolean => {
    if (label === 'All') return selectedRange === null;
    return selectedRange?.label === label;
  };

  return (
    <div className="time-range-selector">
      <span className="selector-label">Time Range:</span>
      <div className="range-options">
        {PRESET_RANGES.map(({ label, getRange }) => (
          <button
            key={label}
            className={`range-btn ${isRangeSelected(label) ? 'active' : ''}`}
            onClick={() => onRangeChange(getRange ? getRange(duration) : null)}
          >
            {label}
          </button>
        ))}
      </div>

      <style>{`
        .time-range-selector {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .selector-label {
          font-size: 12px;
          color: ${theme.textMuted};
        }

        .range-options {
          display: flex;
          gap: 4px;
        }

        .range-btn {
          padding: 6px 12px;
          font-size: 11px;
          font-weight: 500;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 4px;
          color: ${theme.textMuted};
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .range-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          color: ${theme.text};
        }

        .range-btn.active {
          background: ${theme.accent}20;
          border-color: ${theme.accent};
          color: ${theme.accent};
        }
      `}</style>
    </div>
  );
};

export default TimeRangeSelector;
