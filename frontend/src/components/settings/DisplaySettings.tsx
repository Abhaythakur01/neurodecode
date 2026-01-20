/**
 * DisplaySettings - Display options component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { Toggle } from '../ui/Toggle';
import { Slider } from '../ui/Slider';

export const DisplaySettings: React.FC = () => {
  const { settings, setDisplaySettings, setChartSettings } = useSettingsStore();
  const { display, charts, theme } = settings;

  return (
    <div className="display-settings">
      <section className="settings-section">
        <h4>Canvas Display</h4>
        <div className="settings-group">
          <Toggle
            label="Show trajectory trail"
            checked={display.showTrajectory}
            onChange={(checked) => setDisplaySettings({ showTrajectory: checked })}
          />
          <Toggle
            label="Show uncertainty ellipse"
            checked={display.showUncertainty}
            onChange={(checked) => setDisplaySettings({ showUncertainty: checked })}
          />
          <Toggle
            label="Show grid"
            checked={display.showGrid}
            onChange={(checked) => setDisplaySettings({ showGrid: checked })}
          />
          <Toggle
            label="Show target zones"
            checked={display.showTargets}
            onChange={(checked) => setDisplaySettings({ showTargets: checked })}
          />
        </div>

        <div className="settings-group sliders">
          <Slider
            label="Trail length"
            value={display.trailLength}
            min={10}
            max={200}
            step={10}
            onChange={(value) => setDisplaySettings({ trailLength: value })}
            formatValue={(v) => `${v} points`}
          />
          <Slider
            label="Animation speed"
            value={display.animationSpeed}
            min={0.5}
            max={2}
            step={0.1}
            onChange={(value) => setDisplaySettings({ animationSpeed: value })}
            formatValue={(v) => `${v}x`}
          />
        </div>
      </section>

      <section className="settings-section">
        <h4>Charts</h4>
        <div className="settings-group">
          <Toggle
            label="Show R² chart"
            checked={charts.showR2Chart}
            onChange={(checked) => setChartSettings({ showR2Chart: checked })}
          />
          <Toggle
            label="Show latency chart"
            checked={charts.showLatencyChart}
            onChange={(checked) => setChartSettings({ showLatencyChart: checked })}
          />
        </div>

        <div className="settings-group sliders">
          <Slider
            label="Chart history points"
            value={charts.maxPoints}
            min={50}
            max={500}
            step={50}
            onChange={(value) => setChartSettings({ maxPoints: value })}
            formatValue={(v) => `${v}`}
          />
          <Slider
            label="Latency threshold"
            value={charts.latencyThreshold}
            min={10}
            max={200}
            step={10}
            onChange={(value) => setChartSettings({ latencyThreshold: value })}
            formatValue={(v) => `${v}ms`}
          />
        </div>
      </section>

      <style>{`
        .display-settings {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .settings-section h4 {
          margin: 0 0 16px 0;
          font-size: 14px;
          color: ${theme.text};
        }

        .settings-group {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .settings-group.sliders {
          margin-top: 16px;
          gap: 20px;
        }
      `}</style>
    </div>
  );
};

export default DisplaySettings;
