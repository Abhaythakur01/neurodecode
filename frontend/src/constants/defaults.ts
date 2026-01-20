/**
 * Default settings for the NeuroDecode application.
 */

import type { DisplaySettings, ChartSettings, UserSettings, LayoutPreset } from '../types';
import { defaultTheme } from './themes';

export const defaultDisplaySettings: DisplaySettings = {
  showTrajectory: true,
  showUncertainty: true,
  showGrid: true,
  showTargets: true,
  trailLength: 100,
  animationSpeed: 1.0,
};

export const defaultChartSettings: ChartSettings = {
  maxPoints: 100,
  latencyThreshold: 50,
  showR2Chart: true,
  showLatencyChart: true,
};

export const defaultLayoutPreset: LayoutPreset = 'default';

export const defaultUserSettings: UserSettings = {
  theme: defaultTheme,
  display: defaultDisplaySettings,
  charts: defaultChartSettings,
  layout: defaultLayoutPreset,
};

export const STORAGE_KEYS = {
  SETTINGS: 'neurodecode-settings',
  LAYOUTS: 'neurodecode-layouts',
  SESSIONS: 'neurodecode-sessions',
} as const;

export const MAX_STORED_SESSIONS = 50;
export const MAX_FRAMES_PER_SESSION = 36000; // ~10 minutes at 60fps
