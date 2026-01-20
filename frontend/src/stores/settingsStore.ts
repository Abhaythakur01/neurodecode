/**
 * Settings store using Zustand for state management.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { UserSettings, ThemeSettings, DisplaySettings, ChartSettings, LayoutPreset } from '../types';
import { defaultUserSettings } from '../constants/defaults';
import { themes, ThemeName } from '../constants/themes';

interface SettingsState {
  settings: UserSettings;
  setTheme: (themeName: ThemeName) => void;
  setCustomTheme: (theme: ThemeSettings) => void;
  setDisplaySettings: (settings: Partial<DisplaySettings>) => void;
  setChartSettings: (settings: Partial<ChartSettings>) => void;
  setLayout: (layout: LayoutPreset) => void;
  resetSettings: () => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      settings: defaultUserSettings,

      setTheme: (themeName: ThemeName) =>
        set((state) => ({
          settings: {
            ...state.settings,
            theme: themes[themeName] || themes.default,
          },
        })),

      setCustomTheme: (theme: ThemeSettings) =>
        set((state) => ({
          settings: {
            ...state.settings,
            theme,
          },
        })),

      setDisplaySettings: (displaySettings: Partial<DisplaySettings>) =>
        set((state) => ({
          settings: {
            ...state.settings,
            display: {
              ...state.settings.display,
              ...displaySettings,
            },
          },
        })),

      setChartSettings: (chartSettings: Partial<ChartSettings>) =>
        set((state) => ({
          settings: {
            ...state.settings,
            charts: {
              ...state.settings.charts,
              ...chartSettings,
            },
          },
        })),

      setLayout: (layout: LayoutPreset) =>
        set((state) => ({
          settings: {
            ...state.settings,
            layout,
          },
        })),

      resetSettings: () =>
        set(() => ({
          settings: defaultUserSettings,
        })),
    }),
    {
      name: 'neurodecode-settings',
    }
  )
);
