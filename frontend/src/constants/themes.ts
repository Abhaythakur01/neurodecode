/**
 * Theme tokens and presets for glassmorphism design.
 */

import type { ThemeSettings } from '../types';

export const defaultTheme: ThemeSettings = {
  panelBg: 'rgba(20, 20, 30, 0.6)',
  blur: '12px',
  borderColor: 'rgba(255, 255, 255, 0.08)',
  accent: '#00ff88',
  text: '#e0e0e0',
  textMuted: '#888888',
};

export const darkTheme: ThemeSettings = {
  panelBg: 'rgba(10, 10, 15, 0.8)',
  blur: '16px',
  borderColor: 'rgba(255, 255, 255, 0.05)',
  accent: '#00ff88',
  text: '#ffffff',
  textMuted: '#666666',
};

export const blueTheme: ThemeSettings = {
  panelBg: 'rgba(15, 25, 45, 0.7)',
  blur: '12px',
  borderColor: 'rgba(100, 150, 255, 0.15)',
  accent: '#4da6ff',
  text: '#e0e8f0',
  textMuted: '#8090a0',
};

export const purpleTheme: ThemeSettings = {
  panelBg: 'rgba(30, 20, 45, 0.7)',
  blur: '12px',
  borderColor: 'rgba(150, 100, 255, 0.15)',
  accent: '#a855f7',
  text: '#e8e0f0',
  textMuted: '#9080a0',
};

export const themes: Record<string, ThemeSettings> = {
  default: defaultTheme,
  dark: darkTheme,
  blue: blueTheme,
  purple: purpleTheme,
};

export const themeNames = ['default', 'dark', 'blue', 'purple'] as const;
export type ThemeName = (typeof themeNames)[number];
