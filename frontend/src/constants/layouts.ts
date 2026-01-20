/**
 * Predefined layout configurations for the dashboard.
 */

import type { DashboardLayout, LayoutPreset } from '../types';

export const defaultLayout: DashboardLayout = {
  id: 'default',
  name: 'Default',
  items: [
    { i: 'cursor', x: 0, y: 0, w: 6, h: 12, minW: 4, minH: 8 },
    { i: 'controls', x: 6, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
    { i: 'performance', x: 6, y: 4, w: 6, h: 5, minW: 4, minH: 4 },
    { i: 'decoders', x: 6, y: 9, w: 6, h: 3, minW: 3, minH: 3 },
  ],
};

export const compactLayout: DashboardLayout = {
  id: 'compact',
  name: 'Compact',
  items: [
    { i: 'cursor', x: 0, y: 0, w: 8, h: 12, minW: 4, minH: 8 },
    { i: 'controls', x: 8, y: 0, w: 4, h: 3, minW: 3, minH: 2 },
    { i: 'performance', x: 8, y: 3, w: 4, h: 5, minW: 3, minH: 4 },
    { i: 'decoders', x: 8, y: 8, w: 4, h: 4, minW: 3, minH: 3 },
  ],
};

export const analyticsLayout: DashboardLayout = {
  id: 'analytics',
  name: 'Analytics',
  items: [
    { i: 'cursor', x: 0, y: 0, w: 5, h: 8, minW: 4, minH: 6 },
    { i: 'controls', x: 0, y: 8, w: 5, h: 4, minW: 4, minH: 3 },
    { i: 'performance', x: 5, y: 0, w: 7, h: 6, minW: 5, minH: 5 },
    { i: 'decoders', x: 5, y: 6, w: 3, h: 6, minW: 3, minH: 4 },
    { i: 'statistics', x: 8, y: 6, w: 4, h: 6, minW: 3, minH: 4 },
  ],
};

export const recordingLayout: DashboardLayout = {
  id: 'recording',
  name: 'Recording',
  items: [
    { i: 'cursor', x: 0, y: 0, w: 6, h: 10, minW: 4, minH: 8 },
    { i: 'controls', x: 6, y: 0, w: 6, h: 3, minW: 4, minH: 2 },
    { i: 'recording', x: 6, y: 3, w: 6, h: 3, minW: 4, minH: 2 },
    { i: 'performance', x: 6, y: 6, w: 6, h: 4, minW: 4, minH: 3 },
    { i: 'timeline', x: 0, y: 10, w: 12, h: 2, minW: 8, minH: 2 },
  ],
};

export const layouts: Record<LayoutPreset, DashboardLayout> = {
  default: defaultLayout,
  compact: compactLayout,
  analytics: analyticsLayout,
  recording: recordingLayout,
};

export const layoutPresets: LayoutPreset[] = ['default', 'compact', 'analytics', 'recording'];
