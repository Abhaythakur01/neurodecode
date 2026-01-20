/**
 * Layout store using Zustand for dashboard layout management.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { LayoutItem, LayoutPreset, DashboardLayout } from '../types';
import { layouts, defaultLayout } from '../constants/layouts';

interface LayoutState {
  currentPreset: LayoutPreset;
  currentLayout: LayoutItem[];
  customLayouts: Record<string, DashboardLayout>;
  isEditMode: boolean;
  setPreset: (preset: LayoutPreset) => void;
  setLayout: (items: LayoutItem[]) => void;
  saveCustomLayout: (name: string) => void;
  deleteCustomLayout: (id: string) => void;
  resetToPreset: () => void;
  toggleEditMode: () => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set, get) => ({
      currentPreset: 'default',
      currentLayout: defaultLayout.items,
      customLayouts: {},
      isEditMode: false,

      setPreset: (preset: LayoutPreset) => {
        const layout = layouts[preset];
        if (layout) {
          set({
            currentPreset: preset,
            currentLayout: layout.items,
          });
        }
      },

      setLayout: (items: LayoutItem[]) => {
        set({ currentLayout: items });
      },

      saveCustomLayout: (name: string) => {
        const { currentLayout, customLayouts } = get();
        const id = `custom-${Date.now()}`;
        const newLayout: DashboardLayout = {
          id,
          name,
          items: currentLayout,
        };
        set({
          customLayouts: {
            ...customLayouts,
            [id]: newLayout,
          },
        });
      },

      deleteCustomLayout: (id: string) => {
        const { customLayouts } = get();
        const { [id]: _, ...rest } = customLayouts;
        set({ customLayouts: rest });
      },

      resetToPreset: () => {
        const { currentPreset } = get();
        const layout = layouts[currentPreset];
        if (layout) {
          set({ currentLayout: layout.items });
        }
      },

      toggleEditMode: () => {
        set((state) => ({ isEditMode: !state.isEditMode }));
      },
    }),
    {
      name: 'neurodecode-layouts',
    }
  )
);
