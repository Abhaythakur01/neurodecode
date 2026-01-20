/**
 * DashboardLayout - Main react-grid-layout container.
 */

import React, { useMemo } from 'react';
import GridLayout from 'react-grid-layout';
import { useLayoutStore } from '../../stores/layoutStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { DraggablePanel } from './DraggablePanel';
import type { LayoutItem } from '../../types';

import 'react-grid-layout/css/styles.css';

interface DashboardLayoutProps {
  width: number;
  children: Record<string, React.ReactNode>;
  panels: { id: string; title?: string; noPadding?: boolean }[];
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  width,
  children,
  panels,
}) => {
  const { currentLayout, setLayout, isEditMode } = useLayoutStore();
  const theme = useSettingsStore((state) => state.settings.theme);

  const rowHeight = 50;
  const cols = 12;
  const margin: [number, number] = [16, 16];

  const layout = useMemo(() => {
    return currentLayout.map((item) => ({
      ...item,
      isDraggable: isEditMode,
      isResizable: isEditMode,
    }));
  }, [currentLayout, isEditMode]);

  const handleLayoutChange = (newLayout: GridLayout.Layout[]) => {
    if (!isEditMode) return;

    const updatedLayout: LayoutItem[] = newLayout.map((item) => {
      const original = currentLayout.find((l) => l.i === item.i);
      return {
        i: item.i,
        x: item.x,
        y: item.y,
        w: item.w,
        h: item.h,
        minW: original?.minW,
        minH: original?.minH,
        maxW: original?.maxW,
        maxH: original?.maxH,
      };
    });

    setLayout(updatedLayout);
  };

  return (
    <div className="dashboard-layout">
      <GridLayout
        className="layout"
        layout={layout}
        cols={cols}
        rowHeight={rowHeight}
        width={width}
        margin={margin}
        containerPadding={[0, 0]}
        isDraggable={isEditMode}
        isResizable={isEditMode}
        onLayoutChange={handleLayoutChange}
        draggableHandle=".glass-panel-header"
        resizeHandles={['se', 'sw', 'ne', 'nw']}
        compactType="vertical"
        preventCollision={false}
      >
        {panels.map((panel) => (
          <div key={panel.id}>
            <DraggablePanel
              id={panel.id}
              title={panel.title}
              noPadding={panel.noPadding}
            >
              {children[panel.id]}
            </DraggablePanel>
          </div>
        ))}
      </GridLayout>

      <style>{`
        .dashboard-layout {
          width: 100%;
        }

        .layout {
          position: relative;
        }

        .react-grid-item {
          transition: all 200ms ease;
          transition-property: left, top, width, height;
        }

        .react-grid-item.cssTransforms {
          transition-property: transform, width, height;
        }

        .react-grid-item.resizing {
          z-index: 1;
          will-change: width, height;
        }

        .react-grid-item.react-draggable-dragging {
          transition: none;
          z-index: 3;
          will-change: transform;
          opacity: 0.9;
        }

        .react-grid-item.dropping {
          visibility: hidden;
        }

        .react-grid-item > .react-resizable-handle {
          position: absolute;
          width: 20px;
          height: 20px;
        }

        .react-grid-item > .react-resizable-handle::after {
          content: '';
          position: absolute;
          right: 5px;
          bottom: 5px;
          width: 8px;
          height: 8px;
          border-right: 2px solid ${theme.textMuted}40;
          border-bottom: 2px solid ${theme.textMuted}40;
        }

        .react-grid-item:hover > .react-resizable-handle::after {
          border-color: ${theme.accent}80;
        }

        .react-grid-item > .react-resizable-handle.react-resizable-handle-sw {
          bottom: 0;
          left: 0;
          cursor: sw-resize;
          transform: rotate(90deg);
        }

        .react-grid-item > .react-resizable-handle.react-resizable-handle-se {
          bottom: 0;
          right: 0;
          cursor: se-resize;
        }

        .react-grid-item > .react-resizable-handle.react-resizable-handle-nw {
          top: 0;
          left: 0;
          cursor: nw-resize;
          transform: rotate(180deg);
        }

        .react-grid-item > .react-resizable-handle.react-resizable-handle-ne {
          top: 0;
          right: 0;
          cursor: ne-resize;
          transform: rotate(270deg);
        }

        .react-grid-placeholder {
          background: ${theme.accent}20;
          border: 2px dashed ${theme.accent};
          border-radius: 12px;
          opacity: 0.5;
        }
      `}</style>
    </div>
  );
};

export default DashboardLayout;
