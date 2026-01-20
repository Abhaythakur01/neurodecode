/**
 * DraggablePanel - Wrapper component for draggable/resizable panels.
 */

import React from 'react';
import { GlassPanel } from '../ui/GlassPanel';
import { useLayoutStore } from '../../stores/layoutStore';
import { useSettingsStore } from '../../stores/settingsStore';

interface DraggablePanelProps {
  id: string;
  title?: string;
  children: React.ReactNode;
  noPadding?: boolean;
}

export const DraggablePanel: React.FC<DraggablePanelProps> = ({
  id,
  title,
  children,
  noPadding = false,
}) => {
  const isEditMode = useLayoutStore((state) => state.isEditMode);
  const theme = useSettingsStore((state) => state.settings.theme);

  return (
    <div className={`draggable-panel ${isEditMode ? 'edit-mode' : ''}`} data-panel-id={id}>
      <GlassPanel
        header={
          title ? (
            <div className="panel-title-row">
              <span>{title}</span>
              {isEditMode && <span className="drag-indicator">⋮⋮</span>}
            </div>
          ) : undefined
        }
        noPadding={noPadding}
      >
        {children}
      </GlassPanel>

      <style>{`
        .draggable-panel {
          height: 100%;
          display: flex;
          flex-direction: column;
        }

        .draggable-panel .glass-panel {
          height: 100%;
          display: flex;
          flex-direction: column;
        }

        .draggable-panel .glass-panel-content {
          flex: 1;
          overflow: auto;
        }

        .draggable-panel.edit-mode {
          cursor: move;
        }

        .draggable-panel.edit-mode .glass-panel {
          border-style: dashed;
          border-color: ${theme.accent}50;
        }

        .panel-title-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          width: 100%;
        }

        .drag-indicator {
          opacity: 0.5;
          font-size: 14px;
          letter-spacing: 2px;
        }

        .draggable-panel.edit-mode:hover .glass-panel {
          border-color: ${theme.accent};
        }

        .react-grid-item.react-grid-placeholder {
          background: ${theme.accent}30;
          border: 2px dashed ${theme.accent};
          border-radius: 12px;
        }
      `}</style>
    </div>
  );
};

export default DraggablePanel;
