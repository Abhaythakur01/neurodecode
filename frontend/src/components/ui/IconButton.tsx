/**
 * IconButton - Icon button component.
 */

import React from 'react';
import { useSettingsStore } from '../../stores/settingsStore';

interface IconButtonProps {
  onClick: () => void;
  children: React.ReactNode;
  title?: string;
  variant?: 'default' | 'primary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  active?: boolean;
}

export const IconButton: React.FC<IconButtonProps> = ({
  onClick,
  children,
  title,
  variant = 'default',
  size = 'md',
  disabled = false,
  active = false,
}) => {
  const theme = useSettingsStore((state) => state.settings.theme);

  const sizeStyles = {
    sm: { width: 28, height: 28, fontSize: 14 },
    md: { width: 36, height: 36, fontSize: 18 },
    lg: { width: 44, height: 44, fontSize: 22 },
  };

  const variantColors = {
    default: {
      bg: 'rgba(255, 255, 255, 0.08)',
      hoverBg: 'rgba(255, 255, 255, 0.12)',
      color: theme.text,
    },
    primary: {
      bg: `${theme.accent}20`,
      hoverBg: `${theme.accent}30`,
      color: theme.accent,
    },
    danger: {
      bg: 'rgba(255, 107, 107, 0.15)',
      hoverBg: 'rgba(255, 107, 107, 0.25)',
      color: '#ff6b6b',
    },
  };

  const colors = variantColors[variant];
  const sizeStyle = sizeStyles[size];

  return (
    <button
      className={`icon-button ${active ? 'active' : ''}`}
      onClick={onClick}
      title={title}
      disabled={disabled}
      style={{
        width: sizeStyle.width,
        height: sizeStyle.height,
        fontSize: sizeStyle.fontSize,
      }}
    >
      {children}
      <style>{`
        .icon-button {
          display: flex;
          align-items: center;
          justify-content: center;
          border: none;
          border-radius: 8px;
          background: ${colors.bg};
          color: ${colors.color};
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .icon-button:hover:not(:disabled) {
          background: ${colors.hoverBg};
          transform: scale(1.05);
        }

        .icon-button:active:not(:disabled) {
          transform: scale(0.95);
        }

        .icon-button.active {
          background: ${theme.accent};
          color: #000;
        }

        .icon-button:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
      `}</style>
    </button>
  );
};

export default IconButton;
