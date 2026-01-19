/**
 * CursorCanvas - Real-time cursor visualization using HTML5 Canvas.
 *
 * Renders the decoded cursor position with trajectory trail and
 * uncertainty ellipse for 60fps smooth animation.
 */

import React, { useCallback, useEffect, useRef } from 'react';
import type { TrajectoryPoint } from '../types';

interface CursorCanvasProps {
  /** Current cursor position [-1, 1] normalized coordinates */
  position: [number, number];
  /** Uncertainty estimate [x_std, y_std] */
  uncertainty: [number, number];
  /** Trajectory history for trail */
  trajectory: TrajectoryPoint[];
  /** Maximum trail length */
  maxTrailLength?: number;
  /** Canvas width */
  width?: number;
  /** Canvas height */
  height?: number;
}

const CURSOR_RADIUS = 12;
const TRAIL_WIDTH = 3;
const GRID_COLOR = 'rgba(100, 100, 120, 0.3)';
const CURSOR_COLOR = '#00ff88';
const TRAIL_COLOR = 'rgba(0, 255, 136, 0.6)';
const UNCERTAINTY_COLOR = 'rgba(0, 255, 136, 0.15)';
const TARGET_ZONE_COLOR = 'rgba(255, 200, 100, 0.2)';

export const CursorCanvas: React.FC<CursorCanvasProps> = ({
  position,
  uncertainty,
  trajectory,
  maxTrailLength = 100,
  width = 500,
  height = 500,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // Convert normalized coordinates [-1, 1] to canvas coordinates
  const toCanvasCoords = useCallback(
    (x: number, y: number): [number, number] => {
      const margin = 40;
      const canvasX = margin + ((x + 1) / 2) * (width - 2 * margin);
      const canvasY = margin + ((1 - y) / 2) * (height - 2 * margin);
      return [canvasX, canvasY];
    },
    [width, height]
  );

  // Scale uncertainty to canvas size
  const scaleUncertainty = useCallback(
    (std: number): number => {
      const margin = 40;
      return (std * (width - 2 * margin)) / 2;
    },
    [width]
  );

  // Draw the canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;

    // Vertical lines
    for (let i = 0; i <= 4; i++) {
      const x = 40 + (i * (width - 80)) / 4;
      ctx.beginPath();
      ctx.moveTo(x, 40);
      ctx.lineTo(x, height - 40);
      ctx.stroke();
    }

    // Horizontal lines
    for (let i = 0; i <= 4; i++) {
      const y = 40 + (i * (height - 80)) / 4;
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 40, y);
      ctx.stroke();
    }

    // Draw center crosshair
    ctx.strokeStyle = 'rgba(150, 150, 170, 0.5)';
    ctx.lineWidth = 1;
    const [centerX, centerY] = toCanvasCoords(0, 0);
    ctx.beginPath();
    ctx.moveTo(centerX - 10, centerY);
    ctx.lineTo(centerX + 10, centerY);
    ctx.moveTo(centerX, centerY - 10);
    ctx.lineTo(centerX, centerY + 10);
    ctx.stroke();

    // Draw target zones (corners)
    ctx.fillStyle = TARGET_ZONE_COLOR;
    const targetRadius = 30;
    const corners = [
      [-0.8, 0.8],
      [0.8, 0.8],
      [-0.8, -0.8],
      [0.8, -0.8],
    ];
    corners.forEach(([tx, ty]) => {
      const [tcx, tcy] = toCanvasCoords(tx, ty);
      ctx.beginPath();
      ctx.arc(tcx, tcy, targetRadius, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw trajectory trail
    if (trajectory.length > 1) {
      const recentTrail = trajectory.slice(-maxTrailLength);

      ctx.strokeStyle = TRAIL_COLOR;
      ctx.lineWidth = TRAIL_WIDTH;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      const [startX, startY] = toCanvasCoords(
        recentTrail[0].x,
        recentTrail[0].y
      );
      ctx.moveTo(startX, startY);

      for (let i = 1; i < recentTrail.length; i++) {
        const [px, py] = toCanvasCoords(recentTrail[i].x, recentTrail[i].y);
        ctx.lineTo(px, py);

        // Fade trail
        const alpha = 0.1 + (i / recentTrail.length) * 0.5;
        ctx.strokeStyle = `rgba(0, 255, 136, ${alpha})`;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(px, py);
      }
    }

    // Draw uncertainty ellipse
    const [cursorX, cursorY] = toCanvasCoords(position[0], position[1]);
    const uncertaintyX = scaleUncertainty(uncertainty[0]) * 2;
    const uncertaintyY = scaleUncertainty(uncertainty[1]) * 2;

    ctx.fillStyle = UNCERTAINTY_COLOR;
    ctx.beginPath();
    ctx.ellipse(cursorX, cursorY, uncertaintyX, uncertaintyY, 0, 0, Math.PI * 2);
    ctx.fill();

    // Draw uncertainty ring
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(cursorX, cursorY, uncertaintyX, uncertaintyY, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw cursor
    ctx.fillStyle = CURSOR_COLOR;
    ctx.shadowColor = CURSOR_COLOR;
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(cursorX, cursorY, CURSOR_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    // Draw cursor center dot
    ctx.fillStyle = '#ffffff';
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(cursorX, cursorY, 3, 0, Math.PI * 2);
    ctx.fill();

    // Draw position text
    ctx.fillStyle = '#888';
    ctx.font = '12px monospace';
    ctx.fillText(
      `x: ${position[0].toFixed(3)} y: ${position[1].toFixed(3)}`,
      10,
      height - 10
    );
    ctx.fillText(
      `uncertainty: ${uncertainty[0].toFixed(3)}, ${uncertainty[1].toFixed(3)}`,
      10,
      height - 25
    );
  }, [
    position,
    uncertainty,
    trajectory,
    maxTrailLength,
    width,
    height,
    toCanvasCoords,
    scaleUncertainty,
  ]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [draw]);

  return (
    <div className="cursor-canvas-container">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          border: '1px solid #333',
          borderRadius: '8px',
          background: '#0a0a0f',
        }}
      />
    </div>
  );
};

export default CursorCanvas;
