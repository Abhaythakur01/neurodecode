/**
 * Export utilities for session data.
 */

import type { SessionRecording, ExportOptions } from '../types';

export function exportSessionToJSON(session: SessionRecording, options: ExportOptions): string {
  const data: Record<string, unknown> = {
    id: session.id,
    name: session.name,
    createdAt: new Date(session.createdAt).toISOString(),
    duration: session.duration,
  };

  if (options.includeMetadata) {
    data.metadata = session.metadata;
  }

  if (options.includeAllFrames) {
    let frames = session.frames;

    if (options.dateRange) {
      frames = frames.filter(
        (f) =>
          f.timestamp >= options.dateRange!.start &&
          f.timestamp <= options.dateRange!.end
      );
    }

    data.frames = frames;
    data.frameCount = frames.length;
  } else {
    data.frameCount = session.frames.length;
  }

  return JSON.stringify(data, null, 2);
}

export function exportSessionToCSV(session: SessionRecording, options: ExportOptions): string {
  let frames = session.frames;

  if (options.dateRange) {
    frames = frames.filter(
      (f) =>
        f.timestamp >= options.dateRange!.start &&
        f.timestamp <= options.dateRange!.end
    );
  }

  // CSV header
  const headers = [
    'timestamp',
    'position_x',
    'position_y',
    'uncertainty_x',
    'uncertainty_y',
    'r2_score',
    'latency_ms',
    'active_decoders',
  ];

  // CSV rows
  const rows = frames.map((frame) => [
    frame.timestamp,
    frame.position[0].toFixed(6),
    frame.position[1].toFixed(6),
    frame.uncertainty[0].toFixed(6),
    frame.uncertainty[1].toFixed(6),
    frame.metrics.r2.toFixed(4),
    frame.metrics.latency.toFixed(2),
    frame.decoders.join(';'),
  ]);

  // Add metadata as comment if requested
  let csv = '';
  if (options.includeMetadata) {
    csv += `# Session: ${session.name}\n`;
    csv += `# ID: ${session.id}\n`;
    csv += `# Created: ${new Date(session.createdAt).toISOString()}\n`;
    csv += `# Duration: ${session.duration}ms\n`;
    csv += `# Average R²: ${session.metadata.averageR2.toFixed(4)}\n`;
    csv += `# Average Latency: ${session.metadata.averageLatency.toFixed(2)}ms\n`;
    csv += `# Decoders: ${session.metadata.selectedDecoders.join(', ')}\n`;
    csv += '#\n';
  }

  csv += headers.join(',') + '\n';
  csv += rows.map((row) => row.join(',')).join('\n');

  return csv;
}

export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportSession(
  session: SessionRecording,
  options: ExportOptions
): void {
  const timestamp = new Date().toISOString().slice(0, 10);
  const safeName = session.name.replace(/[^a-zA-Z0-9]/g, '_');

  if (options.format === 'json') {
    const content = exportSessionToJSON(session, options);
    downloadFile(content, `${safeName}_${timestamp}.json`, 'application/json');
  } else {
    const content = exportSessionToCSV(session, options);
    downloadFile(content, `${safeName}_${timestamp}.csv`, 'text/csv');
  }
}
