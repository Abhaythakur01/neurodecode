/**
 * Frontend configuration
 *
 * Handles environment-specific settings for API and WebSocket connections.
 */

// API base URL - empty string means same origin (works with Vite proxy in dev)
export const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Helper to construct API URLs
export const apiUrl = (path: string): string => {
  const base = API_BASE_URL;
  // Ensure path starts with /
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalizedPath}`;
};
