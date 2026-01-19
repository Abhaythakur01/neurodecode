/**
 * WebSocket hook for real-time BCI data streaming.
 *
 * Handles connection, reconnection, and message parsing.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ConnectionState,
  DecoderInfo,
  PredictionResponse,
  WebSocketMessage,
} from '../types';

const WS_URL = `ws://${window.location.hostname}:8000/ws/decode`;
const RECONNECT_DELAY = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;

interface UseWebSocketOptions {
  onPrediction?: (prediction: PredictionResponse) => void;
  onDecoderStates?: (states: DecoderInfo[]) => void;
  onError?: (error: string) => void;
  autoConnect?: boolean;
}

interface UseWebSocketReturn {
  connectionState: ConnectionState;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: object) => void;
  startSimulation: () => void;
  stopSimulation: () => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    onPrediction,
    onDecoderStates,
    onError,
    autoConnect = false,
  } = options;

  const [connectionState, setConnectionState] = useState<ConnectionState>({
    isConnected: false,
    isConnecting: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<number | null>(null);

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
  }, []);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      switch (message.type) {
        case 'prediction':
          onPrediction?.(message);
          if (message.decoder_states) {
            onDecoderStates?.(message.decoder_states);
          }
          break;

        case 'error':
          onError?.(message.error);
          break;

        case 'heartbeat':
          setConnectionState(prev => ({
            ...prev,
            lastPingTime: message.timestamp,
          }));
          break;

        case 'status':
          // Handle status updates if needed
          break;

        default:
          console.log('Unknown message type:', message);
      }
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  }, [onPrediction, onDecoderStates, onError]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    clearReconnectTimeout();

    setConnectionState({
      isConnected: false,
      isConnecting: true,
    });

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts.current = 0;
        setConnectionState({
          isConnected: true,
          isConnecting: false,
        });
      };

      ws.onmessage = handleMessage;

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState(prev => ({
          ...prev,
          error: 'Connection error',
        }));
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        wsRef.current = null;

        setConnectionState({
          isConnected: false,
          isConnecting: false,
          error: event.reason || undefined,
        });

        // Auto-reconnect
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          console.log(
            `Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttempts.current})`
          );
          reconnectTimeout.current = window.setTimeout(connect, RECONNECT_DELAY);
        } else {
          onError?.('Max reconnection attempts reached');
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setConnectionState({
        isConnected: false,
        isConnecting: false,
        error: 'Failed to connect',
      });
    }
  }, [handleMessage, clearReconnectTimeout, onError]);

  const disconnect = useCallback(() => {
    clearReconnectTimeout();
    reconnectAttempts.current = MAX_RECONNECT_ATTEMPTS; // Prevent auto-reconnect

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect');
      wsRef.current = null;
    }

    setConnectionState({
      isConnected: false,
      isConnecting: false,
    });
  }, [clearReconnectTimeout]);

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  const startSimulation = useCallback(() => {
    sendMessage({ type: 'start_simulation' });
  }, [sendMessage]);

  const stopSimulation = useCallback(() => {
    sendMessage({ type: 'stop_simulation' });
  }, [sendMessage]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      clearReconnectTimeout();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount');
      }
    };
  }, [autoConnect, connect, clearReconnectTimeout]);

  return {
    connectionState,
    connect,
    disconnect,
    sendMessage,
    startSimulation,
    stopSimulation,
  };
}

export default useWebSocket;
