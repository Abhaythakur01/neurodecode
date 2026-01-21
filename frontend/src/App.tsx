/**
 * App - Main application component for NeuroDecode BCI.
 *
 * Integrates all visualization components with draggable dashboard layout,
 * session recording/playback, and advanced analytics.
 */

import React, { useCallback, useEffect, useState, useRef } from 'react';
import CursorCanvas from './components/CursorCanvas';
import PerformanceChart from './components/PerformanceChart';
import DecoderPanel from './components/DecoderPanel';
import Controls from './components/Controls';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { LayoutSelector } from './components/layout/LayoutSelector';
import { SettingsPanel } from './components/settings/SettingsPanel';
import { SessionRecorder } from './components/session/SessionRecorder';
import { SessionList } from './components/session/SessionList';
import { SessionPlayback } from './components/session/SessionPlayback';
import { StatisticsSummary } from './components/analytics/StatisticsSummary';
import { Modal } from './components/ui/Modal';
import { useWebSocket } from './hooks/useWebSocket';
import { useSessionRecording } from './hooks/useSessionRecording';
import { useSessionPlayback } from './hooks/useSessionPlayback';
import { useAnalytics } from './hooks/useAnalytics';
import { useSettingsStore } from './stores/settingsStore';
import { apiUrl } from './config';
import { useLayoutStore } from './stores/layoutStore';
import { useSessionStore } from './stores/sessionStore';
import type {
  DecoderInfo,
  PerformanceMetrics,
  PredictionResponse,
  SimulationConfig,
  TrajectoryPoint,
} from './types';

const MAX_TRAJECTORY_LENGTH = 200;
const MAX_METRICS_LENGTH = 500;

const DEFAULT_CONFIG: SimulationConfig = {
  pattern: 'circular',
  speed: 1.0,
  noise_level: 0.1,
  n_neurons: 50,
};

function App() {
  // Refs for layout dimensions
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(1200);

  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [showLayouts, setShowLayouts] = useState(false);
  const [showSessions, setShowSessions] = useState(false);

  // Theme
  const theme = useSettingsStore((state) => state.settings.theme);
  const displaySettings = useSettingsStore((state) => state.settings.display);
  const chartSettings = useSettingsStore((state) => state.settings.charts);
  const currentPreset = useLayoutStore((state) => state.currentPreset);
  const loadSession = useSessionStore((state) => state.loadSession);
  const unloadSession = useSessionStore((state) => state.unloadSession);

  // Cursor state
  const [position, setPosition] = useState<[number, number]>([0, 0]);
  const [uncertainty, setUncertainty] = useState<[number, number]>([0.1, 0.1]);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);

  // Decoder state
  const [decoders, setDecoders] = useState<DecoderInfo[]>([]);
  const [selectedDecoders, setSelectedDecoders] = useState<string[]>([]);
  const [decoderWeights, setDecoderWeights] = useState<Record<string, number>>({});

  // Performance metrics
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    r2Scores: [],
    latencies: [],
    timestamps: [],
  });

  // Simulation state
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [config, setConfig] = useState<SimulationConfig>(DEFAULT_CONFIG);

  // Session recording
  const {
    isRecording,
    recordingDuration,
    frameCount,
    startRecording,
    stopRecording,
    cancelRecording,
    recordFrame,
  } = useSessionRecording();

  // Session playback
  const playback = useSessionPlayback({
    onFrame: (frame) => {
      setPosition(frame.position);
      setUncertainty(frame.uncertainty);
      setSelectedDecoders(frame.decoders);
      setDecoderWeights(frame.weights);
    },
  });

  // Analytics
  const analytics = useAnalytics();

  // Track container width for responsive layout
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Handle incoming predictions
  const handlePrediction = useCallback(
    (prediction: PredictionResponse) => {
      // Update position and uncertainty
      setPosition(prediction.prediction);
      setUncertainty(prediction.uncertainty);

      // Add to trajectory
      setTrajectory((prev) => {
        const newPoint: TrajectoryPoint = {
          x: prediction.prediction[0],
          y: prediction.prediction[1],
          timestamp: prediction.timestamp,
          uncertainty: prediction.uncertainty,
        };
        const updated = [...prev, newPoint];
        return updated.slice(-MAX_TRAJECTORY_LENGTH);
      });

      // Update decoder info
      setSelectedDecoders(prediction.selected_decoders);
      setDecoderWeights(prediction.decoder_weights);

      // Calculate average R² from decoder weights
      const avgWeight = Object.values(prediction.decoder_weights).reduce(
        (a, b) => a + b,
        0
      );

      // Update metrics
      setMetrics((prev) => {
        const newR2 = [...prev.r2Scores, avgWeight > 0 ? avgWeight : 0.5];
        const newLatencies = [...prev.latencies, prediction.latency_ms];
        const newTimestamps = [...prev.timestamps, prediction.timestamp];

        return {
          r2Scores: newR2.slice(-MAX_METRICS_LENGTH),
          latencies: newLatencies.slice(-MAX_METRICS_LENGTH),
          timestamps: newTimestamps.slice(-MAX_METRICS_LENGTH),
        };
      });

      // Record frame if recording
      if (isRecording) {
        recordFrame({
          position: prediction.prediction,
          uncertainty: prediction.uncertainty,
          decoders: prediction.selected_decoders,
          weights: prediction.decoder_weights,
          metrics: {
            r2: avgWeight > 0 ? avgWeight : 0.5,
            latency: prediction.latency_ms,
          },
        });
      }
    },
    [isRecording, recordFrame]
  );

  // Handle decoder state updates
  const handleDecoderStates = useCallback((states: DecoderInfo[]) => {
    setDecoders(states);
  }, []);

  // Handle errors
  const handleError = useCallback((error: string) => {
    console.error('WebSocket error:', error);
  }, []);

  // WebSocket hook
  const {
    connectionState,
    connect,
    disconnect,
    startSimulation,
    stopSimulation,
  } = useWebSocket({
    onPrediction: handlePrediction,
    onDecoderStates: handleDecoderStates,
    onError: handleError,
    autoConnect: false,
  });

  // Handle simulation start/stop
  const handleStartSimulation = useCallback(() => {
    startSimulation();
    setIsSimulationRunning(true);
  }, [startSimulation]);

  const handleStopSimulation = useCallback(() => {
    stopSimulation();
    setIsSimulationRunning(false);
  }, [stopSimulation]);

  // Handle calibration
  const handleCalibrate = useCallback(async () => {
    try {
      const response = await fetch(apiUrl('/api/simulation/calibrate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: 500 }),
      });
      const data = await response.json();
      console.log('Calibration result:', data);
    } catch (err) {
      console.error('Calibration failed:', err);
    }
  }, []);

  // Handle session load
  const handleLoadSession = useCallback(
    (sessionId: string) => {
      // Stop simulation if running
      if (isSimulationRunning) {
        handleStopSimulation();
      }
      loadSession(sessionId);
    },
    [isSimulationRunning, handleStopSimulation, loadSession]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          if (playback.hasSession) {
            if (playback.isPlaying) {
              playback.pause();
            } else {
              playback.play();
            }
          } else if (isSimulationRunning) {
            handleStopSimulation();
          } else if (connectionState.isConnected) {
            handleStartSimulation();
          }
          break;
        case 'KeyC':
          if (connectionState.isConnected && !isSimulationRunning) {
            handleCalibrate();
          }
          break;
        case 'KeyR':
          if (!connectionState.isConnected) {
            connect();
          }
          break;
        case 'KeyS':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            setShowSettings(true);
          }
          break;
        case 'Escape':
          setShowSettings(false);
          setShowLayouts(false);
          setShowSessions(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    connectionState.isConnected,
    isSimulationRunning,
    playback,
    handleStartSimulation,
    handleStopSimulation,
    handleCalibrate,
    connect,
  ]);

  // Define dashboard panels based on current layout preset
  const getPanels = () => {
    const basePanels = [
      { id: 'cursor', title: 'Neural Cursor', noPadding: true },
      { id: 'controls', title: 'Controls' },
      { id: 'performance', title: 'Performance' },
      { id: 'decoders', title: 'Decoders' },
    ];

    if (currentPreset === 'analytics') {
      basePanels.push({ id: 'statistics', title: 'Statistics' });
    }

    if (currentPreset === 'recording') {
      basePanels.push({ id: 'recording', title: 'Recording' });
      basePanels.push({ id: 'timeline', title: 'Timeline', noPadding: true });
    }

    return basePanels;
  };

  // Render panel content
  const renderPanelContent = () => ({
    cursor: (
      <CursorCanvas
        position={position}
        uncertainty={uncertainty}
        trajectory={trajectory}
        maxTrailLength={displaySettings.trailLength}
        width={480}
        height={480}
      />
    ),
    controls: (
      <div className="controls-wrapper">
        <Controls
          connectionState={connectionState}
          isSimulationRunning={isSimulationRunning}
          onConnect={connect}
          onDisconnect={disconnect}
          onStartSimulation={handleStartSimulation}
          onStopSimulation={handleStopSimulation}
          onCalibrate={handleCalibrate}
          config={config}
          onConfigChange={setConfig}
        />
        <div className="toolbar">
          <SessionRecorder
            isRecording={isRecording}
            duration={recordingDuration}
            frameCount={frameCount}
            onStart={startRecording}
            onStop={stopRecording}
            onCancel={cancelRecording}
            disabled={!isSimulationRunning && !connectionState.isConnected}
          />
        </div>
      </div>
    ),
    performance: (
      <PerformanceChart
        metrics={metrics}
        maxPoints={chartSettings.maxPoints}
        width={420}
        height={220}
        latencyThreshold={chartSettings.latencyThreshold}
      />
    ),
    decoders: (
      <DecoderPanel
        decoders={decoders}
        selectedDecoders={selectedDecoders}
        decoderWeights={decoderWeights}
      />
    ),
    statistics: <StatisticsSummary statistics={analytics.statistics} />,
    recording: (
      <SessionRecorder
        isRecording={isRecording}
        duration={recordingDuration}
        frameCount={frameCount}
        onStart={startRecording}
        onStop={stopRecording}
        onCancel={cancelRecording}
        disabled={!isSimulationRunning && !connectionState.isConnected}
      />
    ),
    timeline: playback.hasSession ? (
      <SessionPlayback
        isPlaying={playback.isPlaying}
        playbackState={playback.playbackState}
        currentTime={playback.currentTime}
        duration={playback.duration}
        progress={playback.progress}
        speed={playback.speed}
        sessionName={playback.sessionName}
        hasSession={playback.hasSession}
        onPlay={playback.play}
        onPause={playback.pause}
        onStop={playback.stop}
        onSeek={playback.seek}
        onSetSpeed={playback.setSpeed}
        onStepForward={playback.stepForward}
        onStepBackward={playback.stepBackward}
        onClose={unloadSession}
      />
    ) : (
      <div className="timeline-empty">No session loaded</div>
    ),
  });

  return (
    <div
      className="app"
      ref={containerRef}
      style={{
        '--accent': theme.accent,
        '--panel-bg': theme.panelBg,
        '--border-color': theme.borderColor,
        '--text': theme.text,
        '--text-muted': theme.textMuted,
      } as React.CSSProperties}
    >
      <header className="app-header">
        <div className="header-left">
          <h1>NeuroDecode BCI</h1>
          <span className="subtitle">Real-time Neural Decoding Interface</span>
        </div>
        <div className="header-right">
          <button className="header-btn" onClick={() => setShowSessions(true)}>
            Sessions
          </button>
          <button className="header-btn" onClick={() => setShowLayouts(true)}>
            Layout
          </button>
          <button className="header-btn" onClick={() => setShowSettings(true)}>
            Settings
          </button>
        </div>
      </header>

      {playback.hasSession && (
        <div className="playback-banner">
          <SessionPlayback
            isPlaying={playback.isPlaying}
            playbackState={playback.playbackState}
            currentTime={playback.currentTime}
            duration={playback.duration}
            progress={playback.progress}
            speed={playback.speed}
            sessionName={playback.sessionName}
            hasSession={playback.hasSession}
            onPlay={playback.play}
            onPause={playback.pause}
            onStop={playback.stop}
            onSeek={playback.seek}
            onSetSpeed={playback.setSpeed}
            onStepForward={playback.stepForward}
            onStepBackward={playback.stepBackward}
            onClose={unloadSession}
          />
        </div>
      )}

      <main className="app-main">
        <DashboardLayout
          width={containerWidth - 40}
          panels={getPanels()}
        >
          {renderPanelContent()}
        </DashboardLayout>
      </main>

      {/* Modals */}
      <SettingsPanel isOpen={showSettings} onClose={() => setShowSettings(false)} />

      <Modal
        isOpen={showLayouts}
        onClose={() => setShowLayouts(false)}
        title="Dashboard Layout"
        width="500px"
      >
        <LayoutSelector onClose={() => setShowLayouts(false)} />
      </Modal>

      <SessionList
        isOpen={showSessions}
        onClose={() => setShowSessions(false)}
        onLoadSession={handleLoadSession}
      />

      <style>{`
        .app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background: #0a0a0f;
        }

        .app-header {
          padding: 12px 24px;
          background: ${theme.panelBg};
          border-bottom: 1px solid ${theme.borderColor};
          display: flex;
          align-items: center;
          justify-content: space-between;
          backdrop-filter: blur(${theme.blur});
        }

        .header-left {
          display: flex;
          align-items: baseline;
          gap: 15px;
        }

        .app-header h1 {
          margin: 0;
          font-size: 20px;
          font-weight: 600;
          color: ${theme.accent};
          letter-spacing: -0.5px;
        }

        .subtitle {
          font-size: 13px;
          color: ${theme.textMuted};
        }

        .header-right {
          display: flex;
          gap: 8px;
        }

        .header-btn {
          padding: 8px 16px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 6px;
          color: ${theme.textMuted};
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .header-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          color: ${theme.text};
        }

        .playback-banner {
          padding: 0 20px;
          background: rgba(77, 166, 255, 0.05);
          border-bottom: 1px solid rgba(77, 166, 255, 0.1);
        }

        .app-main {
          flex: 1;
          padding: 20px;
          overflow: auto;
        }

        .controls-wrapper {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .toolbar {
          display: flex;
          gap: 12px;
          padding-top: 12px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .timeline-empty {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: ${theme.textMuted};
          font-size: 13px;
        }

        @media (max-width: 900px) {
          .header-left {
            flex-direction: column;
            gap: 4px;
          }

          .subtitle {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
