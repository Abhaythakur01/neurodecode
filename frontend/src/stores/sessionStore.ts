/**
 * Session store using Zustand for recording/playback management.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import type {
  SessionRecording,
  RecordingFrame,
  RecordingState,
  PlaybackControl,
  SessionMetadata
} from '../types';
import { MAX_STORED_SESSIONS, MAX_FRAMES_PER_SESSION } from '../constants/defaults';

interface SessionState {
  // Sessions list
  sessions: SessionRecording[];

  // Recording state
  recording: RecordingState;

  // Playback state
  playback: PlaybackControl;
  activeSession: SessionRecording | null;

  // Recording actions
  startRecording: () => void;
  stopRecording: (metadata: SessionMetadata) => SessionRecording | null;
  addFrame: (frame: Omit<RecordingFrame, 'timestamp'>) => void;
  cancelRecording: () => void;

  // Session management
  saveSession: (session: SessionRecording) => void;
  deleteSession: (id: string) => void;
  renameSession: (id: string, name: string) => void;
  clearAllSessions: () => void;

  // Playback actions
  loadSession: (id: string) => void;
  unloadSession: () => void;
  play: () => void;
  pause: () => void;
  stop: () => void;
  seek: (time: number) => void;
  setSpeed: (speed: number) => void;
  nextFrame: () => RecordingFrame | null;
}

const initialRecordingState: RecordingState = {
  isRecording: false,
  startTime: null,
  frames: [],
};

const initialPlaybackState: PlaybackControl = {
  state: 'idle',
  currentTime: 0,
  duration: 0,
  speed: 1,
  currentFrameIndex: 0,
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      recording: initialRecordingState,
      playback: initialPlaybackState,
      activeSession: null,

      // Recording actions
      startRecording: () => {
        set({
          recording: {
            isRecording: true,
            startTime: Date.now(),
            frames: [],
          },
        });
      },

      stopRecording: (metadata: SessionMetadata) => {
        const { recording } = get();
        if (!recording.isRecording || recording.frames.length === 0) {
          set({ recording: initialRecordingState });
          return null;
        }

        const duration = Date.now() - (recording.startTime || Date.now());
        const session: SessionRecording = {
          id: uuidv4(),
          name: `Recording ${new Date().toLocaleString()}`,
          createdAt: recording.startTime || Date.now(),
          duration,
          frames: recording.frames,
          metadata,
        };

        set({ recording: initialRecordingState });
        return session;
      },

      addFrame: (frameData) => {
        const { recording } = get();
        if (!recording.isRecording) return;
        if (recording.frames.length >= MAX_FRAMES_PER_SESSION) return;

        const frame: RecordingFrame = {
          ...frameData,
          timestamp: Date.now() - (recording.startTime || Date.now()),
        };

        set({
          recording: {
            ...recording,
            frames: [...recording.frames, frame],
          },
        });
      },

      cancelRecording: () => {
        set({ recording: initialRecordingState });
      },

      // Session management
      saveSession: (session: SessionRecording) => {
        const { sessions } = get();
        let updatedSessions = [session, ...sessions];

        // Limit stored sessions
        if (updatedSessions.length > MAX_STORED_SESSIONS) {
          updatedSessions = updatedSessions.slice(0, MAX_STORED_SESSIONS);
        }

        set({ sessions: updatedSessions });
      },

      deleteSession: (id: string) => {
        const { sessions, activeSession } = get();
        set({
          sessions: sessions.filter((s) => s.id !== id),
          activeSession: activeSession?.id === id ? null : activeSession,
        });
      },

      renameSession: (id: string, name: string) => {
        const { sessions, activeSession } = get();
        set({
          sessions: sessions.map((s) => (s.id === id ? { ...s, name } : s)),
          activeSession:
            activeSession?.id === id ? { ...activeSession, name } : activeSession,
        });
      },

      clearAllSessions: () => {
        set({
          sessions: [],
          activeSession: null,
          playback: initialPlaybackState,
        });
      },

      // Playback actions
      loadSession: (id: string) => {
        const { sessions } = get();
        const session = sessions.find((s) => s.id === id);
        if (session) {
          set({
            activeSession: session,
            playback: {
              ...initialPlaybackState,
              duration: session.duration,
            },
          });
        }
      },

      unloadSession: () => {
        set({
          activeSession: null,
          playback: initialPlaybackState,
        });
      },

      play: () => {
        const { activeSession } = get();
        if (!activeSession) return;
        set((state) => ({
          playback: { ...state.playback, state: 'playing' },
        }));
      },

      pause: () => {
        set((state) => ({
          playback: { ...state.playback, state: 'paused' },
        }));
      },

      stop: () => {
        set((state) => ({
          playback: {
            ...state.playback,
            state: 'idle',
            currentTime: 0,
            currentFrameIndex: 0,
          },
        }));
      },

      seek: (time: number) => {
        const { activeSession, playback } = get();
        if (!activeSession) return;

        const clampedTime = Math.max(0, Math.min(time, activeSession.duration));

        // Find frame index for this time
        let frameIndex = 0;
        for (let i = 0; i < activeSession.frames.length; i++) {
          if (activeSession.frames[i].timestamp <= clampedTime) {
            frameIndex = i;
          } else {
            break;
          }
        }

        set({
          playback: {
            ...playback,
            currentTime: clampedTime,
            currentFrameIndex: frameIndex,
          },
        });
      },

      setSpeed: (speed: number) => {
        set((state) => ({
          playback: { ...state.playback, speed: Math.max(0.25, Math.min(4, speed)) },
        }));
      },

      nextFrame: () => {
        const { activeSession, playback } = get();
        if (!activeSession || playback.state !== 'playing') return null;

        const nextIndex = playback.currentFrameIndex + 1;
        if (nextIndex >= activeSession.frames.length) {
          // End of session
          set({
            playback: {
              ...playback,
              state: 'idle',
              currentTime: activeSession.duration,
              currentFrameIndex: activeSession.frames.length - 1,
            },
          });
          return null;
        }

        const frame = activeSession.frames[nextIndex];
        set({
          playback: {
            ...playback,
            currentTime: frame.timestamp,
            currentFrameIndex: nextIndex,
          },
        });
        return frame;
      },
    }),
    {
      name: 'neurodecode-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
      }),
    }
  )
);
