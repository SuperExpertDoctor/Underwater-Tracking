import { useEffect, useRef, useState } from "react";
import type { OperationalFrame, StreamMessage } from "../types/frames";
import {
  createFrameStoreState,
  isHeartbeat,
  reduceOperationalFrame,
} from "../state/frameStore";

const FRAME_PUBLISH_INTERVAL_MS = 1000 / 60;
const HEARTBEAT_INTERVAL_MS = 25_000;
const RECONNECT_CAP_MS = 30_000;

export type StreamStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "error";

export default function useWebSocket(enabled: boolean): {
  frame: OperationalFrame | null;
  status: StreamStatus;
} {
  const [frame, setFrame] = useState<OperationalFrame | null>(null);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const frameRef = useRef<OperationalFrame | null>(null);
  const pendingFrameRef = useRef<OperationalFrame | null>(null);
  const storeRef = useRef(createFrameStoreState());
  const publishHandleRef = useRef<number | null>(null);
  const publishTimerRef = useRef<number | null>(null);
  const lastPublishedAtRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      frameRef.current = null;
      pendingFrameRef.current = null;
      storeRef.current = createFrameStoreState();
      setFrame(null);
      setStatus("idle");
      return undefined;
    }

    let disposed = false;
    let socket: WebSocket | null = null;
    let heartbeat: number | null = null;
    let reconnect: number | null = null;
    let retryCount = 0;
    let streamReady = false;
    let bufferedFrame: OperationalFrame | null = null;
    let snapshotRequestInFlight = false;

    const publishPending = () => {
      if (publishHandleRef.current !== null || publishTimerRef.current !== null) return;
      publishHandleRef.current = window.requestAnimationFrame(() => {
        publishHandleRef.current = null;
        const elapsed = performance.now() - lastPublishedAtRef.current;
        if (elapsed < FRAME_PUBLISH_INTERVAL_MS) {
          publishTimerRef.current = window.setTimeout(() => {
            publishTimerRef.current = null;
            publishPending();
          }, FRAME_PUBLISH_INTERVAL_MS - elapsed);
          return;
        }
        const next = pendingFrameRef.current;
        if (next) {
          lastPublishedAtRef.current = performance.now();
          frameRef.current = next;
          setFrame(next);
        }
      });
    };

    const acceptFrame = (next: OperationalFrame, source: "frame" | "snapshot" = "frame") => {
      const transition = reduceOperationalFrame(storeRef.current, {
        type: source,
        frame: next,
      });
      storeRef.current = transition.state;
      if (!transition.accepted || !transition.state.frame) return;
      pendingFrameRef.current = transition.state.frame;
      publishPending();
      if (transition.requestSnapshot && !snapshotRequestInFlight) {
        void loadSnapshot();
      }
    };

    const acceptBufferedFrame = () => {
      streamReady = true;
      if (bufferedFrame) {
        acceptFrame(bufferedFrame);
        bufferedFrame = null;
      }
    };

    const loadSnapshot = async () => {
      if (snapshotRequestInFlight) return;
      snapshotRequestInFlight = true;
      try {
        const response = await fetch("/api/operational/snapshot");
        if (response.ok) {
          const snapshot = (await response.json()) as OperationalFrame;
          if (isOperationalFrame(snapshot)) acceptFrame(snapshot, "snapshot");
        }
      } catch {
        if (!disposed) setStatus("error");
      } finally {
        snapshotRequestInFlight = false;
        if (!disposed) acceptBufferedFrame();
      }
    };

    const connect = () => {
      if (disposed) return;
      setStatus(retryCount > 0 ? "reconnecting" : "connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/operational`);
      socket.onopen = () => {
        const wasRetry = retryCount > 0;
        retryCount = 0;
        setStatus(wasRetry ? "connected" : "connected");
        heartbeat = window.setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) socket.send("ping");
        }, HEARTBEAT_INTERVAL_MS);
        void loadSnapshot();
      };
      socket.onmessage = (event) => {
        if (event.data === "pong") return;
        try {
          const parsed = JSON.parse(String(event.data)) as StreamMessage;
          if (isHeartbeat(parsed)) return;
          if (!isOperationalFrame(parsed)) {
            setStatus("error");
            return;
          }
          if (!streamReady) {
            const buffered = bufferedFrame;
            const transition = reduceOperationalFrame(
              createFrameStoreState(buffered),
              { type: "frame", frame: parsed },
            );
            bufferedFrame = transition.accepted ? parsed : buffered ?? parsed;
            return;
          }
          acceptFrame(parsed);
        } catch {
          setStatus("error");
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (heartbeat !== null) window.clearInterval(heartbeat);
        heartbeat = null;
        if (disposed) return;
        setStatus("reconnecting");
        const delay = Math.min(1000 * 2 ** retryCount, RECONNECT_CAP_MS);
        retryCount += 1;
        reconnect = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      disposed = true;
      if (heartbeat !== null) window.clearInterval(heartbeat);
      if (reconnect !== null) window.clearTimeout(reconnect);
      if (publishHandleRef.current !== null) window.cancelAnimationFrame(publishHandleRef.current);
      if (publishTimerRef.current !== null) window.clearTimeout(publishTimerRef.current);
      publishHandleRef.current = null;
      publishTimerRef.current = null;
      pendingFrameRef.current = null;
      frameRef.current = null;
      bufferedFrame = null;
      lastPublishedAtRef.current = 0;
      socket?.close();
    };
  }, [enabled]);

  return { frame, status };
}

function isOperationalFrame(value: unknown): value is OperationalFrame {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<OperationalFrame>;
  return typeof candidate.frame_id === "number" && typeof candidate.sim_time_s === "number";
}
