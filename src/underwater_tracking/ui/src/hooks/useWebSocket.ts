import { useEffect, useRef, useState } from "react";
import type { OperationalFrame } from "../frameTypes";

/**
 * Live operational frame stream (migrated from the reference project's
 * useWebSocket hook; data-flow shape preserved, transport endpoint renamed
 * to the plan's /ws/operational).
 *
 * WebSocket delivery is asynchronous and can burst when the simulator is
 * faster than the display, so incoming snapshots are conflated and published
 * at a ~60 Hz animation cadence instead of scheduling an unbounded React
 * render queue.  On close the connection reconnects with exponential backoff.
 * Later tasks replace this with the full frame store (Task 7).
 */

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
  const retryCount = useRef(0);
  const pendingFrame = useRef<OperationalFrame | null>(null);
  const publishFrameHandle = useRef<number | null>(null);
  const publishTimerHandle = useRef<number | null>(null);
  const lastPublishedAt = useRef(0);
  const latestFrame = useRef<OperationalFrame | null>(null);

  useEffect(() => {
    if (!enabled) {
      setStatus("idle");
      return undefined;
    }
    let disposed = false;
    let socket: WebSocket | null = null;
    let heartbeat: number | null = null;
    let reconnect: number | null = null;

    const scheduleFramePublish = () => {
      if (publishFrameHandle.current != null || publishTimerHandle.current != null) {
        return;
      }
      publishFrameHandle.current = window.requestAnimationFrame(() => {
        publishFrameHandle.current = null;
        const elapsed = performance.now() - lastPublishedAt.current;
        if (elapsed < FRAME_PUBLISH_INTERVAL_MS) {
          publishTimerHandle.current = window.setTimeout(() => {
            publishTimerHandle.current = null;
            scheduleFramePublish();
          }, FRAME_PUBLISH_INTERVAL_MS - elapsed);
          return;
        }
        lastPublishedAt.current = performance.now();
        if (pendingFrame.current) {
          latestFrame.current = pendingFrame.current;
          setFrame(pendingFrame.current);
        }
      });
    };

    const connect = () => {
      if (disposed) return;
      setStatus(retryCount.current ? "reconnecting" : "connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/operational`);
      socket.onopen = () => {
        retryCount.current = 0;
        setStatus("connected");
        heartbeat = window.setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) socket.send("ping");
        }, HEARTBEAT_INTERVAL_MS);
      };
      socket.onmessage = (event) => {
        if (event.data === "pong") return;
        try {
          const next = JSON.parse(event.data) as OperationalFrame;
          if (next?.frame_id != null) {
            pendingFrame.current = {
              ...(pendingFrame.current ?? latestFrame.current ?? {}),
              ...next,
            } as OperationalFrame;
            scheduleFramePublish();
          }
        } catch {
          setStatus("error");
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (heartbeat != null) window.clearInterval(heartbeat);
        if (disposed) return;
        setStatus("reconnecting");
        const delay = Math.min(1000 * 2 ** retryCount.current, RECONNECT_CAP_MS);
        retryCount.current += 1;
        reconnect = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      disposed = true;
      if (heartbeat != null) window.clearInterval(heartbeat);
      if (reconnect != null) window.clearTimeout(reconnect);
      if (publishFrameHandle.current != null) {
        window.cancelAnimationFrame(publishFrameHandle.current);
      }
      if (publishTimerHandle.current != null) window.clearTimeout(publishTimerHandle.current);
      publishFrameHandle.current = null;
      publishTimerHandle.current = null;
      pendingFrame.current = null;
      lastPublishedAt.current = 0;
      latestFrame.current = null;
      socket?.close();
    };
  }, [enabled]);

  return { frame, status };
}
