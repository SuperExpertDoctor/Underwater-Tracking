import { useEffect, useState } from "react";
import type { OperationalFrame } from "../types/frames";
import type { StreamStatus } from "../hooks/useWebSocket";
import { MOCK_FRAMES } from "./mockFrames";

const MOCK_FRAME_INTERVAL_MS = 1_200;

/** A deterministic live-like stream. It never opens a socket or writes to the API. */
export default function useMockStream(enabled: boolean): {
  frame: OperationalFrame | null;
  status: StreamStatus;
} {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setIndex(0);
      return undefined;
    }
    setIndex(0);
    const timer = window.setInterval(() => {
      setIndex((current) => (current + 1) % MOCK_FRAMES.length);
    }, MOCK_FRAME_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [enabled]);

  return {
    frame: enabled ? MOCK_FRAMES[index] ?? MOCK_FRAMES[0] : null,
    status: enabled ? "connected" : "idle",
  };
}

