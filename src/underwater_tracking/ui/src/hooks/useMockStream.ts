import { useEffect, useState } from "react";
import { createMockFrame, MOCK_FRAME_COUNT } from "../mocks/mockData";
import type { OperationalFrame } from "../types/frames";
import type { StreamStatus } from "./useWebSocket";

const MOCK_TICK_MS = 500;

export default function useMockStream(enabled: boolean): {
  frame: OperationalFrame | null;
  status: StreamStatus;
} {
  const [frame, setFrame] = useState<OperationalFrame | null>(null);
  const [status, setStatus] = useState<StreamStatus>("idle");

  useEffect(() => {
    if (!enabled) {
      setFrame(null);
      setStatus("idle");
      return undefined;
    }

    let frameIndex = 0;
    setStatus("connected");
    setFrame(createMockFrame(frameIndex));
    const timer = window.setInterval(() => {
      frameIndex = (frameIndex + 1) % MOCK_FRAME_COUNT;
      setFrame(createMockFrame(frameIndex));
    }, MOCK_TICK_MS);
    return () => window.clearInterval(timer);
  }, [enabled]);

  return { frame, status };
}
