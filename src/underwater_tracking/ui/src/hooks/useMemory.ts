import { useCallback, useEffect, useRef, useState } from "react";
import {
  getMemorySnapshot,
  getMemoryStream,
  type MemorySnapshotView,
  type MemoryStreamEventView,
  type MemoryStatus,
} from "../services/memoryApi";

interface UseMemoryOptions {
  userId: string;
  conversationId: string;
  scenarioId?: string;
  enabled: boolean;
  refreshKey?: number;
}

export default function useMemory({
  userId,
  conversationId,
  scenarioId,
  enabled,
  refreshKey = 0,
}: UseMemoryOptions) {
  const [snapshot, setSnapshot] = useState<MemorySnapshotView | null>(null);
  const [events, setEvents] = useState<MemoryStreamEventView[]>([]);
  const [, setCursor] = useState(0);
  const cursorRef = useRef(0);
  const [snapshotStatus, setSnapshotStatus] = useState<MemoryStatus | "idle">("idle");
  const [streamStatus, setStreamStatus] = useState<MemoryStatus | "idle">("idle");
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [streamLoading, setStreamLoading] = useState(false);
  const [snapshotError, setSnapshotError] = useState("");
  const [streamError, setStreamError] = useState("");
  const [streamDegradedReason, setStreamDegradedReason] = useState<string | null>(null);
  const generationRef = useRef(0);
  const snapshotRequestRef = useRef(0);
  const streamFlightRef = useRef<number | null>(null);
  const scopeKey = `${userId}\u0000${conversationId}\u0000${scenarioId ?? ""}`;
  const scopeReady = enabled && Boolean(scenarioId);

  const refresh = useCallback(async () => {
    if (!scopeReady || !scenarioId) return;
    const generation = generationRef.current;
    const requestId = ++snapshotRequestRef.current;
    setSnapshotLoading(true);
    setSnapshotError("");
    try {
      const next = await getMemorySnapshot({ userId, conversationId, scenarioId });
      if (generation !== generationRef.current || requestId !== snapshotRequestRef.current) return;
      setSnapshot(next);
      setSnapshotStatus(next.memory_status);
    } catch (cause: unknown) {
      if (generation === generationRef.current && requestId === snapshotRequestRef.current) {
        setSnapshotError(cause instanceof Error ? cause.message : "无法读取记忆快照");
      }
    } finally {
      if (generation === generationRef.current && requestId === snapshotRequestRef.current) {
        setSnapshotLoading(false);
      }
    }
  }, [conversationId, scopeReady, scenarioId, userId]);

  const pollStream = useCallback(async () => {
    if (!scopeReady || !scenarioId || streamFlightRef.current !== null) return;
    const generation = generationRef.current;
    const flightId = generation + Date.now();
    streamFlightRef.current = flightId;
    setStreamLoading(true);
    try {
      const next = await getMemoryStream({
        userId,
        conversationId,
        scenarioId,
        afterCursor: cursorRef.current,
      });
      if (generation !== generationRef.current) return;
      setEvents((current) => {
        const byId = new Map(current.map((event) => [event.event_id, event]));
        next.events.forEach((event) => byId.set(event.event_id, event));
        return [...byId.values()].sort((left, right) => left.cursor - right.cursor).slice(-300);
      });
      const nextCursor = Math.max(cursorRef.current, next.next_cursor);
      cursorRef.current = nextCursor;
      setCursor(nextCursor);
      setStreamStatus(next.memory_status);
      setStreamDegradedReason(next.degraded_reason ?? null);
      setStreamError("");
    } catch (cause: unknown) {
      if (generation === generationRef.current) {
        setStreamError(cause instanceof Error ? cause.message : "无法读取记忆流");
      }
    } finally {
      if (streamFlightRef.current === flightId) {
        streamFlightRef.current = null;
        setStreamLoading(false);
      }
    }
  }, [conversationId, scopeReady, scenarioId, userId]);

  useEffect(() => {
    generationRef.current += 1;
    snapshotRequestRef.current += 1;
    streamFlightRef.current = null;
    setSnapshot(null);
    setEvents([]);
    setCursor(0);
    cursorRef.current = 0;
    setSnapshotError("");
    setStreamError("");
    setStreamDegradedReason(null);
    setSnapshotStatus("idle");
    setStreamStatus("idle");
    setSnapshotLoading(scopeReady);
    setStreamLoading(false);
    if (!scopeReady) {
      return undefined;
    }
    void refresh();
    void pollStream();
    const timer = window.setInterval(() => {
      void refresh();
      void pollStream();
    }, 5_000);
    return () => window.clearInterval(timer);
  }, [refresh, pollStream, scopeKey, scopeReady]);

  useEffect(() => {
    if (scopeReady && refreshKey > 0) void refresh();
  }, [refresh, refreshKey, scopeReady]);

  const error = snapshotError || streamError;
  return {
    snapshot,
    events,
    cursor: cursorRef.current,
    snapshotStatus,
    status: streamStatus,
    loading: snapshotLoading,
    streamLoading,
    error,
    refresh,
    scopeUnavailable: enabled && !scenarioId,
    snapshotLoading,
    snapshotError,
    streamStatus,
    streamError,
    streamDegradedReason,
  };
}
