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
  executionRevision?: number;
  frameId?: number;
}

export default function useMemory({
  userId,
  conversationId,
  scenarioId,
  enabled,
  refreshKey = 0,
  executionRevision,
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
  const [streamExecutionRevision, setStreamExecutionRevision] = useState<number | null>(null);
  const [streamFrameId, setStreamFrameId] = useState<number | null>(null);
  const generationRef = useRef(0);
  const snapshotRequestRef = useRef(0);
  const snapshotAbortRef = useRef<AbortController | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamFlightRef = useRef<number | null>(null);
  const scopeKey = `${userId}\u0000${conversationId}\u0000${scenarioId ?? ""}\u0000${executionRevision ?? ""}`;
  const scopeReady = enabled && Boolean(scenarioId);

  const refresh = useCallback(async () => {
    if (!scopeReady || !scenarioId) return;
    const generation = generationRef.current;
    const requestId = ++snapshotRequestRef.current;
    snapshotAbortRef.current?.abort();
    const controller = new AbortController();
    snapshotAbortRef.current = controller;
    setSnapshotLoading(true);
    setSnapshotError("");
    try {
      const next = await getMemorySnapshot({
        userId,
        conversationId,
        scenarioId,
        signal: controller.signal,
      });
      if (generation !== generationRef.current || requestId !== snapshotRequestRef.current) return;
      setSnapshot(next);
      setSnapshotStatus(next.memory_status);
    } catch (cause: unknown) {
      if (
        !isAbortError(cause)
        && generation === generationRef.current
        && requestId === snapshotRequestRef.current
      ) {
        setSnapshotError(cause instanceof Error ? cause.message : "无法读取记忆快照");
      }
    } finally {
      if (snapshotAbortRef.current === controller) snapshotAbortRef.current = null;
      if (generation === generationRef.current && requestId === snapshotRequestRef.current) {
        setSnapshotLoading(false);
      }
    }
  }, [conversationId, executionRevision, scopeReady, scenarioId, userId]);

  const pollStream = useCallback(async () => {
    if (!scopeReady || !scenarioId || streamFlightRef.current !== null) return;
    const generation = generationRef.current;
    const flightId = generation + Date.now();
    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;
    streamFlightRef.current = flightId;
    setStreamLoading(true);
    try {
      const next = await getMemoryStream({
        userId,
        conversationId,
        scenarioId,
        afterCursor: cursorRef.current,
        signal: controller.signal,
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
      setStreamExecutionRevision(next.execution_revision ?? null);
      setStreamFrameId(next.frame_id ?? null);
      setStreamError("");
    } catch (cause: unknown) {
      if (!isAbortError(cause) && generation === generationRef.current) {
        setStreamError(cause instanceof Error ? cause.message : "无法读取记忆流");
      }
    } finally {
      if (streamAbortRef.current === controller) streamAbortRef.current = null;
      if (streamFlightRef.current === flightId) {
        streamFlightRef.current = null;
        setStreamLoading(false);
      }
    }
  }, [conversationId, executionRevision, scopeReady, scenarioId, userId]);

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
    setStreamExecutionRevision(null);
    setStreamFrameId(null);
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
    return () => {
      window.clearInterval(timer);
      snapshotAbortRef.current?.abort();
      streamAbortRef.current?.abort();
      snapshotAbortRef.current = null;
      streamAbortRef.current = null;
      streamFlightRef.current = null;
    };
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
    streamExecutionRevision,
    streamFrameId,
  };
}

function isAbortError(reason: unknown): boolean {
  return typeof reason === "object" && reason !== null && "name" in reason
    && (reason as { name?: unknown }).name === "AbortError";
}
