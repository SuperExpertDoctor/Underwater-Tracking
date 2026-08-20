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
  const [status, setStatus] = useState<MemoryStatus | "idle">("idle");
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError("");
    try {
      const next = await getMemorySnapshot({ userId, conversationId, scenarioId });
      setSnapshot(next);
      setStatus(next.memory_status);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "无法读取记忆快照");
    } finally {
      setLoading(false);
    }
  }, [conversationId, enabled, scenarioId, userId]);

  const pollStream = useCallback(async () => {
    if (!enabled) return;
    try {
      const next = await getMemoryStream({
        userId,
        conversationId,
        scenarioId,
        afterCursor: cursorRef.current,
      });
      setEvents((current) => {
        const byId = new Map(current.map((event) => [event.event_id, event]));
        next.events.forEach((event) => byId.set(event.event_id, event));
        return [...byId.values()].sort((left, right) => left.cursor - right.cursor).slice(-300);
      });
      setCursor((current) => {
        const nextCursor = Math.max(current, next.next_cursor);
        cursorRef.current = nextCursor;
        return nextCursor;
      });
      setStatus(next.memory_status);
      if (next.degraded_reason) setError(next.degraded_reason);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "无法读取记忆流");
    }
  }, [conversationId, enabled, scenarioId, userId]);

  useEffect(() => {
    if (!enabled) {
      setSnapshot(null);
      setEvents([]);
      setCursor(0);
      cursorRef.current = 0;
      setStatus("idle");
      return undefined;
    }
    setEvents([]);
    setCursor(0);
    cursorRef.current = 0;
    void refresh();
    void pollStream();
    const timer = window.setInterval(() => {
      void refresh();
      void pollStream();
    }, 5_000);
    return () => window.clearInterval(timer);
  }, [enabled, refresh, pollStream]);

  useEffect(() => {
    if (enabled && refreshKey > 0) void refresh();
  }, [enabled, refresh, refreshKey]);

  return { snapshot, events, status, loading, error, refresh };
}
