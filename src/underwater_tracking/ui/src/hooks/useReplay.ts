import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { EventView, OperationalFrame } from "../frameTypes";

/**
 * Replay browser for recorded operational frames (migrated from the
 * reference project's useReplay hook; data-flow shape preserved).
 *
 * Frames are fetched in chunks as the operator approaches the end of the
 * loaded data.  Later tasks adapt this to the plan's time-range replay API
 * (/api/replay?start_s=&end_s=) and frame store (Task 7).
 */

const CHUNK_SIZE = 120;
/** Event types worth pinning on the playback timeline. */
const MARKER_EVENT_TYPES = ["target_found", "plan_decision", "target_lost"];

interface ReplayListResponse {
  files: string[];
}

interface ReplayChunkResponse {
  frames: Array<OperationalFrame | null>;
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

export interface ReplayMarker {
  frameIndex: number;
  type: string;
}

export default function useReplay(enabled: boolean) {
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [frames, setFrames] = useState<Array<OperationalFrame | null>>([]);
  const [total, setTotal] = useState(0);
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const framesRef = useRef<Array<OperationalFrame | null>>([]);
  const totalRef = useRef(0);
  const loadedOffsetsRef = useRef(new Set<string>());

  useEffect(() => {
    if (!enabled) return;
    setError("");
    fetch("/api/replay/list")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ReplayListResponse>;
      })
      .then((data) => setFiles(data.files || []))
      .catch(() => setError("无法读取回放列表"));
  }, [enabled]);

  /** Fetch one chunk [offset, offset+CHUNK_SIZE) and merge it into framesRef. */
  const fetchChunk = useCallback(async (filename: string, offset: number) => {
    const key = `${filename}|${offset}`;
    if (loadedOffsetsRef.current.has(key)) return;
    loadedOffsetsRef.current.add(key);
    const url = `/api/replay?file=${encodeURIComponent(filename)}&offset=${offset}&limit=${CHUNK_SIZE}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = (await response.json()) as ReplayChunkResponse & { error?: string };
    if (data.error) throw new Error(data.error);

    const current = [...framesRef.current];
    for (let i = 0; i < data.frames.length; i += 1) {
      const dest = offset + i;
      if (dest < current.length) {
        current[dest] = data.frames[i];
      } else {
        // Extend with sparse holes (filled by subsequent chunks).
        while (current.length < dest) current.push(null);
        current.push(data.frames[i]);
      }
    }
    framesRef.current = current;
    totalRef.current = data.total;
    setFrames([...current]);
    setTotal(data.total);
  }, []);

  const load = useCallback(
    async (filename: string) => {
      setSelectedFile(filename);
      setIsPlaying(false);
      setError("");
      if (!filename) {
        framesRef.current = [];
        loadedOffsetsRef.current.clear();
        setFrames([]);
        setTotal(0);
        setIndex(0);
        return;
      }
      setLoading(true);
      loadedOffsetsRef.current.clear();
      framesRef.current = [];
      totalRef.current = 0;
      setFrames([]);
      setTotal(0);
      setIndex(0);
      try {
        await fetchChunk(filename, 0);
      } catch {
        framesRef.current = [];
        setFrames([]);
        setError("回放加载失败");
      } finally {
        setLoading(false);
      }
    },
    [fetchChunk],
  );

  /** Preload the chunk that contains targetIndex when it is not loaded yet. */
  const ensureLoaded = useCallback(
    async (targetIndex: number) => {
      const safe = Math.max(0, Math.min(targetIndex, Math.max(0, totalRef.current - 1)));
      const slot = framesRef.current[safe];
      if (slot !== null && slot !== undefined) return;
      const chunkOffset = Math.floor(safe / CHUNK_SIZE) * CHUNK_SIZE;
      try {
        await fetchChunk(selectedFile, chunkOffset);
      } catch {
        // Error state is surfaced through `load`; chunk preload is best-effort.
      }
    },
    [fetchChunk, selectedFile],
  );

  const seek = useCallback(
    (nextIndex: number) => {
      const upper = Math.max(0, totalRef.current - 1);
      const clamped = Math.max(0, Math.min(upper, Number(nextIndex) || 0));
      setIndex(clamped);
      void ensureLoaded(clamped);
    },
    [ensureLoaded],
  );

  useEffect(() => {
    if (!enabled || !isPlaying) return undefined;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= Math.max(0, totalRef.current - 1)) {
          setIsPlaying(false);
          return current;
        }
        const next = current + 1;
        void ensureLoaded(next);
        return next;
      });
    }, 1000 / speed);
    return () => window.clearInterval(timer);
  }, [enabled, isPlaying, speed, ensureLoaded]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!enabled) return;
      const tag = (event.target as HTMLElement | null)?.tagName || "";
      if (/INPUT|SELECT|TEXTAREA/.test(tag)) return;
      if (event.code === "Space") {
        event.preventDefault();
        setIsPlaying((current) => !current);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        seek(index - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        seek(index + 1);
      } else if (/^[0-9]$/.test(event.key)) {
        seek(Math.round((Number(event.key) / 10) * Math.max(0, totalRef.current - 1)));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled, index, seek]);

  const markers = useMemo(() => {
    const unique = new Map<string, ReplayMarker>();
    frames.forEach((frame, frameIndex) => {
      if (!frame) return;
      (frame.events || [] as EventView[])
        .filter((event) => MARKER_EVENT_TYPES.includes(event.type))
        .forEach((event) => {
          const key = `${event.time}|${event.type}|${JSON.stringify(event.data)}`;
          if (!unique.has(key)) {
            unique.set(key, { frameIndex, type: event.type });
          }
        });
    });
    return [...unique.values()];
  }, [frames]);

  const frame = (() => {
    const slot = frames[index];
    if (slot !== null && slot !== undefined) return slot;
    // Fallback: nearest non-null frame around the requested index.
    for (let offset = 0; offset < Math.max(frames.length, 10); offset += 1) {
      const before = frames[index - offset];
      if (before !== null && before !== undefined) return before;
      const after = frames[index + offset];
      if (after !== null && after !== undefined) return after;
    }
    return null;
  })();

  return {
    files,
    selectedFile,
    load,
    frames,
    total,
    frame,
    index,
    seek,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    loading,
    error,
    markers,
  };
}
