import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";
import { RadioTower } from "lucide-react";
import type { OperationalFrame } from "../frameTypes";

export type TrailMode = "tail" | "full" | "comet";

interface CanvasMapProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
  showGrid: boolean;
  trailMode: TrailMode;
}

/**
 * Tactical map region (migrated from the reference project's CanvasMap;
 * component boundary preserved).  The shell keeps the canvas surface, size
 * tracking, empty state and entity hit-testing; real rendering of estimates,
 * bearings, groups and routes is added in Task 6 (TacticalMap + layers).
 * Grid and trail preferences are exposed as data attributes for the later
 * render layers.
 */
export default function CanvasMap({
  frame,
  selectedUuvId,
  onSelectUuv,
  showGrid,
  trailMode,
}: CanvasMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const updateSize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleClick = useCallback(
    (event: MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || !frame) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      // Grid-cell hit test with a fixed cell size, mirroring the reference.
      const cellSize = 20;
      for (const uuv of frame.uuvs) {
        const [col, row] = uuv.position;
        const centerX = (col + 0.5) * cellSize;
        const centerY = (row + 0.5) * cellSize;
        if (Math.hypot(mouseX - centerX, mouseY - centerY) < Math.max(9, cellSize * 0.55)) {
          onSelectUuv(uuv.id === selectedUuvId ? null : uuv.id);
          return;
        }
      }
    },
    [frame, onSelectUuv, selectedUuvId],
  );

  return (
    <div
      className="canvas-area"
      ref={containerRef}
      data-show-grid={showGrid}
      data-trail-mode={trailMode}
    >
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{ cursor: hovered ? "crosshair" : "default" }}
        aria-label="Operational map"
      />
      {!frame && (
        <div className="map-empty" role="status">
          <RadioTower size={22} />
          <strong>WAITING FOR OPERATIONAL DATA</strong>
          <span>Live telemetry or replay frames will appear here.</span>
        </div>
      )}
      <div className="map-scale" aria-hidden="true"><i />20 KM</div>
    </div>
  );
}
