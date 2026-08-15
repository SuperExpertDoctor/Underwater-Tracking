import { useEffect, useMemo, useState } from "react";
import { Grid3X3, History, PanelBottom, PanelRight, Radio, Route, Wind } from "lucide-react";
import BottomDrawer from "./components/BottomDrawer";
import CanvasMap, { type TrailMode } from "./components/CanvasMap";
import PlaybackBar from "./components/PlaybackBar";
import RightSidebar from "./components/RightSidebar";
import type { EventView, OperationalFrame, PlanCycleView } from "./frameTypes";
import useReplay from "./hooks/useReplay";
import useWebSocket from "./hooks/useWebSocket";

type Mode = "live" | "replay";

const CONNECTION_LABELS: Record<string, string> = {
  idle: "待机",
  connecting: "连接中",
  connected: "实时连接",
  reconnecting: "正在重连",
  error: "数据错误",
};

/**
 * Command UI shell (migrated from the reference project's App; component
 * boundaries and data flow preserved, domain renamed to this plan's
 * UUV/target/belief/group/plan semantics).  The live stream hook and replay
 * hook keep their reference data-flow shapes; real transport and frame
 * store land in Task 7.
 */
export default function App() {
  const [mode, setMode] = useState<Mode>("live");
  const [selectedUuvId, setSelectedUuvId] = useState<string | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [trailMode, setTrailMode] = useState<TrailMode>("tail");
  const [liveEvents, setLiveEvents] = useState<EventView[]>([]);
  const [lastPlanCycle, setLastPlanCycle] = useState<PlanCycleView | null>(null);

  const live = useWebSocket(mode === "live");
  const replay = useReplay(mode === "replay");
  const frame: OperationalFrame | null = mode === "live" ? live.frame : replay.frame;

  // Merge live events deduplicated by (time, type, data); keep the tail.
  useEffect(() => {
    if (mode !== "live" || !live.frame) return;
    const incoming = live.frame.events || [];
    setLiveEvents((current) => {
      const keys = new Set(current.map((event) => eventKey(event)));
      const merged = [...current];
      for (const event of incoming) {
        if (!keys.has(eventKey(event))) merged.push(event);
      }
      return merged.slice(-300);
    });
    if (live.frame.plan_cycle) setLastPlanCycle(live.frame.plan_cycle);
  }, [live.frame, mode]);

  const replayEvents = useMemo(() => {
    if (mode !== "replay") return [];
    const unique = new Map<string, EventView>();
    replay.frames.slice(0, replay.index + 1).forEach((item) => {
      if (!item) return;
      (item.events || []).forEach((event) => unique.set(eventKey(event), event));
    });
    return [...unique.values()];
  }, [mode, replay.frames, replay.index]);

  const replayPlanCycle = useMemo(() => {
    if (mode !== "replay") return null;
    for (let index = replay.index; index >= 0; index -= 1) {
      const cycle = replay.frames[index]?.plan_cycle;
      if (cycle) return cycle;
    }
    return null;
  }, [mode, replay.frames, replay.index]);
  const displayedPlanCycle = mode === "replay" ? replayPlanCycle : lastPlanCycle;

  // Deselect when the selected UUV leaves the visible frame.
  useEffect(() => {
    if (selectedUuvId && frame && !frame.uuvs.some((uuv) => uuv.id === selectedUuvId)) {
      setSelectedUuvId(null);
    }
  }, [frame, selectedUuvId]);

  const connectionLabel = CONNECTION_LABELS[live.status] || live.status;

  return (
    <main className={`app-layout ${mode === "replay" ? "replay-active" : ""}`}>
      <header className="top-bar">
        <div className="product-mark" aria-label="水下跟踪指挥界面">
          <span className="mark-index">UT</span>
          <span>水下跟踪指挥界面</span>
        </div>
        <div className="mode-switch" aria-label="数据模式">
          <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>
            <Radio size={15} />直播
          </button>
          <button className={mode === "replay" ? "active" : ""} onClick={() => setMode("replay")}>
            <History size={15} />回放
          </button>
        </div>
        {mode === "replay" && (
          <select
            className="file-select"
            value={replay.selectedFile}
            onChange={(event) => void replay.load(event.target.value)}
            aria-label="选择回放文件"
          >
            <option value="">选择任务记录</option>
            {replay.files.map((file) => <option key={file} value={file}>{file}</option>)}
          </select>
        )}
        <span className={`connection-state ${mode === "live" ? live.status : replay.loading ? "connecting" : "connected"}`}>
          <span className="connection-dot" />
          {mode === "live" ? connectionLabel : replay.error || (replay.loading ? "载入中" : `${replay.frames.length} 帧`)}
        </span>
        <div className="top-actions">
          <div className="trail-mode-switch" role="group" aria-label="UUV 轨迹显示模式">
            <button
              className={trailMode === "full" ? "active" : ""}
              onClick={() => setTrailMode("full")}
              title="完整 UUV 轨迹"
              aria-label="完整 UUV 轨迹"
              aria-pressed={trailMode === "full"}
            >
              <Route size={16} />
            </button>
            <button
              className={trailMode === "tail" ? "active" : ""}
              onClick={() => setTrailMode("tail")}
              title="渐变长尾 UUV 轨迹"
              aria-label="渐变长尾 UUV 轨迹"
              aria-pressed={trailMode === "tail"}
            >
              <Wind size={16} />
            </button>
          </div>
          <button
            className="trail-mode-compact"
            onClick={() => setTrailMode((value) => (value === "full" ? "tail" : "full"))}
            title={`UUV 轨迹: ${trailMode === "full" ? "完整" : "渐变长尾"} — 点击切换`}
            aria-label="切换 UUV 轨迹显示模式"
          >
            {trailMode === "full" ? <Route size={17} /> : <Wind size={17} />}
          </button>
          <button
            className={showGrid ? "icon-btn active" : "icon-btn"}
            onClick={() => setShowGrid((value) => !value)}
            title="网格"
            aria-label="切换网格"
          >
            <Grid3X3 size={17} />
          </button>
          <button
            className={drawerVisible ? "icon-btn active" : "icon-btn"}
            onClick={() => setDrawerVisible((value) => !value)}
            title="任务详情"
            aria-label="切换任务详情面板"
            aria-pressed={drawerVisible}
          >
            <PanelBottom size={17} />
          </button>
          <button
            className="icon-btn mobile-only"
            onClick={() => setSidebarOpen((value) => !value)}
            title="编队状态"
            aria-label="切换编队状态面板"
          >
            <PanelRight size={17} />
          </button>
        </div>
      </header>

      <CanvasMap
        frame={frame}
        selectedUuvId={selectedUuvId}
        onSelectUuv={setSelectedUuvId}
        showGrid={showGrid}
        trailMode={trailMode}
      />
      <RightSidebar
        frame={frame}
        selectedUuvId={selectedUuvId}
        onSelectUuv={setSelectedUuvId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        lastPlanCycle={displayedPlanCycle}
      />
      <BottomDrawer
        frame={frame}
        events={mode === "live" ? liveEvents : replayEvents}
        planCycle={displayedPlanCycle}
        visible={drawerVisible}
        onToggle={() => setDrawerVisible((value) => !value)}
      />
      <PlaybackBar
        visible={mode === "replay"}
        isPlaying={replay.isPlaying}
        onPlayPause={() => replay.setIsPlaying((value) => !value)}
        frameIndex={replay.index}
        totalFrames={replay.total || replay.frames.length}
        onSeek={replay.seek}
        playSpeed={replay.speed}
        onSpeedChange={replay.setSpeed}
        frame={frame}
        markers={replay.markers}
      />
    </main>
  );
}

function eventKey(event: EventView): string {
  return `${event.time}|${event.type}|${JSON.stringify(event.data)}`;
}
