import { useEffect, useMemo, useState } from "react";
import {
  Grid3X3,
  History,
  PanelBottom,
  PanelRight,
  Radio,
  Route,
  Search,
} from "lucide-react";
import BottomDrawer, {
  type LlmThinkingHistoryItem,
} from "./components/BottomDrawer";
import CanvasMap from "./components/CanvasMap";
import AssignmentPanel from "./components/assistant/AssignmentPanel";
import ConversationPanel from "./components/assistant/ConversationPanel";
import EvaluationPanel from "./components/evaluation/EvaluationPanel";
import PlaybackBar from "./components/PlaybackBar";
import RightSidebar from "./components/RightSidebar";
import SonarBadges from "./components/map/SonarBadges";
import { setSensorMode } from "./services/assistantApi";
import type { EventView, OperationalFrame } from "./types/frames";
import { DEFAULT_VIEW_CONFIG } from "./types/viewConfig";
import useMockReplay from "./hooks/useMockReplay";
import useMockStream from "./hooks/useMockStream";
import useReplay from "./hooks/useReplay";
import useWebSocket, { type StreamStatus } from "./hooks/useWebSocket";

type Mode = "live" | "replay";

const CONNECTION_LABELS: Record<StreamStatus, string> = {
  idle: "待机",
  connecting: "连接中",
  connected: "实时连接",
  reconnecting: "正在重连",
  error: "数据错误",
};
const mockMode = import.meta.env.VITE_MOCK_MODE === "true";
const evaluationEnabled =
  !mockMode && import.meta.env.VITE_EVALUATION_ENABLED === "true";

export default function App() {
  const [mode, setMode] = useState<Mode>("live");
  const [selectedUuvId, setSelectedUuvId] = useState<string | null>(null);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showPredictedRegions, setShowPredictedRegions] = useState(true);
  const [showRegionHandoffs, setShowRegionHandoffs] = useState(true);
  const [viewConfig, setViewConfig] = useState(DEFAULT_VIEW_CONFIG);
  const [liveEvents, setLiveEvents] = useState<EventView[]>([]);
  const [liveThinkingHistory, setLiveThinkingHistory] = useState<
    LlmThinkingHistoryItem[]
  >([]);
  const [highlightEvidenceId, setHighlightEvidenceId] = useState<string | null>(
    null,
  );
  const [replayStart, setReplayStart] = useState("0");
  const [replayEnd, setReplayEnd] = useState("");

  // Hooks stay mounted in both branches so switching between live/replay does not
  // change hook order. Mock mode simply disables the network-backed hooks.
  const live = useWebSocket(mode === "live" && !mockMode);
  const mockLive = useMockStream(mode === "live" && mockMode);
  const replay = useReplay(mode === "replay" && !mockMode);
  const mockReplay = useMockReplay(mode === "replay" && mockMode);
  const activeReplay = mockMode ? mockReplay : replay;
  const frame: OperationalFrame | null =
    mode === "live"
      ? mockMode
        ? mockLive.frame
        : live.frame
      : activeReplay.frame;
  const liveFrame = mockMode ? mockLive.frame : live.frame;

  useEffect(() => {
    if (!liveFrame) return;
    setLiveEvents((current) => {
      const byId = new Map(current.map((event) => [event.event_id, event]));
      liveFrame.events.forEach((event) => byId.set(event.event_id, event));
      return [...byId.values()]
        .sort((left, right) => left.sim_time_s - right.sim_time_s)
        .slice(-300);
    });
  }, [liveFrame]);

  useEffect(() => {
    if (!liveFrame?.llm_thinking) return;
    const next: LlmThinkingHistoryItem = {
      sim_time_s: liveFrame.sim_time_s,
      plan_version: liveFrame.plan_version,
      content: liveFrame.llm_thinking,
      trigger: liveFrame.llm_thinking_trigger ?? null,
    };
    setLiveThinkingHistory((current) => {
      const previous = current.at(-1);
      if (
        previous?.content === next.content &&
        previous.trigger === next.trigger
      )
        return current;
      if (previous && next.sim_time_s < previous.sim_time_s) return [next];
      return [...current, next];
    });
  }, [liveFrame]);

  const replayEvents = useMemo(() => {
    const byId = new Map<string, EventView>();
    activeReplay.frames
      .slice(0, activeReplay.index + 1)
      .forEach((item) =>
        item.events.forEach((event) => byId.set(event.event_id, event)),
      );
    return [...byId.values()].sort(
      (left, right) => left.sim_time_s - right.sim_time_s,
    );
  }, [activeReplay.frames, activeReplay.index]);

  const replayThinkingHistory = useMemo(() => {
    const history: LlmThinkingHistoryItem[] = [];
    activeReplay.frames.slice(0, activeReplay.index + 1).forEach((item) => {
      if (
        !item.llm_thinking ||
        (history.at(-1)?.content === item.llm_thinking &&
          history.at(-1)?.trigger === (item.llm_thinking_trigger ?? null))
      )
        return;
      history.push({
        sim_time_s: item.sim_time_s,
        plan_version: item.plan_version,
        content: item.llm_thinking,
        trigger: item.llm_thinking_trigger ?? null,
      });
    });
    return history;
  }, [activeReplay.frames, activeReplay.index]);
  const thinkingHistory =
    mode === "live" ? liveThinkingHistory : replayThinkingHistory;

  useEffect(() => {
    if (
      selectedUuvId &&
      frame &&
      !frame.uuvs.some((uuv) => uuv.uuv_id === selectedUuvId)
    )
      setSelectedUuvId(null);
  }, [frame, selectedUuvId]);

  const selectedTargetIds = useMemo(() => {
    if (!frame) return [];
    const selected = frame.uuvs.find((uuv) => uuv.uuv_id === selectedUuvId);
    const targetId = selected?.group_id
      ? frame.groups.find((group) => group.group_id === selected.group_id)
          ?.target_id
      : undefined;
    return targetId
      ? [targetId]
      : frame.target_estimates.slice(0, 1).map((target) => target.target_id);
  }, [frame, selectedUuvId]);

  const handleSensorMode = (
    uuvId: string,
    modeValue: "passive" | "active",
    targetId: string | null,
  ) => {
    if (mode !== "live" || !liveFrame) return;
    void setSensorMode({
      uuv_id: uuvId,
      mode: modeValue,
      target_id: targetId,
      expected_plan_version: liveFrame.plan_version,
    }).catch(() => undefined);
  };

  const selectEvidence = (evidenceId: string) => {
    setHighlightEvidenceId(evidenceId);
    setDrawerVisible(true);
  };

  const connection =
    mode === "live"
      ? CONNECTION_LABELS[mockMode ? mockLive.status : live.status]
      : activeReplay.error ||
        (activeReplay.loading ? "载入回放" : `${activeReplay.total} 帧回放`);

  return (
    <main className={`app-layout ${mode === "replay" ? "replay-active" : ""}`}>
      <header className="top-bar">
        <div className="product-mark" aria-label="水下跟踪指挥界面">
          <span className="mark-index">UT</span>
          <span>水下跟踪 / 指挥台</span>
        </div>
        <div className="run-mode-controls">
          {mockMode && (
            <span
              className="mock-mode-chip"
              title="当前仅使用浏览器本地 Mock 数据"
            >
              MOCK 数据
            </span>
          )}
          <div className="mode-switch" aria-label="数据模式">
            <button
              className={mode === "live" ? "active" : ""}
              onClick={() => setMode("live")}
            >
              <Radio size={14} />
              实时
            </button>
            <button
              className={mode === "replay" ? "active" : ""}
              onClick={() => setMode("replay")}
            >
              <History size={14} />
              回放
            </button>
          </div>
        </div>
        {mode === "replay" && (
          <div className="replay-range" aria-label="回放时间范围">
            <input
              type="number"
              min="0"
              value={replayStart}
              onChange={(event) => setReplayStart(event.target.value)}
              aria-label="回放开始秒数"
              placeholder="开始 s"
            />
            <span>—</span>
            <input
              type="number"
              min="0"
              value={replayEnd}
              onChange={(event) => setReplayEnd(event.target.value)}
              aria-label="回放结束秒数"
              placeholder="结束 s"
            />
            <button
              onClick={() =>
                void activeReplay.loadRange(
                  Number(replayStart) || 0,
                  replayEnd ? Number(replayEnd) : undefined,
                )
              }
              aria-label="加载回放范围"
            >
              <Search size={14} />
            </button>
          </div>
        )}
        <span
          className={`connection-state ${mode === "live" ? (mockMode ? mockLive.status : live.status) : activeReplay.loading ? "connecting" : "connected"}`}
        >
          <span className="connection-dot" />
          {connection}
        </span>
        <div className="top-actions">
          <button
            className={showPredictedRegions ? "icon-btn active" : "icon-btn"}
            onClick={() => setShowPredictedRegions((value) => !value)}
            aria-label="切换目标预测区域"
          >
            <Route size={16} />
          </button>
          <button
            className={showRegionHandoffs ? "icon-btn active" : "icon-btn"}
            onClick={() => setShowRegionHandoffs((value) => !value)}
            aria-label="切换区域接力"
          >
            <History size={16} />
          </button>
          <button
            className={showGrid ? "icon-btn active" : "icon-btn"}
            onClick={() => setShowGrid((value) => !value)}
            aria-label="切换网格"
          >
            <Grid3X3 size={16} />
          </button>
          <button
            className={
              viewConfig.showDetectionRange ? "icon-btn active" : "icon-btn"
            }
            onClick={() =>
              setViewConfig((value) => ({
                ...value,
                showDetectionRange: !value.showDetectionRange,
              }))
            }
            aria-label="切换探测范围"
          >
            <Radio size={16} />
          </button>
          <button
            className={drawerVisible ? "icon-btn active" : "icon-btn"}
            onClick={() => setDrawerVisible((value) => !value)}
            aria-label="切换任务详情"
          >
            <PanelBottom size={16} />
          </button>
          <button
            className="icon-btn mobile-only"
            onClick={() => setSidebarOpen((value) => !value)}
            aria-label="切换编队状态"
          >
            <PanelRight size={16} />
          </button>
        </div>
      </header>

      <div className="map-stage">
        <CanvasMap
          frame={frame}
          selectedUuvId={selectedUuvId}
          onSelectUuv={setSelectedUuvId}
          selectedRegionId={selectedRegionId}
          onSelectRegion={setSelectedRegionId}
          showGrid={showGrid}
          showPredictedRegions={showPredictedRegions}
          showRegionHandoffs={showRegionHandoffs}
          showDetectionRange={viewConfig.showDetectionRange}
          trailMode="tail"
          viewConfig={viewConfig}
        />
        <SonarBadges uuvs={frame?.uuvs ?? []} />
        {mode === "replay" && (
          <div className="mode-banner">历史态势 · 专家干预已锁定</div>
        )}
        <EvaluationPanel
          enabled={evaluationEnabled}
          simTimeS={frame?.sim_time_s ?? 0}
        />
      </div>
      <RightSidebar
        frame={frame}
        selectedUuvId={selectedUuvId}
        onSelectUuv={setSelectedUuvId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSensorMode={handleSensorMode}
        predictionPanel={
          <>
            <AssignmentPanel
              targets={frame?.target_estimates ?? []}
              uuvs={frame?.uuvs ?? []}
              regionalPlans={frame?.regional_plans}
              selectedRegionId={selectedRegionId}
              onSelectRegion={setSelectedRegionId}
            />
          </>
        }
        llmClientPanel={
          <ConversationPanel
            frame={mode === "live" ? frame : null}
            selectedTargetIds={selectedTargetIds}
            disabled={mode !== "live"}
          />
        }
      />
      <BottomDrawer
        frame={frame}
        events={mode === "live" ? liveEvents : replayEvents}
        thinkingHistory={thinkingHistory}
        visible={drawerVisible}
        onToggle={() => setDrawerVisible((value) => !value)}
        onSelectEvidence={selectEvidence}
        highlightEvidenceId={highlightEvidenceId}
        selectedRegionId={selectedRegionId}
        onSelectRegion={setSelectedRegionId}
        dockedToPlayback={mode === "replay"}
      />
      <PlaybackBar
        visible={mode === "replay"}
        isPlaying={activeReplay.isPlaying}
        onPlayPause={() => activeReplay.setIsPlaying((value) => !value)}
        frameIndex={activeReplay.index}
        totalFrames={activeReplay.total}
        onSeek={activeReplay.seek}
        startTimeS={activeReplay.startTimeS}
        endTimeS={activeReplay.endTimeS}
        onSeekTime={activeReplay.seekTime}
        playSpeed={viewConfig.playbackRate}
        onSpeedChange={(playbackRate) => {
          setViewConfig((value) => ({ ...value, playbackRate }));
          activeReplay.setSpeed(playbackRate);
        }}
        frame={frame}
        markers={activeReplay.markers}
      />
    </main>
  );
}
