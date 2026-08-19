import { useEffect, useMemo, useState } from "react";
import { Grid3X3, History, PanelBottom, PanelRight, Radio, Route, Search, Wind } from "lucide-react";
import BottomDrawer from "./components/BottomDrawer";
import CanvasMap, { type TrailMode } from "./components/CanvasMap";
import AssignmentPanel from "./components/assistant/AssignmentPanel";
import AssignmentReview from "./components/assistant/AssignmentReview";
import ConversationPanel from "./components/assistant/ConversationPanel";
import EvaluationPanel from "./components/evaluation/EvaluationPanel";
import PlaybackBar from "./components/PlaybackBar";
import RightSidebar from "./components/RightSidebar";
import SonarBadges from "./components/map/SonarBadges";
import {
  applyDirective,
  assignTargets,
  AssistantApiError,
  setSensorMode,
  waitForDirectiveStatus,
  type DirectiveStatus,
} from "./services/assistantApi";
import type { EventView, OperationalFrame } from "./types/frames";
import { DEFAULT_VIEW_CONFIG } from "./types/viewConfig";
import useReplay from "./hooks/useReplay";
import useWebSocket, { type StreamStatus } from "./hooks/useWebSocket";

type Mode = "live" | "replay";

const CONNECTION_LABELS: Record<StreamStatus, string> = {
  idle: "待机", connecting: "连接中", connected: "实时连接", reconnecting: "正在重连", error: "数据错误",
};
const evaluationEnabled = import.meta.env.VITE_EVALUATION_ENABLED === "true";

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
  const [trailMode, setTrailMode] = useState<TrailMode>("tail");
  const [liveEvents, setLiveEvents] = useState<EventView[]>([]);
  const [highlightEvidenceId, setHighlightEvidenceId] = useState<string | null>(null);
  const [assignmentNotice, setAssignmentNotice] = useState("");
  const [assignmentJob, setAssignmentJob] = useState<DirectiveStatus | null>(null);
  const [assignmentBusy, setAssignmentBusy] = useState(false);
  const [assignmentError, setAssignmentError] = useState("");
  const [replayStart, setReplayStart] = useState("0");
  const [replayEnd, setReplayEnd] = useState("");

  const live = useWebSocket(mode === "live");
  const replay = useReplay(mode === "replay");
  const frame: OperationalFrame | null = mode === "live" ? live.frame : replay.frame;
  const liveFrame = live.frame;

  useEffect(() => {
    if (!liveFrame) return;
    setLiveEvents((current) => {
      const byId = new Map(current.map((event) => [event.event_id, event]));
      liveFrame.events.forEach((event) => byId.set(event.event_id, event));
      return [...byId.values()].sort((left, right) => left.sim_time_s - right.sim_time_s).slice(-300);
    });
  }, [liveFrame]);

  const replayEvents = useMemo(() => {
    const byId = new Map<string, EventView>();
    replay.frames.slice(0, replay.index + 1).forEach((item) => item.events.forEach((event) => byId.set(event.event_id, event)));
    return [...byId.values()].sort((left, right) => left.sim_time_s - right.sim_time_s);
  }, [replay.frames, replay.index]);

  useEffect(() => {
    if (selectedUuvId && frame && !frame.uuvs.some((uuv) => uuv.uuv_id === selectedUuvId)) setSelectedUuvId(null);
  }, [frame, selectedUuvId]);

  useEffect(() => {
    if (!assignmentJob || !["queued", "processing", "applying"].includes(assignmentJob.status)) return undefined;
    let active = true;
    void waitForDirectiveStatus(assignmentJob.request_id)
      .then((next) => {
        if (!active) return;
        setAssignmentJob(next);
        setAssignmentError(next.error ?? "");
      })
      .catch((reason: unknown) => {
        if (!active) return;
        const message = errorMessage(reason);
        setAssignmentJob((current) => current ? { ...current, status: "error", error: message } : current);
        setAssignmentError(message);
      });
    return () => { active = false; };
  }, [assignmentJob?.request_id, assignmentJob?.status]);

  const selectedTargetIds = useMemo(() => {
    if (!frame) return [];
    const selected = frame.uuvs.find((uuv) => uuv.uuv_id === selectedUuvId);
    const targetId = selected?.group_id ? frame.groups.find((group) => group.group_id === selected.group_id)?.target_id : undefined;
    return targetId ? [targetId] : frame.target_estimates.slice(0, 1).map((target) => target.target_id);
  }, [frame, selectedUuvId]);

  const handleAssignment = (uuvIds: string[], targetId: string) => {
    if (mode !== "live" || !liveFrame) return;
    setAssignmentError("");
    setAssignmentNotice("指派预览已排队…");
    void assignTargets({ target_id: targetId, uuv_ids: uuvIds, expected_plan_version: liveFrame.plan_version })
      .then((response) => {
        setAssignmentJob({ request_id: response.request_id, status: response.status });
        setAssignmentNotice(`指派请求 ${response.request_id} 已排队，等待预览。`);
      })
      .catch((reason: unknown) => setAssignmentNotice(errorMessage(reason)));
  };

  const confirmAssignment = async () => {
    if (!assignmentJob) return;
    setAssignmentBusy(true);
    setAssignmentError("");
    try {
      const response = await applyDirective(assignmentJob.request_id);
      setAssignmentJob((current) => current ? { ...current, status: response.status } : current);
      setAssignmentNotice("指派已确认，等待下一轮 LangGraph 重规划后生效。");
    } catch (reason: unknown) {
      setAssignmentError(errorMessage(reason));
    } finally {
      setAssignmentBusy(false);
    }
  };

  const handleSensorMode = (uuvId: string, modeValue: "passive" | "active", targetId: string | null) => {
    if (mode !== "live" || !liveFrame) return;
    setAssignmentError("");
    setAssignmentNotice(`${uuvId} 声纳模式调整已排队…`);
    void setSensorMode({
      uuv_id: uuvId,
      mode: modeValue,
      target_id: targetId,
      expected_plan_version: liveFrame.plan_version,
    })
      .then(() => setAssignmentNotice(`${uuvId} 已切换为${modeValue === "active" ? "主动脉冲 + 被动持续" : "被动持续监听"}。`))
      .catch((reason: unknown) => setAssignmentError(errorMessage(reason)));
  };

  const selectEvidence = (evidenceId: string) => {
    setHighlightEvidenceId(evidenceId);
    setDrawerVisible(true);
  };

  const connection = mode === "live"
    ? CONNECTION_LABELS[live.status]
    : replay.error || (replay.loading ? "载入回放" : `${replay.total} 帧回放`);

  return <main className={`app-layout ${mode === "replay" ? "replay-active" : ""}`}>
    <header className="top-bar">
      <div className="product-mark" aria-label="水下跟踪指挥界面"><span className="mark-index">UT</span><span>水下跟踪 / 指挥台</span></div>
      <div className="mode-switch" aria-label="数据模式">
        <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}><Radio size={14} />实时</button>
        <button className={mode === "replay" ? "active" : ""} onClick={() => setMode("replay")}><History size={14} />回放</button>
      </div>
      {mode === "replay" && <div className="replay-range" aria-label="回放时间范围">
        <input type="number" min="0" value={replayStart} onChange={(event) => setReplayStart(event.target.value)} aria-label="回放开始秒数" placeholder="开始 s" />
        <span>—</span><input type="number" min="0" value={replayEnd} onChange={(event) => setReplayEnd(event.target.value)} aria-label="回放结束秒数" placeholder="结束 s" />
        <button onClick={() => void replay.loadRange(Number(replayStart) || 0, replayEnd ? Number(replayEnd) : undefined)} aria-label="加载回放范围"><Search size={14} /></button>
      </div>}
      <span className={`connection-state ${mode === "live" ? live.status : replay.loading ? "connecting" : "connected"}`}><span className="connection-dot" />{connection}</span>
      <div className="top-actions">
        <div className="trail-mode-switch" role="group" aria-label="UUV 轨迹显示模式"><button className={trailMode === "full" ? "active" : ""} onClick={() => setTrailMode("full")} aria-label="完整轨迹"><Route size={15} /></button><button className={trailMode === "tail" ? "active" : ""} onClick={() => setTrailMode("tail")} aria-label="长尾轨迹"><Wind size={15} /></button></div>
        <button className="trail-mode-compact" onClick={() => setTrailMode((value) => value === "full" ? "tail" : "full")} aria-label="切换轨迹模式"><Wind size={16} /></button>
        <button className={showPredictedRegions ? "icon-btn active" : "icon-btn"} onClick={() => setShowPredictedRegions((value) => !value)} aria-label="切换目标预测区域"><Route size={16} /></button>
        <button className={showRegionHandoffs ? "icon-btn active" : "icon-btn"} onClick={() => setShowRegionHandoffs((value) => !value)} aria-label="切换区域接力"><History size={16} /></button>
        <button className={showGrid ? "icon-btn active" : "icon-btn"} onClick={() => setShowGrid((value) => !value)} aria-label="切换网格"><Grid3X3 size={16} /></button>
        <button className={viewConfig.showDetectionRange ? "icon-btn active" : "icon-btn"} onClick={() => setViewConfig((value) => ({ ...value, showDetectionRange: !value.showDetectionRange }))} aria-label="切换探测范围"><Radio size={16} /></button>
        <button className={drawerVisible ? "icon-btn active" : "icon-btn"} onClick={() => setDrawerVisible((value) => !value)} aria-label="切换任务详情"><PanelBottom size={16} /></button>
        <button className="icon-btn mobile-only" onClick={() => setSidebarOpen((value) => !value)} aria-label="切换编队状态"><PanelRight size={16} /></button>
      </div>
    </header>

    <div className="map-stage">
      <CanvasMap frame={frame} selectedUuvId={selectedUuvId} onSelectUuv={setSelectedUuvId} selectedRegionId={selectedRegionId} onSelectRegion={setSelectedRegionId} showGrid={showGrid} showPredictedRegions={showPredictedRegions} showRegionHandoffs={showRegionHandoffs} showDetectionRange={viewConfig.showDetectionRange} trailMode={trailMode} viewConfig={viewConfig} />
      <SonarBadges uuvs={frame?.uuvs ?? []} missionModes={frame?.uuv_mission_modes} />
      {mode === "replay" && <div className="mode-banner">历史态势 · 专家干预已锁定</div>}
      <EvaluationPanel enabled={evaluationEnabled} simTimeS={frame?.sim_time_s ?? 0} />
    </div>
    <RightSidebar
      frame={frame}
      selectedUuvId={selectedUuvId}
      onSelectUuv={setSelectedUuvId}
      open={sidebarOpen}
      onClose={() => setSidebarOpen(false)}
      onSensorMode={handleSensorMode}
      predictionPanel={<>
        <AssignmentPanel
          targets={frame?.target_estimates ?? []}
          uuvs={frame?.uuvs ?? []}
          onAssign={handleAssignment}
          frame={frame}
          regionalPlans={frame?.regional_plans}
          selectedRegionId={selectedRegionId}
          onSelectRegion={setSelectedRegionId}
        />
        {assignmentNotice && <p className="assistant-notice" role="status">{assignmentNotice}</p>}
      </>}
      llmClientPanel={<>
        {mode === "live" && assignmentJob && <AssignmentReview job={assignmentJob} onConfirm={() => void confirmAssignment()} busy={assignmentBusy} error={assignmentError} />}
        <ConversationPanel frame={mode === "live" ? frame : null} selectedTargetIds={selectedTargetIds} disabled={mode !== "live"} onSelectEvidence={selectEvidence} />
      </>}
    />
    <BottomDrawer frame={frame} events={mode === "live" ? liveEvents : replayEvents} visible={drawerVisible} onToggle={() => setDrawerVisible((value) => !value)} onSelectEvidence={selectEvidence} highlightEvidenceId={highlightEvidenceId} selectedRegionId={selectedRegionId} onSelectRegion={setSelectedRegionId} />
    <PlaybackBar visible={mode === "replay"} isPlaying={replay.isPlaying} onPlayPause={() => replay.setIsPlaying((value) => !value)} frameIndex={replay.index} totalFrames={replay.total} onSeek={replay.seek} startTimeS={replay.startTimeS} endTimeS={replay.endTimeS} onSeekTime={replay.seekTime} playSpeed={viewConfig.playbackRate} onSpeedChange={(playbackRate) => { setViewConfig((value) => ({ ...value, playbackRate })); replay.setSpeed(playbackRate); }} frame={frame} markers={replay.markers} />
  </main>;
}

function errorMessage(reason: unknown): string {
  if (reason instanceof AssistantApiError && reason.status === 409) return "方案已更新，请重新确认当前版本后提交指派。";
  return reason instanceof Error ? reason.message : "指派请求失败，请检查连接。";
}
