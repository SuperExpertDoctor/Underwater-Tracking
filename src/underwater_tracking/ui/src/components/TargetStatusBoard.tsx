import type { OperationalFrame, TaskGroupInstanceView, TargetEstimateView } from "../types/frames";
import { displayTargetName } from "../utils/presentation";
import {
  ESTIMATE_STATUS_LABELS,
  ESTIMATE_STATUS_SHORT_LABELS,
  GROUP_LIFECYCLE_LABELS,
  GROUP_SENSOR_LABELS,
  HANDOFF_STATUS_LABELS,
  deploymentProgress,
  entryEvidenceForRegion,
  estimateDataAge,
  estimateObservationIds,
  estimateTimestamp,
  estimateTrackRevision,
  estimateValidUntil,
  estimatePresentationStatus,
  formatPercent,
  formatSimSeconds,
  groupPassiveObservationIds,
  handoffProgress,
  scanCoveragePercent,
  scanPingCount,
  scanRouteProgressPercent,
  scanTelemetryForRegion,
  structuredBlockingReasons,
  targetGroups,
} from "../domain/operationalStatus";

interface TargetStatusBoardProps {
  frame: OperationalFrame;
}

function finite(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value);
}

function formatEstimateFreshness(frame: OperationalFrame, target: TargetEstimateView) {
  const timestamp = estimateTimestamp(target);
  const age = estimateDataAge(frame, target);
  const validUntil = estimateValidUntil(target);
  const revision = estimateTrackRevision(target);
  const observations = estimateObservationIds(target);
  const facts: string[] = [];
  if (timestamp != null) facts.push(`估计 ${formatSimSeconds(timestamp)}`);
  if (age != null) facts.push(`数据龄 ${formatSimSeconds(age)}`);
  if (validUntil != null) facts.push(`有效至 ${formatSimSeconds(validUntil)}`);
  if (revision != null) facts.push(`轨迹修订 v${revision}`);
  if (observations.length) facts.push(`观测 ${observations.length} 条`);
  return facts;
}

function lifecycleClass(group: TaskGroupInstanceView): string {
  return group.lifecycle.replaceAll("_", "-");
}

function groupOwnerLabel(frame: OperationalFrame, group: TaskGroupInstanceView): string {
  const ownerId = frame.execution?.tracking_control?.tracking_owner_group_id;
  if (ownerId === group.group_instance_id || group.ownership_status === "owner") return "OWNER / 当前负责";
  if (frame.execution?.tracking_control?.pending_successor_group_id === group.group_instance_id) return "SUCCESSOR / 待接替";
  return group.ownership_status || "未声明归属";
}

function GroupStatusRow({ frame, group }: { frame: OperationalFrame; group: TaskGroupInstanceView }) {
  const progress = deploymentProgress(frame, group);
  const deployed = progress.deployed == null ? "不可用" : `${progress.deployed}/${progress.total}`;
  const passiveIds = groupPassiveObservationIds(group);
  return (
    <div
      className={`target-status-group lifecycle-${lifecycleClass(group)}`}
      data-group-lifecycle={group.lifecycle}
      data-group-id={group.group_instance_id}
      data-owner-group-id={
        groupOwnerLabel(frame, group).startsWith("OWNER")
          ? group.group_instance_id
          : undefined
      }
    >
      <div className="target-status-group-heading">
        <strong>实例 · {group.group_instance_id}</strong>
        <span>{GROUP_LIFECYCLE_LABELS[group.lifecycle]}</span>
      </div>
      <div className="target-status-facts">
        <span>传感器 <b>{GROUP_SENSOR_LABELS[group.sensor_mode]}</b></span>
        <span>部署 <b data-deployment-progress={deployed}>{deployed}</b></span>
        <span>归属 <b>{groupOwnerLabel(frame, group)}</b></span>
      </div>
      <div className="target-status-members">
        {group.member_uuv_ids.map((id) => <span key={id}>{id}</span>)}
      </div>
      {passiveIds.length > 0 && (
        <small className="target-status-evidence">被动观测：{passiveIds.join("、")}</small>
      )}
      {(group.evidence_ids?.length ?? 0) > 0 && (
        <small className="target-status-evidence">证据：{group.evidence_ids?.join("、")}</small>
      )}
      {group.reason && <small className="target-status-reason">后端原因：{group.reason}</small>}
    </div>
  );
}

function RegionEntryRow({ frame, regionId }: { frame: OperationalFrame; regionId: string }) {
  const execution = frame.execution;
  const region = execution?.regions.find((candidate) => candidate.region_id === regionId);
  if (!region) return null;
  const evidence = entryEvidenceForRegion(execution, regionId);
  const probability = evidence?.probability;
  const count = evidence?.confirmation_count;
  const required = evidence?.required_cycles ?? execution?.entry_confirmation_required_cycles;
  const telemetry = scanTelemetryForRegion(region);
  const countText = count == null
    ? "不可用"
    : `${Math.max(0, Math.round(count))}/${required == null ? "?" : Math.max(0, Math.round(required))}`;
  const route = scanRouteProgressPercent(telemetry);
  const coverage = scanCoveragePercent(telemetry);
  const pings = scanPingCount(telemetry);
  const round = telemetry?.scan_round;
  const threshold = telemetry?.scan_completion_threshold;
  const blockers = evidence?.blocking_reasons ?? [];
  return (
    <div
      className="target-status-region"
      data-region-id={regionId}
      data-entry-confirmation={count == null ? undefined : countText}
      data-scan-completed={telemetry?.scan_completed == null ? undefined : String(telemetry.scan_completed)}
    >
      <div className="target-status-region-heading">
        <strong>{regionId}</strong>
        <span>{region.status}</span>
      </div>
      <div className="target-status-facts">
        <span>入区概率 <b>{finite(probability) ? formatPercent(probability * 100) : "不可用"}</b></span>
        <span>确认 <b>{countText}</b></span>
        <span>路线 <b>{formatPercent(route)}</b></span>
      </div>
      <div className="target-status-facts target-status-scan-facts">
        <span>主动覆盖 <b>{formatPercent(coverage)}</b></span>
        <span>source-backed ping <b>{pings == null ? "不可用" : pings}</b></span>
        <span>扫描 <b>{telemetry?.scan_completed == null ? "不可用" : telemetry.scan_completed ? "本轮扫描完成" : "本轮扫描未完成"}</b></span>
      </div>
      <small className="target-status-evidence">
        本轮 {round == null ? "不可用" : `第 ${Math.max(0, Math.round(round))} 轮`}
        {threshold == null ? "" : ` · 完成阈值 ${formatPercent(threshold * 100)}`}
      </small>
      {blockers.length > 0 && <small className="target-status-blocker">阻塞：{blockers.join("；")}</small>}
      {(evidence?.evidence_ids?.length ?? 0) > 0 && <small className="target-status-evidence">证据：{evidence?.evidence_ids?.join("、")}</small>}
    </div>
  );
}

function HandoffRow({ frame }: { frame: OperationalFrame }) {
  const progress = handoffProgress(frame.execution);
  const handoff = frame.execution?.handoff_evidence;
  if (!handoff || !progress) return null;
  const progressText = `${progress.valid}/${progress.required}`;
  return (
    <div className="target-status-handoff" data-handoff-progress={progressText}>
      <div className="target-status-region-heading">
        <strong>接力证据</strong>
        <span>{HANDOFF_STATUS_LABELS[progress.status]}</span>
      </div>
      <div className="target-status-facts">
        <span>前任 <b>{handoff.predecessor_group_id ?? "—"}</b></span>
        <span>后继 <b>{handoff.successor_group_id ?? "—"}</b></span>
        <span>有效观测 <b>{progressText}</b></span>
      </div>
      {progress.missing.length > 0 && (
        <small className="target-status-blocker">尚缺：{progress.missing.join("、")}</small>
      )}
      {(handoff.evidence_ids?.length ?? 0) > 0 && <small className="target-status-evidence">证据：{handoff.evidence_ids?.join("、")}</small>}
    </div>
  );
}

function ReplacementRows({ frame }: { frame: OperationalFrame }) {
  const replacements = frame.execution?.replacements ?? [];
  if (!replacements.length) return null;
  return (
    <div className="target-status-replacements" aria-label="区域替补编组">
      <div className="target-status-region-heading"><strong>替补/几何修订</strong><span>{replacements.length} 条</span></div>
      {replacements.map((replacement) => (
        <div
          className="target-status-replacement"
          key={`${replacement.region_id}:${replacement.target_geometry_revision}`}
          data-replacement-region-id={replacement.region_id}
          data-incoming-group-id={replacement.incoming_group_id}
          data-outgoing-group-id={replacement.outgoing_group_id}
          data-geometry-revision={replacement.target_geometry_revision}
        >
          <span>{replacement.region_id}</span>
          <span className="replacement-flow"><b>OUT</b> {replacement.outgoing_group_id} → <b>IN</b> {replacement.incoming_group_id}</span>
          <span>Geometry v{replacement.source_geometry_revision} → v{replacement.target_geometry_revision}</span>
          {replacement.batch_id && <span>Batch {replacement.batch_id}</span>}
          {replacement.latest_pending_geometry_revision != null && <span>待发布 v{replacement.latest_pending_geometry_revision}</span>}
        </div>
      ))}
    </div>
  );
}

function DedicatedRow({ frame }: { frame: OperationalFrame }) {
  const execution = frame.execution;
  if (!execution) return null;
  const dedicated = execution.tracking_control.mode === "dedicated"
    || execution.task_groups.some((group) => group.lifecycle === "dedicated_track" || group.lifecycle === "dedicated_release_pending");
  if (!dedicated) return null;
  const threshold = execution.tracking_policy.dedicated_release_remaining_mileage_m;
  const triggered = execution.tracking_control.dedicated_release_triggered_at_m;
  const dedicatedGroupIds = new Set(
    execution.task_groups
      .filter((group) => group.lifecycle === "dedicated_track" || group.lifecycle === "dedicated_release_pending")
      .map((group) => group.group_instance_id),
  );
  const remaining = (frame.uuvs ?? [])
    .filter((uuv) => uuv.group_instance_id != null && dedicatedGroupIds.has(uuv.group_instance_id))
    .map((uuv) => uuv.remaining_range_m ?? uuv.endurance_remaining_m)
    .filter((value): value is number => finite(value));
  const minimumRemaining = remaining.length ? Math.min(...remaining) : null;
  const releasePending = execution.task_groups.some((group) => group.lifecycle === "dedicated_release_pending");
  return (
    <div className="target-status-dedicated" data-release-threshold-m={threshold}>
      <div className="target-status-region-heading"><strong>专用跟踪</strong><span>{releasePending ? "DEDICATED_TRACK · RELEASE_PENDING" : execution.tracking_control.mode === "dedicated" ? "DEDICATED_TRACK" : "释放待定"}</span></div>
      <div className="target-status-facts">
        <span>释放阈值 <b>{finite(threshold) ? `${threshold.toFixed(0)} m` : "不可用"}</b></span>
        <span>剩余里程最低 <b>{finite(minimumRemaining) ? `${minimumRemaining.toFixed(0)} m` : "不可用"}</b></span>
        <span>触发里程 <b>{finite(triggered) ? `${triggered.toFixed(0)} m` : "未触发"}</b></span>
        <span>原因 <b>{execution.tracking_control.dedicated_release_reason ?? "—"}</b></span>
      </div>
    </div>
  );
}

function Blockers({ frame, targetId }: { frame: OperationalFrame; targetId: string }) {
  const blockers = structuredBlockingReasons(frame.execution, undefined);
  if (!blockers.length) return null;
  return (
    <div className="target-status-blockers" data-blocking-reason-count={blockers.length}>
      <strong>当前阻塞</strong>
      <span>{blockers.join("；")}</span>
      <small>作用域：{targetId}</small>
    </div>
  );
}

function TargetCard({ frame, target }: { frame: OperationalFrame; target: TargetEstimateView }) {
  const status = estimatePresentationStatus(frame, target);
  const groups = targetGroups(frame, target.target_id);
  const regions = frame.execution?.regions.filter((region) => region.target_id === target.target_id) ?? [];
  const freshnessFacts = formatEstimateFreshness(frame, target);
  const expired = status === "expired";
  return (
    <article
      className={`target-status-card estimate-${status}`}
      data-target-status={status}
      data-estimate-status={status}
      data-target-id={target.target_id}
    >
      <header className="target-status-card-heading">
        <div>
          <span className="eyebrow">TARGET STATUS / OPERATIONAL FRAME</span>
          <strong>{displayTargetName(target.target_id)} · {target.target_id}</strong>
        </div>
        <span className={`target-status-badge status-${status}`}>{ESTIMATE_STATUS_LABELS[status]}</span>
      </header>
      <div className="target-status-frame-facts">
        <span>Frame #{frame.frame_id}</span><span>仿真 {formatSimSeconds(frame.sim_time_s)}</span><span>方案 #{frame.plan_version}</span>
      </div>
      <div className={`target-status-freshness ${expired ? "expired" : ""}`} role={expired ? "alert" : undefined}>
        <strong>{expired ? "目标估计已过期 · 等待刷新" : `目标估计：${ESTIMATE_STATUS_SHORT_LABELS[status]}`}</strong>
        <span>{freshnessFacts.length ? freshnessFacts.join(" · ") : "未提供估计新鲜度字段，状态不可判定"}</span>
        {expired && frame.run_phase === "running" && <small>物理仿真仍在运行，地图上的旧位置/走廊已降级显示，不代表实时真值。</small>}
      </div>

      <section className="target-status-section" aria-label={`${target.target_id} 编组生命周期`}>
        <div className="target-status-section-heading"><span>编组生命周期</span><small>{groups.length ? `${groups.length} 个实例` : "不可用"}</small></div>
        {groups.length ? groups.map((group) => <GroupStatusRow frame={frame} group={group} key={group.group_instance_id} />) : <div className="target-status-unavailable">后端未提供 task_groups</div>}
      </section>
      {regions.length > 0 && (
        <section className="target-status-section" aria-label={`${target.target_id} 区域扫描证据`}>
          <div className="target-status-section-heading"><span>区域扫描与入区证据</span><small>不以路线完成代替扫描完成</small></div>
          {regions.map((region) => <RegionEntryRow frame={frame} regionId={region.region_id} key={region.region_id} />)}
        </section>
      )}
      <HandoffRow frame={frame} />
      <ReplacementRows frame={frame} />
      <DedicatedRow frame={frame} />
      <Blockers frame={frame} targetId={target.target_id} />
    </article>
  );
}

export default function TargetStatusBoard({ frame }: TargetStatusBoardProps) {
  const targets = frame.target_estimates ?? [];
  return (
    <section className="target-status-board" aria-label="每目标运行状态" data-target-status-board>
      <div className="target-status-board-heading">
        <span className="eyebrow">PER-TARGET / EVIDENCE</span>
        <strong>目标状态与接力</strong>
        <small>{targets.length ? `${targets.length} 个估计` : "估计不可用"}</small>
      </div>
      {targets.length ? targets.map((target) => <TargetCard frame={frame} target={target} key={target.target_id} />) : (
        <div className="target-status-unavailable" data-estimate-status="unavailable">当前帧未提供目标估计，不能推断目标状态。</div>
      )}
    </section>
  );
}
