import { BrainCircuit } from "lucide-react";
import type {
  TargetEstimateView,
  WorldModelEventView,
  WorldModelHorizon,
} from "../types/frames";
import { displayTargetName } from "../utils/presentation";

const EVENT_LABEL: Record<string, string> = {
  target_turn_left: "目标左转",
  target_turn_right: "目标右转",
  high_speed_escape: "高速逃逸",
  area_exit_risk: "离开任务区",
  geometry_degradation: "观测角度变差",
  track_loss_risk: "失去跟踪风险",
  decoy_or_new_contact_ambiguity: "诱饵 / 新目标混淆",
  uuv_coverage_gap: "UUV 覆盖缺口",
  target_abnormal_stop: "目标异常低速",
};

interface WorldModelPanelProps {
  targets: TargetEstimateView[];
}

export default function WorldModelPanel({ targets }: WorldModelPanelProps) {
  const forecasts = targets.flatMap((target) =>
    target.world_model ? [{ target, forecast: target.world_model }] : [],
  );

  if (forecasts.length === 0) return null;

  return (
    <section className="world-model-panel" aria-label="规则世界模型预测">
      <div className="world-model-heading">
        <span>
          <BrainCircuit size={14} aria-hidden="true" />
          未来事件推演
        </span>
        <em>只读</em>
      </div>
      <p className="world-model-disclaimer">
        根据 IMM（运动状态估计）和 B-spline（未来轨迹）推演，只作提示，不直接控制 UUV。
      </p>
      <div className="world-model-targets">
        {forecasts.map(({ target, forecast }) => (
          <article className="world-model-target" key={target.target_id}>
            <div className="world-model-target-heading">
              <strong>{displayTargetName(target.target_id)}</strong>
              <span className={`data-${forecast.data_status}`}>
                {{ready: "输入完整", degraded: "降级推演", expired: "数据已过期", unavailable: "暂不可用"}[forecast.data_status]}
              </span>
            </div>
            {forecast.events.length === 0 && (
              <p className="world-model-clear">{["expired", "unavailable"].includes(forecast.data_status)
                ? "当前输入不足以判断未来事件" : "当前规则未发现明显未来事件"}</p>
            )}
            <div className="world-model-horizons">
              {forecast.horizons.map((horizon) => {
                const events = forecast.events.filter(
                  (event) => event.horizon === horizon.name,
                );
                return (
                  <section
                    className={`world-model-horizon ${horizon.covered ? "covered" : "uncovered"}`}
                    key={horizon.name}
                    aria-label={`${horizon.name} 预测窗口`}
                  >
                    <div className="world-model-horizon-heading">
                      <strong>{horizon.name}</strong>
                      <span>{formatRange(horizon.start_offset_s, horizon.end_offset_s)}</span>
                      <small>{horizon.covered ? `${horizon.sample_count} 点` : "无轨迹"}</small>
                    </div>
                    {events.length === 0 ? (
                      <p>{horizon.covered ? "未触发事件" : "预测未覆盖"}</p>
                    ) : (
                      <div className="world-model-events">
                        {events.map((event) => (
                          <EventCard event={event} key={event.event_id} />
                        ))}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
            {forecast.warnings.length > 0 && (
              <details className="world-model-warnings">
                <summary>{`${forecast.warnings.length} 条输入说明`}</summary>
                <ul>
                  {forecast.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </details>
            )}
            <div className="world-model-provenance">
              <span>{`IMM ${leadingImmModel(forecast.imm_model_probabilities)}`}</span>
              <span title={forecast.source_prediction_id}>已接受的预测轨迹</span>
              {forecast.source_plan_revision != null && (
                <span>{`方案 v${forecast.source_plan_revision}`}</span>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function EventCard({ event }: { event: WorldModelEventView }) {
  return (
    <article
      className={`world-model-event level-${event.level}`}
      data-event-type={event.event_type}
    >
      <div>
        <strong>{EVENT_LABEL[event.event_type] ?? event.event_type}</strong>
        <span>{formatOffset(event.time_to_event_s)}</span>
      </div>
      <p>{event.summary}</p>
      <footer>
        <span>{`规则置信度 ${Math.round(event.confidence * 100)}%`}</span>
        <small>{`${Math.round(event.predicted_position.x)}, ${Math.round(event.predicted_position.y)} m`}</small>
      </footer>
    </article>
  );
}

function formatOffset(seconds: number) {
  if (seconds < 60) return `T+${Math.round(seconds)} 秒`;
  return `T+${formatMinutes(seconds)} 分钟`;
}

function formatRange(startSeconds: number, endSeconds: number) {
  return `${formatMinutes(startSeconds)}–${formatMinutes(endSeconds)} 分钟`;
}

function formatMinutes(seconds: number) {
  const minutes = seconds / 60;
  return Number.isInteger(minutes) ? String(minutes) : minutes.toFixed(1);
}

function leadingImmModel(probabilities: Record<string, number>) {
  const leading = Object.entries(probabilities).sort(
    ([leftName, left], [rightName, right]) => right - left || leftName.localeCompare(rightName),
  )[0];
  if (!leading) return "—";
  return `${leading[0]} ${Math.round(leading[1] * 100)}%`;
}

export function eventsByHorizon(
  events: WorldModelEventView[],
): Record<WorldModelHorizon, WorldModelEventView[]> {
  return {
    H1: events.filter((event) => event.horizon === "H1"),
    H2: events.filter((event) => event.horizon === "H2"),
    H3: events.filter((event) => event.horizon === "H3"),
    H4: events.filter((event) => event.horizon === "H4"),
  };
}
