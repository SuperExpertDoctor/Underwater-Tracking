import { useState } from "react";
import type { TargetEstimateView, UUVView } from "../../types/frames";

export interface AssignmentPanelProps {
  targets: TargetEstimateView[];
  uuvs: UUVView[];
  onAssign: (uuvIds: string[], targetId: string) => void;
}

export default function AssignmentPanel({ targets, uuvs, onAssign }: AssignmentPanelProps) {
  const [targetId, setTargetId] = useState(targets[0]?.target_id ?? "");
  const [selected, setSelected] = useState<string[]>([]);
  const available = uuvs.filter((uuv) => !uuv.reserved && uuv.status !== "failed");

  const toggle = (uuvId: string) => {
    setSelected((current) => current.includes(uuvId)
      ? current.filter((id) => id !== uuvId)
      : [...current, uuvId]);
  };

  return (
    <section className="assignment-panel" aria-label="人为指派模式">
      <div className="section-heading">
        <span>人为指派</span>
        <small>已指派 {uuvs.filter((uuv) => uuv.reserved).length} 艇</small>
      </div>
      <label className="field">
        <span>跟踪目标</span>
        <select value={targetId} onChange={(event) => setTargetId(event.target.value)} aria-label="选择跟踪目标">
          {targets.map((target) => <option key={target.target_id} value={target.target_id}>{target.target_id}</option>)}
        </select>
      </label>
      <div className="uuv-list compact" role="group" aria-label="可选 UUV">
        {available.map((uuv) => (
          <label key={uuv.uuv_id} className="uuv-check">
            <input type="checkbox" checked={selected.includes(uuv.uuv_id)} onChange={() => toggle(uuv.uuv_id)} />
            <span>{uuv.uuv_id}</span>
            <small>{uuv.sensor_mode === "active" ? "主动声纳" : "被动声纳"}</small>
          </label>
        ))}
        {available.length === 0 && <p>无可用 UUV</p>}
      </div>
      <button
        className="primary-btn"
        disabled={selected.length === 0 || !targetId}
        onClick={() => {
          onAssign([...selected].sort(), targetId);
          setSelected([]);
        }}
      >
        指派跟踪
      </button>
    </section>
  );
}
