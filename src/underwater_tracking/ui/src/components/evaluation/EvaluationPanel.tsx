import { useEffect, useState } from "react";

interface EvaluationTarget {
  target_id: string;
  position_xy: [number, number];
  intent_label: string;
}

interface EvaluationFrame {
  frame_id: number;
  sim_time_s: number;
  targets: EvaluationTarget[];
}

interface EvaluationPanelProps {
  enabled: boolean;
  simTimeS: number;
}

export default function EvaluationPanel({ enabled, simTimeS }: EvaluationPanelProps) {
  const [frame, setFrame] = useState<EvaluationFrame | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!enabled) return undefined;
    const end = Math.max(0, simTimeS);
    void fetch(`/api/evaluation/frames?start_s=${Math.max(0, end - 60)}&end_s=${end}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = (await response.json()) as { frames?: EvaluationFrame[] };
        setFrame(payload.frames?.at(-1) ?? null);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "evaluation unavailable"));
    return undefined;
  }, [enabled, simTimeS]);

  if (!enabled) return null;
  return <aside className="evaluation-panel" aria-label="评估模式">
    <div className="evaluation-banner"><span>EVALUATION / TRUTH ENABLED</span><small>仅用于离线评估，不参与在线决策</small></div>
    {error ? <p>{error}</p> : frame ? <div className="evaluation-targets">{frame.targets.map((target) => <div key={target.target_id}><strong>{target.target_id}</strong><span>{target.position_xy[0].toFixed(1)}, {target.position_xy[1].toFixed(1)} m</span><small>{target.intent_label}</small></div>)}</div> : <p>等待评估帧</p>}
  </aside>;
}
