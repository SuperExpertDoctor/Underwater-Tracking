import type { CarrierStatus, OperationalFrame } from "../types/frames";

const CARRIER_STATUS_LABELS: Record<CarrierStatus, string> = {
  standby: "待命",
  transit: "航渡",
  deploying: "发送中",
  recovering: "回收中",
};

export default function CarrierStatusPanel({ frame }: { frame: OperationalFrame | null }) {
  const carriers = frame?.carriers?.length
    ? frame.carriers
    : frame?.carrier
      ? [frame.carrier]
      : [];
  const carrier = frame?.carrier ?? carriers[0] ?? null;

  if (!carrier) {
    return <section className="carrier-status-panel" aria-label="载体舰 / 发送回收">
      <div className="section-heading"><span>载体舰 / 发送回收</span></div>
      <p className="carrier-status-empty">等待载体态势</p>
    </section>;
  }

  const failed = frame?.uuvs.filter((uuv) => uuv.deployment_state === "failed").length ?? 0;
  const roleLabel = carrier.role === "mother_ship" ? "母舰" : "航母";
  const fleetRoleSummary = `${carriers.filter((item) => item.role !== "mother_ship").length}航母 · ${carriers.filter((item) => item.role === "mother_ship").length}母舰`;
  const counts = [
    ["在舰", carriers.reduce((total, item) => total + item.onboard_uuv_ids.length, 0)],
    ["已部署", carriers.reduce((total, item) => total + item.deployed_uuv_ids.length, 0)],
    ["回收", carriers.reduce((total, item) => total + item.returning_uuv_ids.length, 0)],
    ["故障", failed],
  ];

  return <section className="carrier-status-panel" aria-label="载体舰 / 发送回收">
    <div className="section-heading">
      <span>载体舰 / 发送回收</span>
      <small>{fleetRoleSummary} · {CARRIER_STATUS_LABELS[carrier.status]}</small>
    </div>
    <div className="carrier-identity"><strong>{roleLabel} · {carrier.carrier_id}</strong><span>{CARRIER_STATUS_LABELS[carrier.status]}</span></div>
    <div className="carrier-counts" aria-label="发送回收统计">
      {counts.map(([label, count]) => <div key={label}><b>{`${label} ${count}`}</b></div>)}
    </div>
    <div className="carrier-returning">
      <span>返航艇</span>
      {carrier.returning_uuv_ids.length ? <ul>{carrier.returning_uuv_ids.map((uuvId) => <li key={uuvId}>{uuvId}</li>)}</ul> : <small>暂无</small>}
    </div>
  </section>;
}
