import type { CarrierMissionView, CarrierStatus, OperationalFrame } from "../types/frames";

const CARRIER_STATUS_LABELS: Record<CarrierStatus, string> = {
  standby: "待命",
  transit: "航渡",
  deploying: "发送中",
  recovering: "回收中",
};

const ROUTE_STATUS_LABELS: Record<CarrierMissionView["route_status"], string> = {
  TO_DEPLOY: "待部署",
  DEPLOYING: "部署中",
  EN_ROUTE_NEXT_DEPLOY: "前往下一站",
  RETURNING_TO_FLEET: "返航编队",
  RECOVERING: "回收中",
  COMPLETE: "已完成",
  FAILED: "任务失败",
};

const MISSION_TYPE_LABELS: Record<CarrierMissionView["mission_type"], string> = {
  DEPLOY: "部署",
  RECOVER: "回收",
  DEPLOY_AND_RECOVER: "部署 + 回收",
};

function UuvInventory({ mission }: { mission: CarrierMissionView }) {
  const counts = [
    ["在舰", mission.onboard_uuv_ids.length],
    ["待命", mission.ready_uuv_ids.length],
    ["预留", mission.reserved_uuv_ids.length],
    ["可回收", mission.recoverable_uuv_ids.length],
  ];
  return <div className="carrier-counts" aria-label="UUV 载荷库存">
    {counts.map(([label, count]) => <div key={label}><b>{`${label} ${count}`}</b></div>)}
  </div>;
}

function CarrierMissionCard({ mission }: { mission: CarrierMissionView }) {
  return <article className="carrier-mission-card" aria-label={`${mission.carrier_id} 航母任务`}>
    <div className="carrier-identity">
      <strong>{mission.carrier_id}</strong>
      <span>{ROUTE_STATUS_LABELS[mission.route_status]}</span>
    </div>
    <div className="carrier-mission-facts">
      <span>任务 <b>{MISSION_TYPE_LABELS[mission.mission_type]}</b></span>
      <span>母港 <b>{mission.home_battle_group_id}</b></span>
      <span>航点 <b>{mission.stop_ids.length}</b></span>
    </div>
    <UuvInventory mission={mission} />
    <div className="carrier-route-stops">
      <span>任务航点</span>
      {mission.stop_ids.length ? <ol>{mission.stop_ids.map((stopId) => <li key={stopId}>{stopId}</li>)}</ol> : <small>暂无部署 / 回收航点</small>}
      <small className={mission.route.length >= 2 && mission.route[0].x === mission.route.at(-1)?.x && mission.route[0].y === mission.route.at(-1)?.y ? "route-home-ok" : "route-home-warning"}>
        {mission.route.length >= 2 && mission.route[0].x === mission.route.at(-1)?.x && mission.route[0].y === mission.route.at(-1)?.y ? "航线闭合返回母港" : "待验证母港返航"}
      </small>
    </div>
  </article>;
}

export default function CarrierStatusPanel({ frame }: { frame: OperationalFrame | null }) {
  const missions = frame?.carrier_missions ?? [];
  const carrier = frame?.carrier;

  if (missions.length) {
    return <section className="carrier-status-panel" aria-label="载体舰 / 发送回收">
      <div className="section-heading"><span>载体舰 / 发送回收</span><small>{missions.length} 项任务</small></div>
      <div className="carrier-mission-list">{missions.map((mission) => <CarrierMissionCard key={`${mission.carrier_id}-${mission.mission_type}`} mission={mission} />)}</div>
    </section>;
  }

  if (!carrier) {
    return <section className="carrier-status-panel" aria-label="载体舰 / 发送回收">
      <div className="section-heading"><span>载体舰 / 发送回收</span></div>
      <p className="carrier-status-empty">等待载体态势</p>
    </section>;
  }

  const failed = frame.uuvs.filter((uuv) => uuv.deployment_state === "failed").length;
  const counts = [
    ["在舰", carrier.onboard_uuv_ids.length],
    ["已部署", carrier.deployed_uuv_ids.length],
    ["回收", carrier.returning_uuv_ids.length],
    ["故障", failed],
  ];

  return <section className="carrier-status-panel" aria-label="载体舰 / 发送回收">
    <div className="section-heading">
      <span>载体舰 / 发送回收</span>
      <small>{CARRIER_STATUS_LABELS[carrier.status]}</small>
    </div>
    <div className="carrier-identity"><strong>{carrier.carrier_id}</strong><span>{CARRIER_STATUS_LABELS[carrier.status]}</span></div>
    <div className="carrier-counts" aria-label="发送回收统计">
      {counts.map(([label, count]) => <div key={label}><b>{`${label} ${count}`}</b></div>)}
    </div>
    <div className="carrier-returning">
      <span>返航艇</span>
      {carrier.returning_uuv_ids.length ? <ul>{carrier.returning_uuv_ids.map((uuvId) => <li key={uuvId}>{uuvId}</li>)}</ul> : <small>暂无</small>}
    </div>
  </section>;
}
