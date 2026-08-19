import type { UUVView } from "../../types/frames";

export interface SonarBadgesProps {
  uuvs: UUVView[];
  missionModes?: Record<string, string>;
}

export default function SonarBadges({ uuvs, missionModes = {} }: SonarBadgesProps) {
  const active = uuvs.filter((uuv) => missionModes[uuv.uuv_id] === "ACTIVE_SCAN" || (!missionModes[uuv.uuv_id] && uuv.sensor_mode === "active"));
  const passive = uuvs.filter((uuv) => missionModes[uuv.uuv_id] === "PASSIVE_TRACK");
  const reserved = uuvs.filter((uuv) => uuv.reserved);
  return (
    <div className="sonar-badges" aria-label="主动声纳状态">
      {active.map((uuv) => <span key={`active-${uuv.uuv_id}`} className="badge badge-active">{uuv.uuv_id} 主动</span>)}
      {passive.length > 0 && <span className="badge badge-passive">{passive.length} 艇被动跟踪</span>}
      {reserved.map((uuv) => <span key={`reserved-${uuv.uuv_id}`} className="badge badge-reserved">{uuv.uuv_id} 指派</span>)}
      {active.length === 0 && passive.length === 0 && reserved.length === 0 && <span className="badge badge-passive">全部被动</span>}
    </div>
  );
}
