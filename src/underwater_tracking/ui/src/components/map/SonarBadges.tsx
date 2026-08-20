import type { UUVView } from "../../types/frames";

export interface SonarBadgesProps {
  uuvs: UUVView[];
}

export default function SonarBadges({ uuvs }: SonarBadgesProps) {
  const active = uuvs.filter((uuv) => uuv.sensor_mode === "active");
  const reserved = uuvs.filter((uuv) => uuv.reserved);
  return (
    <div className="sonar-badges" aria-label="主动声纳状态">
      {active.map((uuv) => <span key={`active-${uuv.uuv_id}`} className="badge badge-active">{uuv.uuv_id} 主动</span>)}
      {reserved.map((uuv) => <span key={`reserved-${uuv.uuv_id}`} className="badge badge-reserved">{uuv.uuv_id} 指派</span>)}
      {active.length === 0 && reserved.length === 0 && <span className="badge badge-passive">全部被动</span>}
    </div>
  );
}
