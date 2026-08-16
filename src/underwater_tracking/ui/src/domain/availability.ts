import type { UUVView } from "../types/frames";

/** Browser mirror of the domain deployability contract. */
export function isDeployableUuv(uuv: UUVView): boolean {
  return uuv.status !== "failed" && uuv.deployment_state === "deployed";
}
