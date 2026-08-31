import type { OperationalFrame, UUVView } from "./frames";

type IsRequired<T, K extends keyof T> = {} extends Pick<T, K> ? false : true;
type Assert<T extends true> = T;

export type UUVDeploymentStateMustBeRequired = Assert<IsRequired<UUVView, "deployment_state">>;
export type OperationalFrameCarrierMustBeRequired = Assert<IsRequired<OperationalFrame, "carrier">>;
