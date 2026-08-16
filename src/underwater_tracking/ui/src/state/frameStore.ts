import type { OperationalFrame, StreamMessage } from "../types/frames";

export const MAX_REPLAY_FRAMES = 600;

/** Compare the monotonic identity carried by operational frames. */
export function frameOrder(left: OperationalFrame, right: OperationalFrame): number {
  if (left.sim_time_s !== right.sim_time_s) return left.sim_time_s - right.sim_time_s;
  return left.frame_id - right.frame_id;
}

export function acceptLiveFrame(
  current: OperationalFrame | null,
  next: OperationalFrame,
): { accepted: boolean; frame: OperationalFrame } {
  if (current && frameOrder(next, current) <= 0) {
    return { accepted: false, frame: current };
  }
  return { accepted: true, frame: next };
}

/** Merge a replay page while keeping a hard memory ceiling and chronological order. */
export function mergeReplayFrames(
  current: OperationalFrame[],
  incoming: OperationalFrame[],
  limit = MAX_REPLAY_FRAMES,
): OperationalFrame[] {
  const byIdentity = new Map<string, OperationalFrame>();
  [...current, ...incoming].forEach((frame) => {
    byIdentity.set(`${frame.sim_time_s}:${frame.frame_id}`, frame);
  });
  return [...byIdentity.values()]
    .sort(frameOrder)
    .slice(-Math.max(1, limit));
}

export function isHeartbeat(message: StreamMessage): message is { type: "heartbeat"; sim_time_s: number | null } {
  return typeof message === "object" && "type" in message && message.type === "heartbeat";
}
