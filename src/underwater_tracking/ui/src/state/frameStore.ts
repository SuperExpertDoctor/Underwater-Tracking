import type { OperationalFrame, StreamMessage } from "../types/frames";

/** Covers the complete 8-hour acceptance run at the current five-second step. */
export const MAX_REPLAY_FRAMES = 10_000;

export interface FrameStoreState {
  frame: OperationalFrame | null;
  needsSnapshot: boolean;
}

export type FrameStoreAction =
  | { type: "frame"; frame: OperationalFrame }
  | { type: "snapshot"; frame: OperationalFrame }
  | { type: "clear" };

export interface FrameStoreTransition {
  accepted: boolean;
  requestSnapshot: boolean;
  state: FrameStoreState;
}

export function createFrameStoreState(
  frame: OperationalFrame | null = null,
): FrameStoreState {
  return { frame, needsSnapshot: false };
}

/** Compare the monotonic identity carried by operational frames. */
export function frameOrder(left: OperationalFrame, right: OperationalFrame): number {
  if (left.frame_id !== right.frame_id) return left.frame_id - right.frame_id;
  return left.sim_time_s - right.sim_time_s;
}

export function frameGap(
  current: OperationalFrame | null,
  next: OperationalFrame,
): boolean {
  return current !== null && next.frame_id > current.frame_id + 1;
}

/** Apply one live or HTTP-recovered frame without allowing mixed revisions. */
export function reduceOperationalFrame(
  state: FrameStoreState,
  action: FrameStoreAction,
): FrameStoreTransition {
  if (action.type === "clear") {
    return { accepted: false, requestSnapshot: false, state: createFrameStoreState() };
  }
  if (state.frame && frameOrder(action.frame, state.frame) <= 0) {
    return { accepted: false, requestSnapshot: false, state };
  }
  const requestSnapshot = action.type === "frame" && frameGap(state.frame, action.frame);
  return {
    accepted: true,
    requestSnapshot,
    state: {
      frame: action.frame,
      needsSnapshot: action.type === "snapshot"
        ? false
        : state.needsSnapshot || requestSnapshot,
    },
  };
}

export function acceptLiveFrame(
  current: OperationalFrame | null,
  next: OperationalFrame,
): { accepted: boolean; frame: OperationalFrame } {
  const transition = reduceOperationalFrame(
    createFrameStoreState(current),
    { type: "frame", frame: next },
  );
  return {
    accepted: transition.accepted,
    frame: transition.state.frame ?? next,
  };
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
