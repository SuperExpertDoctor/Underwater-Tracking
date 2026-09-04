import type {
  ExecutionView,
  OperationalFrame,
  TaskGroupInstanceView,
  UUVView,
} from "../types/frames";

export interface ExecutionCounts {
  visibleUuvs: number;
  enteringGroups: number;
  exitingGroups: number;
  activeScanGroups: number;
  passiveTrackGroups: number;
}

export function groupInstanceId(group: TaskGroupInstanceView): string {
  return group.group_instance_id;
}

export function groupForUuv(
  frame: OperationalFrame,
  uuvId: string,
): TaskGroupInstanceView | undefined {
  const uuv = (frame.uuvs ?? []).find((candidate) => candidate.uuv_id === uuvId);
  if (!uuv || !frame.execution) return undefined;
  return frame.execution.task_groups.find(
    (group) => group.group_instance_id === uuv.group_instance_id,
  );
}

export function groupsByRegionSlot(
  frame: OperationalFrame,
): Map<string, TaskGroupInstanceView[]> {
  const groups = new Map<string, TaskGroupInstanceView[]>();
  frame.execution?.task_groups.forEach((group) => {
    const slotGroups = groups.get(group.region_id) ?? [];
    slotGroups.push(group);
    groups.set(group.region_id, slotGroups);
  });
  return groups;
}

export function ownerGroup(
  frame: OperationalFrame,
): TaskGroupInstanceView | undefined {
  const execution = frame.execution;
  if (!execution) return undefined;
  const ownerId = execution.tracking_control?.tracking_owner_group_id;
  if (ownerId) {
    const explicitOwner = execution.task_groups.find(
      (group) => group.group_instance_id === ownerId,
    );
    if (explicitOwner) return explicitOwner;
  }
  return execution.task_groups.find((group) => group.ownership_status === "owner");
}

function runtimeVisibleGroupIds(execution: ExecutionView): Set<string> {
  return new Set(
    execution.task_groups
      .filter((group) => group.lifecycle !== "disappeared")
      .map((group) => group.group_instance_id),
  );
}

/**
 * Returns the UUV entities represented by the authoritative execution view.
 * Runtime frames use deployment-aware instance IDs so two generations in one
 * region remain independently renderable during replacement.
 */
export function visibleExecutionUuvs(frame: OperationalFrame): UUVView[] {
  const execution = frame.execution;
  if (!execution) return [];

  const visibleGroupIds = runtimeVisibleGroupIds(execution);
  return (frame.uuvs ?? []).filter(
    (uuv) => uuv.physically_exposed
      && Boolean(uuv.group_instance_id)
      && visibleGroupIds.has(uuv.group_instance_id!),
  );
}

export function executionCounts(frame: OperationalFrame): ExecutionCounts {
  const groups = frame.execution?.task_groups ?? [];
  return {
    visibleUuvs: visibleExecutionUuvs(frame).length,
    enteringGroups: groups.filter(
      (group) => group.lifecycle === "entering",
    ).length,
    exitingGroups: groups.filter(
      (group) => group.lifecycle === "exiting",
    ).length,
    activeScanGroups: groups.filter(
      (group) => group.lifecycle === "active_scan",
    ).length,
    passiveTrackGroups: groups.filter(
      (group) => group.lifecycle === "passive_track",
    ).length,
  };
}
