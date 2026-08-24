"""Deterministic market allocation for heterogeneous UUV task slots.

This module implements the reproducible core of a distributed market/auction
allocator: a task exposes role slots, each UUV submits a utility bid, and the
auctioneer resolves conflicts in rounds while preserving hard capability and
assignment constraints. The bid combines information value, temporal
urgency, travel cost, energy margin, and continuity. It is intentionally
interpretable so every award can be audited from public planning inputs.

The design follows the market-based multi-AUV allocation pattern described by
You, Pang, and Jiang (Journal of Marine Science and Application, 2005,
doi:10.1007/s11804-005-0026-z), with continuity and multi-task resource
pressure used as explicit bid terms. It is not a claim to reproduce a paper's
complete vehicle-specific implementation; it reproduces the transferable
allocation logic that is executable and testable in this repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite

from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class AuctionTask:
    """One candidate region represented as role slots in the market."""

    task_id: str
    center_xy: tuple[float, float]
    entry_s: int
    exit_s: int
    probability: float
    priority: float
    active_slots: int = 0
    passive_slots: int = 0
    reserve_slots: int = 0
    continuity_uuv_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("auction task_id must not be empty")
        if self.entry_s < 0 or self.exit_s <= self.entry_s:
            raise ValueError("auction task window must be positive")
        if not all(isfinite(value) for value in self.center_xy):
            raise ValueError("auction task center must be finite")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("auction task probability must be in [0, 1]")
        if self.priority < 0.0 or not isfinite(self.priority):
            raise ValueError("auction task priority must be finite and non-negative")
        if any(
            count < 0
            for count in (self.active_slots, self.passive_slots, self.reserve_slots)
        ):
            raise ValueError("auction task slot counts must be non-negative")
        if len(self.continuity_uuv_ids) != len(set(self.continuity_uuv_ids)):
            raise ValueError("auction task continuity IDs must be unique")


@dataclass(frozen=True, slots=True)
class AuctionUUV:
    """Public resource facts a UUV may use to submit a bid."""

    uuv_id: str
    position_xy: tuple[float, float]
    speed_mps: float
    energy_fraction: float
    active_capable: bool
    carrier_id: str

    def __post_init__(self) -> None:
        if not self.uuv_id or not self.carrier_id:
            raise ValueError("auction UUV identifiers must not be empty")
        if not all(isfinite(value) for value in self.position_xy):
            raise ValueError("auction UUV position must be finite")
        if self.speed_mps <= 0.0 or not isfinite(self.speed_mps):
            raise ValueError("auction UUV speed must be positive and finite")
        if not 0.0 <= self.energy_fraction <= 1.0:
            raise ValueError("auction UUV energy must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AuctionAward:
    """Winning role members for one task."""

    task_id: str
    active_uuv_ids: tuple[str, ...] = ()
    passive_uuv_ids: tuple[str, ...] = ()
    reserve_uuv_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AuctionAllocation:
    """Complete deterministic result, including unmet slots for degradation."""

    awards: tuple[AuctionAward, ...]
    unfilled_slots: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True, slots=True)
class _Slot:
    task: AuctionTask
    role: str
    index: int


def market_allocate(
    tasks: Sequence[AuctionTask],
    uuvs: Sequence[AuctionUUV],
    *,
    locked_uuv_ids_by_task: Mapping[str, Sequence[str]] | None = None,
) -> AuctionAllocation:
    """Build bids and resolve the market with a deterministic Hungarian solve.

    A UUV may win at most one role slot. Active slots are capability-gated;
    locked members are awarded first and remain hard assignments. The remaining
    bid matrix is solved globally, so a locally attractive award cannot consume
    a resource needed by a higher-value role elsewhere. Dummy rows and columns
    represent explicit unfilled capacity rather than fabricating assignments.
    """
    ordered_tasks = tuple(
        sorted(tasks, key=lambda task: (task.entry_s, -task.probability, task.task_id))
    )
    if len({task.task_id for task in ordered_tasks}) != len(ordered_tasks):
        raise ValueError("auction task IDs must be unique")
    ordered_uuvs = tuple(sorted(uuvs, key=lambda uuv: uuv.uuv_id))
    if len({uuv.uuv_id for uuv in ordered_uuvs}) != len(ordered_uuvs):
        raise ValueError("auction UUV IDs must be unique")
    by_uuv_id = {uuv.uuv_id: uuv for uuv in ordered_uuvs}
    task_by_id = {task.task_id: task for task in ordered_tasks}
    locks = {
        task_id: tuple(dict.fromkeys(str(uuv_id) for uuv_id in uuv_ids))
        for task_id, uuv_ids in (locked_uuv_ids_by_task or {}).items()
    }
    if set(locks) - set(task_by_id):
        raise ValueError("locked auction task is unknown")
    unknown_locked = {
        uuv_id for uuv_ids in locks.values() for uuv_id in uuv_ids
    } - set(by_uuv_id)
    if unknown_locked:
        raise ValueError(f"locked auction UUV is unavailable: {sorted(unknown_locked)}")
    all_locked = [uuv_id for uuv_ids in locks.values() for uuv_id in uuv_ids]
    if len(all_locked) != len(set(all_locked)):
        raise ValueError("a locked auction UUV cannot serve two tasks")

    slots = [slot for task in ordered_tasks for slot in _task_slots(task)]
    assignment: dict[tuple[str, str], list[str]] = {}
    assigned: set[str] = set()
    filled_slot_keys: set[tuple[str, str, int]] = set()
    unfilled: list[tuple[str, str, int]] = []

    for task in ordered_tasks:
        locked = list(locks.get(task.task_id, ()))
        for uuv_id in locked:
            role_slot = next(
                (
                    slot
                    for slot in slots
                    if slot.task.task_id == task.task_id
                    and (slot.role != "active" or by_uuv_id[uuv_id].active_capable)
                ),
                None,
            )
            if role_slot is None:
                raise ValueError(f"locked auction UUV cannot fill task {task.task_id!r}")
            assignment.setdefault((task.task_id, role_slot.role), []).append(uuv_id)
            assigned.add(uuv_id)
            filled_slot_keys.add((role_slot.task.task_id, role_slot.role, role_slot.index))
            slots.remove(role_slot)

    remaining_uuvs = tuple(uuv for uuv in ordered_uuvs if uuv.uuv_id not in assigned)
    if slots and remaining_uuvs:
        row_count = len(remaining_uuvs) + len(slots)
        column_count = len(slots) + len(remaining_uuvs)
        costs = [[0.0 for _ in range(column_count)] for _ in range(row_count)]
        for row, uuv in enumerate(remaining_uuvs):
            for column, slot in enumerate(slots):
                bid = _bid(uuv, slot)
                feasible = slot.role != "active" or uuv.active_capable
                # Dummy rows/columns have zero cost. A negative real cost is
                # therefore selected only when its bid is strictly positive.
                costs[row][column] = (
                    -bid + _tie_break(row, column, row_count, column_count)
                    if feasible
                    else float("inf")
                )
        row_indices, column_indices = linear_sum_assignment(costs)
        for row, column in zip(row_indices, column_indices, strict=True):
            if row >= len(remaining_uuvs) or column >= len(slots):
                continue
            uuv = remaining_uuvs[row]
            slot = slots[column]
            if _bid(uuv, slot) <= 0.0:
                continue
            assignment.setdefault((slot.task.task_id, slot.role), []).append(uuv.uuv_id)
            assigned.add(uuv.uuv_id)
            filled_slot_keys.add((slot.task.task_id, slot.role, slot.index))

    unfilled.extend(
        (slot.task.task_id, slot.role, slot.index)
        for task in ordered_tasks
        for slot in _task_slots(task)
        if (slot.task.task_id, slot.role, slot.index) not in filled_slot_keys
    )
    awards = tuple(
        AuctionAward(
            task_id=task.task_id,
            active_uuv_ids=tuple(assignment.get((task.task_id, "active"), ())),
            passive_uuv_ids=tuple(assignment.get((task.task_id, "passive"), ())),
            reserve_uuv_ids=tuple(assignment.get((task.task_id, "reserve"), ())),
        )
        for task in ordered_tasks
        if any(
            assignment.get((task.task_id, role))
            for role in ("active", "passive", "reserve")
        )
    )
    return AuctionAllocation(
        awards=awards,
        unfilled_slots=tuple(sorted(unfilled)),
    )


def rank_positive_bidders(
    task: AuctionTask,
    uuvs: Sequence[AuctionUUV],
) -> tuple[str, ...]:
    """Rank every capability-feasible UUV with positive utility for a task.

    Rolling mission commitment needs alternatives after an earlier task takes
    the first Hungarian winners.  This ranking preserves the same bid model
    and rejection boundary as ``market_allocate`` while leaving cross-task
    uniqueness to the temporal auctioneer.
    """
    slots = _task_slots(task)
    ranked: list[tuple[float, str]] = []
    for uuv in uuvs:
        bids = tuple(
            _bid(uuv, slot)
            for slot in slots
            if slot.role != "active" or uuv.active_capable
        )
        best_bid = max(bids, default=float("-inf"))
        if best_bid > 0.0:
            ranked.append((best_bid, uuv.uuv_id))
    return tuple(
        uuv_id
        for _, uuv_id in sorted(ranked, key=lambda item: (-item[0], item[1]))
    )


def _task_slots(task: AuctionTask) -> list[_Slot]:
    return [
        *(_Slot(task, "active", index) for index in range(task.active_slots)),
        *(_Slot(task, "passive", index) for index in range(task.passive_slots)),
        *(_Slot(task, "reserve", index) for index in range(task.reserve_slots)),
    ]


def _tie_break(row: int, column: int, row_count: int, column_count: int) -> float:
    """Make equal bids stable without changing a meaningful utility value."""
    return 1e-10 * (row * max(column_count, 1) + column) / max(row_count, 1)


def _bid(uuv: AuctionUUV, slot: _Slot) -> float:
    distance_m = hypot(
        uuv.position_xy[0] - slot.task.center_xy[0],
        uuv.position_xy[1] - slot.task.center_xy[1],
    )
    urgency = 1.0 / (1.0 + slot.task.entry_s / 900.0)
    role_value = {
        "active": 1.30,
        "passive": 1.00,
        "reserve": 0.55,
    }[slot.role]
    information_value = slot.task.probability * role_value
    priority_value = 0.15 * slot.task.priority
    continuity_value = 1.50 if uuv.uuv_id in slot.task.continuity_uuv_ids else 0.0
    # The mission carrier transports onboard UUVs to the task region, so the
    # bid must not treat the complete carrier-to-region leg as immediate UUV
    # self-propulsion.  Costs remain monotonic and reject out-of-theater tasks,
    # but are normalized to one 20 km sortie and cap short-window urgency.
    distance_cost = 0.03 * distance_m / 1_000.0
    energy_cost = 0.10 * distance_m / (
        20_000.0 * max(uuv.energy_fraction, 0.10)
    )
    time_cost = min(
        0.25,
        0.05
        * distance_m
        / max(
            uuv.speed_mps * max(slot.task.exit_s - slot.task.entry_s, 1),
            1.0,
        ),
    )
    return (
        information_value
        + priority_value
        + urgency * 0.25
        + continuity_value
        - distance_cost
        - energy_cost
        - time_cost
    )


__all__ = [
    "AuctionAllocation",
    "AuctionAward",
    "AuctionTask",
    "AuctionUUV",
    "market_allocate",
    "rank_positive_bidders",
]
