# src/underwater_tracking/agent/nodes/active_verification.py
"""Deterministic active-sonar verification protocol (spec 17.3, R5).

The protocol is a strict per-contact state machine:

    idle -> verifying (one available non-reserved UUV pings the contact)
         -> classified_submarine (the engine classified the contact)
         -> in_position (every dispatched member is inside the geometric gate)
    idle -> verifying -> (decoy classified -> drop + return to passive)

``ActiveVerificationNode`` is pure graph logic: it reads the coalesced
events and the live situation, updates the per-contact protocol state,
and emits engine-facing ``VerificationCommand`` rows. The engine runs the
ping simulation and contact classification; the node only routes. A ping
that finds no available UUV leaves the contact absent — the engine
re-emits ``active_ping`` on the next observation cycle, so the protocol
simply waits. Commands are replaced every cycle so stale commands never
leak into the engine. Deterministic: no randomness, no wall clock.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import VerificationCommand
from underwater_tracking.domain.availability import is_deployable
from underwater_tracking.domain.models import (
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    UUVStatus,
)
from underwater_tracking.planning.reservations import ReservationRegistry

# Per-contact protocol states (spec 17.3).
_STATE_VERIFYING = "verifying"
_STATE_CLASSIFIED_SUBMARINE = "classified_submarine"
_STATE_IN_POSITION = "in_position"

_PING_EVENT = "active_ping"
_CLASSIFIED_EVENT = "contact_classified"
_DEFAULT_GATE_M = 1200.0


class _VerificationState(CarrierState, total=False):
    """Branch state: the carrier channels plus the deferred error marker."""

    node_error: str | None


class ActiveVerificationNode:
    """Run the active-sonar verification state machine (spec 17.3, R5).

    One protocol per contact id; the node replaces (never accumulates)
    ``verification_commands`` on every cycle. ``situation_provider``
    resolves the live situation under the cycle's ``snapshot_ref``;
    ``reservations`` is the shared human-assignment registry (reserved
    UUVs are never pingers).
    """

    def __init__(
        self,
        reservations: ReservationRegistry | None,
        situation_provider: Callable[[str], SituationSnapshot] | None = None,
        *,
        in_position_gate_m: float = _DEFAULT_GATE_M,
    ) -> None:
        self._reservations = reservations
        self._situation_provider = situation_provider
        self._in_position_gate_m = in_position_gate_m
        self._states: dict[str, str] = {}
        self._pingers: dict[str, str] = {}

    def __call__(self, state: _VerificationState) -> _VerificationState:
        ref = state.get("snapshot_ref")
        if ref is None:
            return {"node_error": "active_verification requires snapshot_ref in state"}
        if self._situation_provider is None:
            return {"node_error": "active_verification requires a situation provider"}
        situation = self._situation_provider(ref)
        commands: list[VerificationCommand] = []
        for event in state.get("coalesced_events") or ():
            if event.event_type == _PING_EVENT:
                self._on_ping(event, situation, commands)
            elif event.event_type == _CLASSIFIED_EVENT:
                self._on_classified(event, situation, commands)
        for contact in (
            contact
            for contact, protocol_state in self._states.items()
            if protocol_state == _STATE_CLASSIFIED_SUBMARINE
        ):
            if self._in_position(contact, situation):
                self._close_gate(contact, situation, commands)
        return {
            "verification_states": dict(self._states),
            "verification_pingers": dict(self._pingers),
            "verification_commands": tuple(commands),
        }

    def _on_ping(
        self,
        event: RuntimeEvent,
        situation: SituationSnapshot,
        commands: list[VerificationCommand],
    ) -> None:
        """Nearest-first pinger selection; idempotent for known contacts."""
        contact = event.entity_id
        if not contact or contact in self._states:
            return
        pinger = self._nearest_available(event, situation)
        if pinger is None:
            return  # no available UUV: the engine re-emits the ping later
        self._states[contact] = _STATE_VERIFYING
        self._pingers[contact] = pinger
        commands.append(
            VerificationCommand(
                command_id=(
                    f"{situation.scenario_id}:verify:{contact}:ping:{situation.sim_time_s}"
                ),
                target_id=contact,
                sensor_mode="ping",
                uuv_ids=(pinger,),
                sim_time_s=situation.sim_time_s,
            )
        )

    def _nearest_available(
        self, event: RuntimeEvent, situation: SituationSnapshot
    ) -> str | None:
        """Choose a nearest available UUV from an operational estimate.

        Engine-generated ping requests contain no position. The optional
        event payload is retained for standalone protocol callers, while the
        live path prefers the contact estimate carried by the situation.
        Without either estimate the protocol still chooses the first admitted
        UUV, allowing the ping to improve the contact estimate instead of
        stalling on a missing coordinate.
        """
        position: Sequence[object] | None = event.payload.get("position_xy")
        contact = next(
            (item for item in situation.contacts if item.contact_id == event.entity_id),
            None,
        )
        if contact is not None and contact.estimated_position_xy is not None:
            position = contact.estimated_position_xy
        estimate: tuple[float, float] | None = None
        if isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
            try:
                raw_x, raw_y = position[0], position[1]
                if isinstance(raw_x, (int, float)) and isinstance(raw_y, (int, float)):
                    estimate = (float(raw_x), float(raw_y))
            except (TypeError, IndexError, ValueError):
                estimate = None
        reserved = (
            self._reservations.reserved_uuvs()
            if self._reservations is not None
            else frozenset()
        )
        busy = set(self._pingers.values())
        candidates = [
            uuv
            for uuv in situation.uuvs
            if uuv.status == UUVStatus.ACTIVE
            and is_deployable(uuv)
            and uuv.uuv_id not in reserved
            and uuv.uuv_id not in busy
        ]
        if not candidates:
            return None
        if estimate is None:
            return min(candidates, key=lambda uuv: uuv.uuv_id).uuv_id
        x, y = estimate
        nearest = min(
            candidates,
            key=lambda uuv: (uuv.position_xy[0] - x) ** 2
            + (uuv.position_xy[1] - y) ** 2,
        )
        return nearest.uuv_id

    def _on_classified(
        self,
        event: RuntimeEvent,
        situation: SituationSnapshot,
        commands: list[VerificationCommand],
    ) -> None:
        """Route the engine's contact classification (spec 17.3)."""
        contact = event.entity_id
        if not contact:
            return
        outcome = str(event.payload.get("outcome", ""))
        if outcome == "submarine":
            # True target: dispatch a tracking group through the existing
            # allocation channel (the engine forms the group).
            self._states[contact] = _STATE_CLASSIFIED_SUBMARINE
            commands.append(
                VerificationCommand(
                    command_id=(
                        f"{situation.scenario_id}:verify:{contact}:dispatch:{situation.sim_time_s}"
                    ),
                    target_id=contact,
                    sensor_mode="dispatch",
                    sim_time_s=situation.sim_time_s,
                )
            )
            return
        if outcome == "decoy":
            pinger = self._pingers.pop(contact, None)
            self._states.pop(contact, None)
            commands.append(
                VerificationCommand(
                    command_id=(
                        f"{situation.scenario_id}:verify:{contact}:drop:{situation.sim_time_s}"
                    ),
                    target_id=contact,
                    sensor_mode="drop",
                    sim_time_s=situation.sim_time_s,
                )
            )
            if pinger is not None:
                commands.append(
                    VerificationCommand(
                        command_id=(
                            f"{situation.scenario_id}:verify:{contact}:return_to_passive:{situation.sim_time_s}"
                        ),
                        target_id=contact,
                        sensor_mode="return_to_passive",
                        uuv_ids=(pinger,),
                        sim_time_s=situation.sim_time_s,
                    )
                )

    def _in_position(self, contact: str, situation: SituationSnapshot) -> bool:
        """All dispatched members inside the geometric gate around the mean."""
        report = self._report_for(contact, situation)
        if report is None or not report.member_ids:
            return False
        mean_x, mean_y = report.belief.mean[0], report.belief.mean[1]
        uuvs_by_id = {uuv.uuv_id: uuv for uuv in situation.uuvs}
        return all(
            uuvs_by_id.get(member) is not None
            and is_deployable(uuvs_by_id[member])
            and (
                (uuvs_by_id[member].position_xy[0] - mean_x) ** 2
                + (uuvs_by_id[member].position_xy[1] - mean_y) ** 2
                <= self._in_position_gate_m**2
            )
            for member in report.member_ids
        )

    def _close_gate(
        self,
        contact: str,
        situation: SituationSnapshot,
        commands: list[VerificationCommand],
    ) -> None:
        """Switch the dispatched team back to passive bearing-only tracking."""
        report = self._report_for(contact, situation)
        assert report is not None, "the gate only closes over an existing report"
        team = set(report.member_ids)
        if pinger := self._pingers.pop(contact, None):
            team.add(pinger)
        self._states[contact] = _STATE_IN_POSITION
        commands.append(
            VerificationCommand(
                command_id=(
                    f"{situation.scenario_id}:verify:{contact}:return_to_passive:{situation.sim_time_s}"
                ),
                target_id=contact,
                sensor_mode="return_to_passive",
                uuv_ids=tuple(sorted(team)),
                sim_time_s=situation.sim_time_s,
            )
        )

    def _report_for(
        self, contact: str, situation: SituationSnapshot
    ) -> GroupReport | None:
        for report in situation.group_reports:
            if report.target_id == contact:
                return report
        return None
