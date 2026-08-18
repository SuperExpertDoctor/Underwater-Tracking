"""Continuous and adaptive target manoeuvre behaviour."""

from __future__ import annotations

import math
import random

from underwater_tracking.simulation.target import HiddenIntent, TargetEntity


def test_active_ping_evasion_is_held_and_turns_smoothly() -> None:
    target = TargetEntity(
        target_id="target_01",
        position_xy=(0.0, 0.0),
        velocity_xy=(8.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        max_acceleration_mps2=0.5,
        max_turn_rate_rad_s=0.2,
    )

    target.apply_evasive_maneuver(math.pi / 2)
    headings: list[float] = []
    speeds: list[float] = []
    rng = random.Random(0)
    for _ in range(12):
        target.step(1.0, rng)
        headings.append(math.atan2(target.velocity_xy[1], target.velocity_xy[0]))
        speeds.append(math.hypot(*target.velocity_xy))

    assert target.intent is HiddenIntent.EVADE
    assert max(abs(b - a) for a, b in zip(headings, headings[1:])) <= 0.2 + 1e-9
    assert max(abs(b - a) for a, b in zip(speeds, speeds[1:])) <= 0.5 + 1e-9
    assert max(headings) - min(headings) > 0.5
    assert max(abs(heading) for heading in headings[6:]) > 0.2


def test_adversary_decision_changes_course_without_teleporting() -> None:
    target = TargetEntity(
        target_id="target_01",
        position_xy=(0.0, 0.0),
        velocity_xy=(8.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        max_acceleration_mps2=0.5,
        max_turn_rate_rad_s=0.2,
    )
    from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision

    target.apply_adversary_decision(AdversaryEscapeDecision(
        target_id="target_01",
        intent="deception",
        waypoint=(20.0, 60.0),
        speed=12.0,
        heading=math.pi / 2,
        maneuver="course_change",
        segment="region_2",
        decoy_action="none",
        decoy_count=0,
        confidence=0.9,
        rationale="Detected platforms require a smooth course change.",
        communications_discipline="silent",
    ), hold_steps=4)
    before = target.position_xy
    target.step(1.0, random.Random(1))
    after = target.position_xy

    assert after != before
    assert math.dist(before, after) < 9.0
    assert target.intent is HiddenIntent.EVADE


def test_adversary_command_interpolates_and_expires_without_heading_jump() -> None:
    from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision

    target = TargetEntity(
        target_id="target_01",
        position_xy=(0.0, 0.0),
        velocity_xy=(8.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        max_acceleration_mps2=0.5,
        max_turn_rate_rad_s=0.2,
    )
    target.apply_adversary_decision(
        AdversaryEscapeDecision(
            target_id="target_01",
            intent="evade",
            waypoint=(0.0, 100.0),
            speed=12.0,
            heading=math.pi / 2,
            maneuver="course_change",
            segment="region_2",
            decoy_action="none",
            decoy_count=0,
            confidence=0.9,
            rationale="Turn toward the safe corridor.",
            communications_discipline="silent",
        ),
        hold_steps=2,
    )

    assert target.maneuver_command is not None
    assert target.maneuver_command.remaining_steps == 2
    headings = []
    speeds = []
    positions = [target.position_xy]
    for _ in range(3):
        target.step(1.0, random.Random(3))
        headings.append(math.atan2(target.velocity_xy[1], target.velocity_xy[0]))
        speeds.append(math.hypot(*target.velocity_xy))
        positions.append(target.position_xy)

    assert max(abs(b - a) for a, b in zip(headings, headings[1:])) <= 0.2 + 1e-9
    assert max(abs(b - a) for a, b in zip(speeds, speeds[1:])) <= 0.5 + 1e-9
    assert all(math.dist(a, b) <= target.max_speed_mps for a, b in zip(positions, positions[1:]))
    assert target.maneuver_command is None


def test_seeded_target_motion_is_reproducible() -> None:
    def trajectory(seed: int) -> list[tuple[float, float]]:
        target = TargetEntity(
            target_id="target_01",
            position_xy=(0.0, 0.0),
            velocity_xy=(8.0, 0.0),
            intent=HiddenIntent.TRANSIT,
            max_acceleration_mps2=0.5,
            max_turn_rate_rad_s=0.2,
        )
        rng = random.Random(seed)
        result = []
        for _ in range(20):
            target.step(1.0, rng)
            result.append(target.position_xy)
        return result

    assert trajectory(17) == trajectory(17)
