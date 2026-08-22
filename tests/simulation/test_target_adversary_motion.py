from __future__ import annotations

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.adversary_models import AdversaryIntentDecision
from underwater_tracking.simulation.engine import SimulationEngine


def test_target_motion_is_deterministic_and_uses_configured_route() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")

    def trajectory() -> list[tuple[float, float]]:
        engine = SimulationEngine(config, seed=42)
        target = engine._targets["target_00"]
        result: list[tuple[float, float]] = []
        for _ in range(60):
            engine._advance_world(engine._clock.sim_time_s)
            result.append(target.position_xy)
        return result

    assert trajectory() == trajectory()


def test_target_intent_resolves_to_guidance_and_keeps_physics_bounded() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=42)
    target = engine._targets["target_00"]
    decision = AdversaryIntentDecision(
        decision_id="target-escape-1",
        target_id="target_00",
        intent="escape_to_region",
        escape_region_id="escape_north",
        confidence=0.9,
        rationale="Local target-owned contact evidence requires the northern escape.",
    )

    engine.apply_adversary_intent(decision)
    before = target.position_xy
    engine._advance_world(0)

    assert target.position_xy != before
    assert target.guidance_command is not None
    assert target.guidance_command.source == "llm"
    assert target.guidance_command.desired_speed_mps <= target.max_speed_mps
