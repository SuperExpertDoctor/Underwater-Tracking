# Adversary LLM and Bounded Kinematics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the submarine into a mission-directed, locally informed LLM adversary whose deterministic guidance and all fallback behavior obey one two-dimensional fixed-depth kinematic model.

**Architecture:** Carry configured task/escape regions and a mission route into target-owned state. Replace LLM-authored physical waypoints with a high-level intent decision, then resolve that intent through deterministic target guidance. Remove random Markov drift and boundary reflection; route every command through acceleration/deceleration, turn-rate, minimum-turn-radius, and boundary/exclusion guards.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, LangGraph structured output, deterministic geometry, Hypothesis, pytest, Ruff, mypy.

## Global Constraints

- The target LLM knows its own navigation state, mission route, task region, escape regions, and local sensor evidence.
- The target LLM never receives blue global positions, inventory, target estimates, sensor internals, or master plans.
- Target detection acquisition is 1200 m and release hysteresis is 1300 m in the default configuration.
- The LLM emits high-level intent only; deterministic guidance emits physical waypoint, heading, and speed.
- No `depth_change` maneuver or depth/pitch/heave state exists in this phase.
- Every target command, including fallback and boundary avoidance, passes through the shared bounded integrator.
- Normal no-contact motion follows the configured mission route; it does not sample random global directions.
- Phase-one and phase-two gates must pass before the adversarial live acceptance is evaluated.

---

### Task 1: Carry target mission orders and navigation exclusions from config to runtime

**Files:**
- Modify: `src/underwater_tracking/config/platform_core.py`
- Modify: `configs/environment_uuv_only.yaml`
- Modify: `src/underwater_tracking/domain/adversary_models.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `tests/config/test_platform_core_loader.py`
- Modify: `tests/config/test_uuv_only_config.py`
- Create: `tests/domain/test_adversary_mission_state.py`

**Interfaces:**

```python
class SubmarineInitialConfig(StrictConfig):
    mission_route_xy: tuple[CoordinateXY, ...] = Field(min_length=2)

class EnvironmentConfig(StrictConfig):
    navigation_exclusion_regions: tuple[RegionConfig, ...] = ()

class AdversaryMissionState(AdversaryStrictModel):
    target_id: str
    task_region_id: str
    task_region_polygon_xy: tuple[Point2D, ...]
    mission_route_xy: tuple[Point2D, ...]
    escape_regions: Mapping[str, tuple[Point2D, ...]]
    current_intent: Literal[
        "continue_mission", "avoid_contact", "break_contact",
        "escape_to_region", "hold_position"
    ]
    current_route_index: int
    local_contact_ids: tuple[str, ...] = ()
    last_decision_id: str | None = None
```

- [ ] **Step 1: Add failing config tests.**

Reject mission routes outside map bounds, routes intersecting exclusion polygons, unknown task/escape IDs, duplicate escape IDs, and an initial target position that cannot join the first route segment under the turn radius derived from initial speed and the configured maximum turn rate.

- [ ] **Step 2: Add a runtime construction test.**

```python
def test_engine_carries_configured_target_mission_state() -> None:
    engine = build_default_engine()
    mission = engine.target_mission_state("target_00")
    assert mission.task_region_id == "mission_east"
    assert tuple(mission.escape_regions) == ("escape_north", "escape_south")
    assert mission.mission_route_xy[-1] == (8500.0, 0.0)
```

- [ ] **Step 3: Run tests and verify missing route/state failures.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_platform_core_loader.py tests/config/test_uuv_only_config.py tests/domain/test_adversary_mission_state.py -q
```

- [ ] **Step 4: Configure the default route and optional exclusions.**

```yaml
mission_route_xy:
  - [-4500.0, -6500.0]
  - [-1500.0, -4000.0]
  - [2500.0, -1800.0]
  - [5500.0, -500.0]
  - [8500.0, 0.0]
navigation_exclusion_regions: []
```

Validate route geometry without exposing it to blue-side `SituationSnapshot`; target mission state is target-private and evaluation-only access is explicit.

- [ ] **Step 5: Pass state into every `TargetEntity` created by `_spawn_explicit_world()`.**

The entity receives copied immutable route/polygon data. Remove the current behavior where `task_region_id` and `escape_region_ids` stop at config validation.

- [ ] **Step 6: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/config/test_platform_core_loader.py tests/config/test_uuv_only_config.py tests/domain/test_adversary_mission_state.py -q
ruff check src/underwater_tracking/config/platform_core.py src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/simulation/engine.py
mypy src/underwater_tracking/config/platform_core.py src/underwater_tracking/domain/adversary_models.py
git add configs/environment_uuv_only.yaml src/underwater_tracking/config/platform_core.py src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/simulation/engine.py tests/config/test_platform_core_loader.py tests/config/test_uuv_only_config.py tests/domain/test_adversary_mission_state.py
git commit -m "feat: carry submarine mission orders into runtime"
```

---

### Task 2: Replace physical LLM output with a high-level adversary intent contract

**Files:**
- Modify: `src/underwater_tracking/domain/adversary_models.py`
- Modify: `src/underwater_tracking/agent/prompts.py`
- Modify: `src/underwater_tracking/agent/nodes/adversary.py`
- Modify: `src/underwater_tracking/agent/graphs/adversary.py`
- Create: `tests/agent/test_adversary_node.py`
- Modify: `tests/agent/test_adversary_graph.py`
- Create: `tests/agent/test_prompts.py`

**Interfaces:**

```python
AdversaryIntent = Literal[
    "continue_mission", "avoid_contact", "break_contact",
    "escape_to_region", "hold_position"
]

class AdversaryIntentDecision(AdversaryStrictModel):
    decision_id: str
    target_id: str
    intent: AdversaryIntent
    escape_region_id: str | None = None
    confidence: Probability
    rationale: str = Field(min_length=1, max_length=1200)
    trigger_event_ids: tuple[str, ...] = ()
```

`AdversaryEscapeInput` gains `mission_state: AdversaryMissionState`; `AdversaryKinematicLimits` gains `max_acceleration_mps2` and `max_deceleration_mps2`. Remove decoy/communications actions, `Maneuver="depth_change"`, physical `waypoint`, `speed`, and `heading` from the structured LLM output. Deterministic guidance owns all speed selection.

- [ ] **Step 1: Add schema tests.**

```python
def test_escape_intent_requires_configured_escape_region() -> None:
    decision = intent_decision(intent="escape_to_region", escape_region_id="unknown")
    with pytest.raises(ValueError, match="configured escape region"):
        validate_adversary_decision(decision, adversary_context())

def test_non_escape_intent_rejects_escape_region() -> None:
    with pytest.raises(ValueError, match="escape_region_id"):
        AdversaryIntentDecision(
            **base_fields(), intent="continue_mission", escape_region_id="escape_north"
        )
```

- [ ] **Step 2: Add prompt-payload leakage tests.**

Serialize the payload and assert it contains task/escape IDs, `own_position_xy`, and local noisy contacts but does not contain `blue_plan`, `uuv_inventory`, `target_estimate`, a field named `true_position`, any blue/exposed platform world coordinate, or `depth_change`. The target's own navigation coordinate is allowed and is distinct from leaked blue truth.

- [ ] **Step 3: Run tests and verify current physical output contract fails.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_adversary_node.py tests/agent/test_adversary_graph.py tests/agent/test_prompts.py -q
```

- [ ] **Step 4: Update prompt and graph operation.**

Use operation `adversary_mission_decision` and increment the adversary prompt version. Explicitly tell the model to select only one high-level intent and one configured escape ID. Continue one structured repair after content errors; schema or semantic failure becomes a degraded target-brain record, not a fabricated decision.

- [ ] **Step 5: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/agent/test_adversary_node.py tests/agent/test_adversary_graph.py tests/agent/test_prompts.py -q
ruff check src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/agent/nodes/adversary.py src/underwater_tracking/agent/graphs/adversary.py
mypy src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/agent/nodes/adversary.py
git add src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/agent/prompts.py src/underwater_tracking/agent/nodes/adversary.py src/underwater_tracking/agent/graphs/adversary.py tests/agent/test_adversary_node.py tests/agent/test_adversary_graph.py tests/agent/test_prompts.py
git commit -m "refactor: make target llm choose mission intent only"
```

---

### Task 3: Add target-local contact memory and event-driven decision gating

**Files:**
- Modify: `src/underwater_tracking/simulation/adversary_sensing.py`
- Modify: `src/underwater_tracking/agent/nodes/adversary.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `tests/simulation/test_adversary_local_sensing.py`
- Modify: `tests/agent/test_adversary_node.py`
- Modify: `tests/integration/test_uuv_initialization_local_perception.py`

**Interfaces:**

```python
class TargetLocalContact(AdversaryStrictModel):
    platform_id: str
    platform_kind: TargetPlatformKind
    first_seen_s: int
    last_seen_s: int
    estimated_range_m: float
    relative_bearing_rad: float
    threat_level: ThreatLevel
    status: Literal["active", "lost"]

class TargetContactMemory:
    def update(self, result: TargetLocalSensingResult, sim_time_s: int) -> tuple[AdversaryTrigger, ...]: ...
    def active(self, sim_time_s: int) -> tuple[TargetLocalContact, ...]: ...
    def context(self, sim_time_s: int) -> tuple[TargetLocalContact, ...]: ...
```

Default contact TTL is 120 seconds after a lost event. `active()` returns only active contacts; `context()` returns active plus unexpired `lost` contacts for LLM context. Range changes are bucketed at 250 m and threat changes trigger only when the discrete threat level changes.

- [ ] **Step 1: Add memory/gate tests.**

Verify acquire emits once, stable noisy observations emit nothing, loss emits once, TTL preserves a lost contact for decision context without marking it active, and expiration removes it. Active-emitter appearance inside 1200 m triggers even when the platform was already retained by the 1300 m hysteresis band.

- [ ] **Step 2: Add initial mission-command trigger.**

After the phase-two `bootstrap_published` barrier opens, still at sim time 0, each target gets exactly one `target_mission_initialized` trigger. The adversary worker cannot start before that barrier, so the immutable bootstrap frame remains `adversary=ready`; the next frame may be running/completed. The gate then waits for contact acquire/loss, discrete range/threat changes, a newly active emitter, route invalidation, or mission-stage changes. Add one test for each trigger family and assert ordinary observation ticks with an unchanged local signature do not invoke the LLM.

- [ ] **Step 3: Run focused tests and verify failure.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_adversary_local_sensing.py tests/agent/test_adversary_node.py tests/integration/test_uuv_initialization_local_perception.py -q
```

- [ ] **Step 4: Implement contact memory per target in `SimulationEngine`.**

Only `LocalPlatformDetection` values enter the memory. Do not store `ExposedPlatform.position_xy`. Build adversary input from mission state, target-owned navigation state, active/lost contact memory, and target-local triggers.

- [ ] **Step 5: Verify no global leakage and commit.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_adversary_local_sensing.py tests/agent/test_adversary_node.py tests/integration/test_uuv_initialization_local_perception.py -q
ruff check src/underwater_tracking/simulation/adversary_sensing.py src/underwater_tracking/simulation/engine.py tests/simulation/test_adversary_local_sensing.py
git add src/underwater_tracking/simulation/adversary_sensing.py src/underwater_tracking/agent/nodes/adversary.py src/underwater_tracking/simulation/engine.py tests/simulation/test_adversary_local_sensing.py tests/agent/test_adversary_node.py tests/integration/test_uuv_initialization_local_perception.py
git commit -m "feat: gate target decisions on local contact episodes"
```

---

### Task 4: Resolve adversary intent through deterministic target guidance

**Files:**
- Create: `src/underwater_tracking/simulation/target_guidance.py`
- Create: `tests/simulation/test_target_guidance.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class TargetGuidanceCommand:
    decision_id: str | None
    intent: AdversaryIntent
    waypoint_xy: tuple[float, float]
    desired_heading_rad: float
    desired_speed_mps: float
    valid_until_s: int
    source: Literal["llm", "mission_route", "boundary_avoidance", "safe_hold"]

@dataclass(frozen=True, slots=True)
class TargetGuidanceResult:
    command: TargetGuidanceCommand
    next_route_index: int

def resolve_target_guidance(
    *,
    decision: AdversaryIntentDecision | None,
    mission: AdversaryMissionState,
    contacts: tuple[TargetLocalContact, ...],
    state: MotionState,
    limits: MotionLimits,
    operating_boundary: AdversaryOperatingBoundary,
    exclusion_regions: tuple[tuple[tuple[float, float], ...], ...],
    sim_time_s: int,
    previous_guidance: TargetGuidanceCommand | None,
) -> TargetGuidanceResult: ...
```

- [ ] **Step 1: Add deterministic mission-route tests.**

No decision/no contacts selects the next configured route point and cruise speed. The function never mutates mission state: reaching a waypoint returns `next_route_index=current_route_index+1` exactly once, and `TargetEntity` applies that returned index after accepting the command. Reaching the final task-region point produces a bounded patrol/hold command inside the task polygon.

- [ ] **Step 2: Add threat and escape tests.**

`avoid_contact` chooses a heading away from the weighted local bearing vector; `escape_to_region` chooses a reachable point inside the selected polygon; `break_contact` cannot select a waypoint outside boundary or through an exclusion polygon. Same input produces byte-equal command.

- [ ] **Step 3: Add fallback tests.**

An unavailable/invalid LLM decision uses `previous_guidance` only when it is unexpired and its segment remains boundary/exclusion safe; if expired or blocked, it returns mission-route guidance. `hold_position` decelerates through the kinematic model rather than writing zero velocity immediately.

- [ ] **Step 4: Run tests and verify missing module.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_target_guidance.py -q
```

- [ ] **Step 5: Implement pure deterministic guidance.**

Use polygon centroids/interior points and segment-intersection checks from existing geometry helpers where available. Do not add randomness. Guidance chooses requested commands; physical feasibility remains the integrator's responsibility in Task 6.

- [ ] **Step 6: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_target_guidance.py -q
ruff check src/underwater_tracking/simulation/target_guidance.py tests/simulation/test_target_guidance.py
mypy src/underwater_tracking/simulation/target_guidance.py
git add src/underwater_tracking/simulation/target_guidance.py tests/simulation/test_target_guidance.py
git commit -m "feat: resolve target intents into deterministic guidance"
```

---

### Task 5: Replace random target motion with mission guidance and safe fallback

**Files:**
- Modify: `src/underwater_tracking/simulation/target.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `tests/simulation/test_target.py`
- Create: `tests/simulation/test_target_adversary_motion.py`
- Modify: `tests/agent/test_runtime_master_slave_adversary.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.test.tsx`

**Interfaces:**

```python
class TargetEntity:
    def apply_adversary_intent(
        self, decision: AdversaryIntentDecision, *, sim_time_s: int
    ) -> None: ...
    def step(self, dt_s: float, sim_time_s: int) -> None: ...
```

- [ ] **Step 1: Add a no-contact route-following test over 600 seconds.**

Assert distance along the configured route increases monotonically within waypoint tolerance, intent remains `continue_mission`, and two identical runs produce identical trajectories without using `random.Random`.

- [ ] **Step 2: Add decision/fallback tests.**

Apply an `escape_to_region` decision, verify its deterministic guidance is active until expiry, then verify the entity returns to mission guidance. A target-brain exception does not stop `engine.step()` and records a degraded adversary summary.

- [ ] **Step 3: Run tests and confirm random Markov behavior fails them.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_target.py tests/simulation/test_target_adversary_motion.py tests/agent/test_runtime_master_slave_adversary.py -q
```

- [ ] **Step 4: Remove `_sample_intent()`, `TRANSITION_PROBABILITIES`, and global direction tables from the live target path.**

Keep legacy replay labels only where required. `TargetEntity.step()` asks `target_guidance` for the current command and passes it to the shared motion executor. Replace `apply_adversary_decision()` with `apply_adversary_intent()`; this single-target scenario continues to reject non-empty decoy inventory.

- [ ] **Step 5: Update engine application and operator summary.**

Store both high-level intent decision and resolved guidance ID. Extend the backend and TypeScript `AdversaryView` contracts with intent, selected escape-region ID, confidence, decision source, and bounded rationale. Render those fields in the existing adversary section and label the resolved guidance speed/waypoint separately. It must not claim the LLM authored the executed speed or waypoint. Add a component test that distinguishes “LLM 意图” from “确定性制导”.

- [ ] **Step 6: Verify and commit.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_target.py tests/simulation/test_target_adversary_motion.py tests/agent/test_runtime_master_slave_adversary.py tests/api/test_frame_pipeline.py -q
npm --prefix src/underwater_tracking/ui test -- --run src/components/RightSidebar.test.tsx
ruff check src/underwater_tracking/simulation/target.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/api/frame_builder.py
git add src/underwater_tracking/simulation/target.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/components/RightSidebar.tsx src/underwater_tracking/ui/src/components/RightSidebar.test.tsx tests/simulation/test_target.py tests/simulation/test_target_adversary_motion.py tests/agent/test_runtime_master_slave_adversary.py tests/api/test_frame_pipeline.py
git commit -m "fix: drive target motion from mission guidance"
```

---

### Task 6: Enforce acceleration, deceleration, turn radius, and boundary avoidance

**Files:**
- Modify: `src/underwater_tracking/domain/platforms.py`
- Modify: `src/underwater_tracking/config/platform_core.py`
- Modify: `configs/platforms.yaml`
- Modify: `src/underwater_tracking/simulation/kinematics.py`
- Modify: `src/underwater_tracking/simulation/target.py`
- Modify: `tests/simulation/test_kinematics.py`
- Create: `tests/simulation/test_target_boundary_avoidance.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/simulation/uuv.py`
- Modify: `tests/agent/test_central_graph.py`
- Modify: `tests/agent/test_regional_plan_pipeline.py`
- Modify: `tests/agent/test_strategy.py`
- Modify: `tests/api/test_frame_pipeline.py`
- Modify: `tests/domain/test_models.py`
- Modify: `tests/domain/test_platform_contracts.py`
- Modify: `tests/planning/test_mission_optimizer.py`
- Modify: `tests/planning/test_regional_allocation.py`
- Modify: `tests/planning/test_regional_validation.py`

**Interfaces:**

```python
class MotionLimits(PlatformModel):
    min_speed_mps: NonNegativeFloat
    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_deceleration_mps2: PositiveFloat
    max_turn_rate_rad_s: PositiveFloat

@dataclass(frozen=True, slots=True)
class NavigationBoundary:
    bounds_xy: tuple[float, float, float, float]
    exclusion_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    safety_margin_m: float = 50.0

def constrain_navigation_command(
    state: MotionState,
    requested: MotionCommand,
    limits: MotionLimits,
    boundary: NavigationBoundary,
    dt_s: float,
) -> MotionCommand: ...
```

- [ ] **Step 1: Extend kinematic property tests.**

For every step assert:

```python
assert abs(next.speed_mps - state.speed_mps) <= (
    limits.max_acceleration_mps2 * dt_s
    if next.speed_mps >= state.speed_mps
    else limits.max_deceleration_mps2 * dt_s
) + 1e-9
assert angular_distance(state.heading_rad, next.heading_rad) <= (
    limits.max_turn_rate_rad_s * dt_s + 1e-9
)
assert limits.min_speed_mps <= next.speed_mps <= limits.max_speed_mps
```

Add a minimum-turn-radius assertion for positive speed: `radius >= speed / max_turn_rate` within integration tolerance.

- [ ] **Step 2: Add boundary regression tests.**

For nonzero cruise/sprint cases, start each target inside the edge/corner by at least `stopping_distance + turn_radius + safety_margin`, heading outward, then advance until it turns back. Separately start exact-edge/corner cases at zero speed and require an inward command before acceleration. Assert no single-step heading jump exceeds the turn limit, speed change respects acceleration/deceleration, no position leaves the map, and no velocity component is instantaneously reflected.

- [ ] **Step 3: Run tests and confirm `_reflect_into_bounds()` fails.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_kinematics.py tests/simulation/test_target_boundary_avoidance.py -q
```

- [ ] **Step 4: Add explicit deceleration to all motion profiles and constructors.**

Set `submarine_standard.min_speed_mps: 0.0` and `submarine_standard.max_deceleration_mps2: 0.10`. Give UUV/legacy profiles explicit `min_speed_mps: 0.0` plus deceleration values equal to their current acceleration unless a more restrictive configured value exists. Update all `MotionLimits` fixtures; do not rely on an implicit default in production config. Validate `min_speed_mps < max_speed_mps`.

- [ ] **Step 5: Implement pre-emptive boundary constraint.**

Compute stopping distance `v^2 / (2 * max_deceleration)` and turn radius `v / max_turn_rate`. Begin avoidance before the nearest boundary is closer than their sum plus `safety_margin_m`. Choose a tangent/inward heading reachable under the turn limit. Use segment-polygon intersection to treat exclusion polygons identically.

- [ ] **Step 6: Remove `_reflect_into_bounds()`.**

Use adaptive substeps near a boundary so the accepted segment remains legal. Keep a numerical invariant check after integration. If no legal positive substep exists, tests raise `NavigationInvariantError`; live mode retains the last complete legal `MotionState`, emits `target_navigation_guard_failed`, and installs an inward safe-hold command for the next tick. It does not clip position or flip velocity.

- [ ] **Step 7: Run motion suites and commit.**

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_kinematics.py tests/simulation/test_target.py tests/simulation/test_target_adversary_motion.py tests/simulation/test_target_boundary_avoidance.py tests/simulation/test_uuv.py -q
ruff check src/underwater_tracking/domain/platforms.py src/underwater_tracking/config/platform_core.py src/underwater_tracking/simulation/kinematics.py src/underwater_tracking/simulation/target.py
mypy src/underwater_tracking/simulation/kinematics.py src/underwater_tracking/simulation/target.py
git add configs/platforms.yaml src/underwater_tracking/domain/platforms.py src/underwater_tracking/config/platform_core.py src/underwater_tracking/simulation/kinematics.py src/underwater_tracking/simulation/target.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/simulation/uuv.py tests/simulation/test_kinematics.py tests/simulation/test_target_boundary_avoidance.py tests/agent/test_central_graph.py tests/agent/test_regional_plan_pipeline.py tests/agent/test_strategy.py tests/api/test_frame_pipeline.py tests/domain/test_models.py tests/domain/test_platform_contracts.py tests/planning/test_mission_optimizer.py tests/planning/test_regional_allocation.py tests/planning/test_regional_validation.py
git commit -m "fix: enforce bounded target turns at navigation boundaries"
```

---

### Task 7: Restore exact translation equivariance in UUV waypoint planning

**Files:**
- Modify: `src/underwater_tracking/planning/waypoints.py`
- Modify: `tests/planning/test_waypoints.py`

**Interfaces:**
- Consumes: `plan_group_waypoints()` existing signature.
- Produces: deterministic results invariant under rigid translation and UUV input permutation.

- [ ] **Step 1: Freeze the currently failing Hypothesis example as a named regression.**

Store only the minimal positions, sigma points, previous waypoints, translation and planner parameters needed to reproduce the endpoint swap. Assert both `waypoints_xy` and `sequence_xy` translate exactly within `1e-7`.

- [ ] **Step 2: Run the exact regression and property test.**

```bash
PYTHONPATH=src python -m pytest tests/planning/test_waypoints.py::test_translation_regression_endpoint_order tests/planning/test_waypoints.py::test_translation_moves_waypoints_exactly -q
```

- [ ] **Step 3: Make tie-breaking translation-relative.**

Do not sort or tie-break by absolute world coordinates. Build tie keys from candidate offsets relative to the corresponding UUV position and target centroid, then stable UUV ID/order. Quantize only at the existing floating tolerance; do not reduce geometric score precision.

- [ ] **Step 4: Run complete waypoint properties.**

```bash
PYTHONPATH=src python -m pytest tests/planning/test_waypoints.py -q
ruff check src/underwater_tracking/planning/waypoints.py tests/planning/test_waypoints.py
```

- [ ] **Step 5: Commit.**

```bash
git add src/underwater_tracking/planning/waypoints.py tests/planning/test_waypoints.py
git commit -m "fix: make waypoint tie breaks translation invariant"
```

---

### Task 8: Verify the live enemy-blue interaction chain

**Files:**
- Create: `tests/integration/test_live_adversarial_game_loop.py`
- Modify: `tests/integration/test_uuv_initialization_local_perception.py`
- Modify: `tests/agent/test_runtime_master_slave_adversary.py`

**Interfaces:**
- Consumes: phase-one epoch runtime, phase-two deployment truth, target mission state, local sensing, adversary graph, deterministic guidance, and bounded kinematics.
- Produces: a semantic event sequence with ledger and trajectory evidence.

- [ ] **Step 1: Build a controlled structured LLM suite.**

Master returns one feasible region plan, slave returns active/passive sonar actions, and adversary returns `avoid_contact` on acquisition followed by `escape_to_region=escape_north` on active-emitter escalation. The fake records exact payloads and invocation times.

- [ ] **Step 2: Run until semantic checkpoints, not fixed ticks.**

Wait in order for plan commit, physical deployment, active scan, target local acquisition, target LLM decision, target guidance activation, blue observation change, and a key-event planning epoch.

- [ ] **Step 3: Assert information and motion invariants.**

The target payload contains only locally detected blue IDs plus target-owned mission/navigation state. Assert calls for initialization, acquisition/loss, active-emitter escalation, route invalidation and mission-stage change, with no call on unchanged ticks. Every accepted trajectory delta satisfies speed/acceleration/deceleration/turn bounds. At least one target intent decision and one subsequent master planning epoch appear in their separate ledgers.

- [ ] **Step 4: Run integration tests.**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_live_adversarial_game_loop.py tests/integration/test_uuv_initialization_local_perception.py tests/agent/test_runtime_master_slave_adversary.py -q
```

- [ ] **Step 5: Commit.**

```bash
git add tests/integration/test_live_adversarial_game_loop.py tests/integration/test_uuv_initialization_local_perception.py tests/agent/test_runtime_master_slave_adversary.py
git commit -m "test: verify locally informed adversarial game loop"
```

---

### Task 9: Run the phase-three gate and record evidence

**Files:**
- Create: `docs/superpowers/reports/2026-08-22-adversary-llm-kinematics-acceptance.md`

**Interfaces:**
- Consumes: all phase-three tasks and phase-one/two reports.
- Produces: adversary and physics acceptance evidence required for final E2E release.

- [ ] **Step 1: Run focused adversary and motion tests.**

```bash
PYTHONPATH=src python -m pytest tests/domain/test_adversary_mission_state.py tests/agent/test_adversary_node.py tests/agent/test_adversary_graph.py tests/simulation/test_adversary_local_sensing.py tests/simulation/test_target_guidance.py tests/simulation/test_target.py tests/simulation/test_target_adversary_motion.py tests/simulation/test_target_boundary_avoidance.py tests/planning/test_waypoints.py tests/integration/test_live_adversarial_game_loop.py -q
```

- [ ] **Step 2: Run backend gates.**

```bash
PYTHONPATH=src python -m pytest -m "not real_llm and not long_running" -q
ruff check main.py src tests
mypy src/underwater_tracking
```

- [ ] **Step 3: Record trajectory and ledger evidence.**

For each target step record speed, acceleration, heading delta, computed turn radius, boundary distance, guidance source, LLM decision ID, and locally detected platform IDs. The report includes maxima/minima and proves no constraint breach.

- [ ] **Step 4: Write and commit the report.**

```bash
git add docs/superpowers/reports/2026-08-22-adversary-llm-kinematics-acceptance.md
git commit -m "test: document adversary llm and kinematics acceptance"
```
