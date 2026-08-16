# Battlespace Platform Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the explicit `1 carrier + 4 USV + 12 UUV + 1 submarine` battlefield foundation with strict platform contracts, continuous 2-D kinematics, distance-only communication connectivity, passive/multistatic sonar observations, and a backward-compatible engine adapter.

**Architecture:** Add focused platform and observation contracts beside the legacy UUV-only contracts, load a new explicit single-target scenario from strict YAML files, and expose the new world through a `PlatformSnapshot` while retaining the existing `SituationSnapshot` for current LangGraph consumers. Shared deterministic services own motion, communication, and sonar math; later plans consume these services without changing their interfaces.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy/SciPy already present in the project, PyYAML, Pytest, Ruff, Mypy.

## Global Constraints

- Implement only sections 4-8 and the platform-core portions of sections 17-19 in `docs/superpowers/specs/2026-08-16-hierarchical-adversarial-segmented-tracking-design.md`.
- Python must satisfy the project floor `>=3.11,<3.13`; use `.venv/bin/python` (Python 3.11.5). The conda environment named `lang_py310` is Python 3.10.20 and cannot import `enum.StrEnum`, so it is not a valid execution environment for this repository without upgrading that environment.
- Preserve all existing uncommitted changes in `src/underwater_tracking/agent/`, `src/underwater_tracking/domain/agent_models.py`, `src/underwater_tracking/planning/allocation.py`, and their tests; never stage them in this plan's commits.
- Keep `configs/scenario/default.yaml` and the current UUV-only runtime working until later migration plans remove the compatibility path.
- The new production scenario is `configs/scenario/segmented_single_target.yaml` and must obtain every platform ID, initial position, capability profile, and target region from YAML.
- The new scenario contains exactly one submarine and no decoys.
- Do not add LLM decisions, LangGraph nodes, spatial segmentation, group allocation, or UI changes in this plan.
- Truth may enter simulation services and evaluation output only; no public platform or observation model may expose the submarine truth position as a field named `truth`, `true_position`, or equivalent.
- Use strict Pydantic models with `extra="forbid"` and finite numeric constraints.
- Follow TDD for every task: failing focused test, minimal implementation, passing focused test, then regression tests and a dedicated commit.
- Add no new third-party dependency.

---

## Locked File Structure

### New files

- `src/underwater_tracking/domain/platforms.py` — generic platform capabilities, public platform states, carrier roster, and `PlatformSnapshot`.
- `src/underwater_tracking/domain/observations.py` — generic passive and multistatic sonar observation contracts.
- `src/underwater_tracking/config/platform_core.py` — strict schemas for environment, platform profiles, sensor profiles, and communication ranges.
- `src/underwater_tracking/simulation/kinematics.py` — shared continuous 2-D motion integrator.
- `src/underwater_tracking/simulation/usv.py` — USV simulation entity.
- `src/underwater_tracking/simulation/connectivity.py` — distance-only graph construction and reachability.
- `src/underwater_tracking/simulation/sonar.py` — passive bearing and multistatic observation generation.
- `configs/scenario/segmented_single_target.yaml` — explicit new scenario entry point.
- `configs/environment.yaml` — carrier, USV, UUV, submarine, task region, and escape regions.
- `configs/platforms.yaml` — motion, energy, deployment, and support profiles.
- `configs/sensors.yaml` — passive and active sonar profiles.
- `configs/communications.yaml` — link distance thresholds.
- `tests/domain/test_platform_contracts.py` — platform contract and truth-isolation tests.
- `tests/domain/test_observation_contracts.py` — sonar observation contract tests.
- `tests/config/test_platform_core_loader.py` — explicit configuration tests.
- `tests/simulation/test_connectivity.py` — distance graph tests.
- `tests/simulation/test_multistatic_sonar.py` — passive and multistatic simulation tests.
- `tests/integration/test_platform_core_scenario.py` — new scenario engine acceptance test.

### Modified files

- `src/underwater_tracking/config/models.py:26-153` — add scenario file references and optional loaded platform-core sections.
- `src/underwater_tracking/config/loader.py:22-49` — resolve referenced YAML files below `configs/` and reject path traversal.
- `src/underwater_tracking/simulation/uuv.py:1-35` — delegate motion to the shared integrator while preserving the current constructor and `step` call.
- `src/underwater_tracking/simulation/carrier.py:15-80` — accept explicit initial state, route, speed, and support radius while preserving the no-argument compatibility constructor.
- `src/underwater_tracking/simulation/target.py:100-171` — replace discontinuous velocity changes with shared bounded motion commands.
- `src/underwater_tracking/simulation/engine.py:163-219,241-303,338-403,875-942,1283-1350` — spawn explicit platform-core worlds, advance USVs, compute connectivity, emit additive platform data, and keep the legacy situation adapter.
- `tests/simulation/test_kinematics.py` — verify acceleration/turn continuity and legacy behavior.
- `tests/simulation/test_engine.py` — verify additive frame fields do not change the legacy path.

---

### Task 1: Define Strict Platform Contracts

**Files:**
- Create: `src/underwater_tracking/domain/platforms.py`
- Create: `tests/domain/test_platform_contracts.py`

**Interfaces:**
- Consumes: `underwater_tracking.domain.models.StrictModel`.
- Produces: `PlatformKind`, `MotionLimits`, `SonarCapability`, `CommunicationCapability`, `PlatformCapability`, `USVPlatformState`, `UUVPlatformState`, `CarrierPlatformState`, `PlatformRoster`, `CommunicationLink`, and `PlatformSnapshot`.

- [ ] **Step 1: Write failing contract tests**

Create `tests/domain/test_platform_contracts.py`:

```python
from math import inf

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    USVPlatformState,
    UUVPlatformState,
)


def capability(kind: PlatformKind) -> PlatformCapability:
    return PlatformCapability(
        kind=kind,
        motion=MotionLimits(
            max_speed_mps=6.0,
            max_acceleration_mps2=0.2,
            max_turn_rate_rad_s=0.03,
        ),
        sonar=SonarCapability(
            passive_range_m=5000.0,
            passive_bearing_variance_rad2=0.01,
            active_source_range_m=4000.0,
            active_receive_range_m=5000.0,
            active_range_sigma_m=12.0,
            active_bearing_sigma_rad=0.003,
            active_capable=True,
            ping_cooldown_s=30,
            ping_energy_cost_fraction=0.001,
            clutter_sensitivity=0.2,
            exposure_cost=0.4,
        ),
        communications=CommunicationCapability(
            surface_range_m=12000.0,
            acoustic_range_m=4500.0,
        ),
    )


def test_platform_snapshot_keeps_truth_out_of_public_contract() -> None:
    usv = USVPlatformState(
        platform_id="usv_00",
        platform_index=0,
        position_xy=(100.0, 0.0),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=0.9,
        deployment_state="deployed",
        capability=capability(PlatformKind.USV),
        distance_to_carrier_m=100.0,
    )
    uuv = UUVPlatformState(
        platform_id="uuv_00",
        platform_index=0,
        position_xy=(50.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        energy_fraction=0.8,
        deployment_state="deployed",
        capability=capability(PlatformKind.UUV),
    )
    carrier = CarrierPlatformState(
        carrier_id="carrier_01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=3.0,
        support_radius_m=15000.0,
        onboard_platform_ids=(),
        deployed_platform_ids=("usv_00", "uuv_00"),
        returning_platform_ids=(),
    )
    snapshot = PlatformSnapshot(
        scenario_id="single-target-relay",
        sim_time_s=30,
        carrier=carrier,
        roster=PlatformRoster(usvs=(usv,), uuvs=(uuv,)),
        communication_links=(),
    )

    payload = snapshot.model_dump()
    assert payload["roster"]["usvs"][0]["platform_id"] == "usv_00"
    assert "truth" not in repr(payload).lower()
    assert "true_position" not in repr(payload).lower()


def test_carrier_rejects_duplicate_or_overlapping_relationships() -> None:
    with pytest.raises(ValidationError, match="unique and disjoint"):
        CarrierPlatformState(
            carrier_id="carrier_01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=3.0,
            support_radius_m=15000.0,
            onboard_platform_ids=("uuv_00", "uuv_00"),
            deployed_platform_ids=(),
            returning_platform_ids=(),
        )


def test_roster_rejects_duplicate_indices_within_platform_kind() -> None:
    first = UUVPlatformState(
        platform_id="uuv_00",
        platform_index=0,
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=0.0,
        energy_fraction=1.0,
        deployment_state="onboard",
        capability=capability(PlatformKind.UUV),
    )
    second = first.model_copy(update={"platform_id": "uuv_01"})

    with pytest.raises(ValidationError, match="indices must be unique"):
        PlatformRoster(usvs=(), uuvs=(first, second))


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (MotionLimits, "max_speed_mps", inf),
        (SonarCapability, "passive_range_m", 0.0),
        (CommunicationCapability, "acoustic_range_m", -1.0),
    ],
)
def test_platform_capabilities_reject_non_finite_or_non_positive_values(
    model: type, field: str, value: float
) -> None:
    valid = {
        MotionLimits: {
            "max_speed_mps": 4.0,
            "max_acceleration_mps2": 0.1,
            "max_turn_rate_rad_s": 0.02,
        },
        SonarCapability: {
            "passive_range_m": 4000.0,
            "passive_bearing_variance_rad2": 0.01,
            "active_source_range_m": 3000.0,
            "active_receive_range_m": 4000.0,
            "active_range_sigma_m": 10.0,
            "active_bearing_sigma_rad": 0.003,
            "active_capable": True,
            "ping_cooldown_s": 30,
            "ping_energy_cost_fraction": 0.001,
            "clutter_sensitivity": 0.2,
            "exposure_cost": 0.3,
        },
        CommunicationCapability: {
            "surface_range_m": 10000.0,
            "acoustic_range_m": 4000.0,
        },
    }[model]
    with pytest.raises(ValidationError):
        model.model_validate({**valid, field: value})
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/domain/test_platform_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'underwater_tracking.domain.platforms'`.

- [ ] **Step 3: Implement the platform contracts**

Create `src/underwater_tracking/domain/platforms.py` with these exact public models:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from underwater_tracking.domain.models import StrictModel

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PlatformKind(StrEnum):
    USV = "usv"
    UUV = "uuv"


class MotionLimits(StrictModel):
    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_turn_rate_rad_s: PositiveFloat


class SonarCapability(StrictModel):
    passive_range_m: PositiveFloat
    passive_bearing_variance_rad2: PositiveFloat
    active_source_range_m: PositiveFloat
    active_receive_range_m: PositiveFloat
    active_range_sigma_m: PositiveFloat
    active_bearing_sigma_rad: PositiveFloat
    active_capable: bool
    ping_cooldown_s: int = Field(gt=0)
    ping_energy_cost_fraction: UnitFloat
    clutter_sensitivity: UnitFloat
    exposure_cost: UnitFloat


class CommunicationCapability(StrictModel):
    surface_range_m: PositiveFloat
    acoustic_range_m: PositiveFloat


class PlatformCapability(StrictModel):
    kind: PlatformKind
    motion: MotionLimits
    sonar: SonarCapability
    communications: CommunicationCapability


class MobilePlatformState(StrictModel):
    platform_id: str = Field(min_length=1)
    platform_index: int = Field(ge=0)
    position_xy: tuple[float, float]
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: NonNegativeFloat
    energy_fraction: UnitFloat
    deployment_state: Literal["onboard", "deployed", "returning", "failed"]
    capability: PlatformCapability
    group_id: str | None = None
    sensor_mode: Literal["passive", "active"] = "passive"


class USVPlatformState(MobilePlatformState):
    distance_to_carrier_m: NonNegativeFloat

    @model_validator(mode="after")
    def kind_is_usv(self) -> USVPlatformState:
        if self.capability.kind is not PlatformKind.USV:
            raise ValueError("USV state requires a USV capability")
        return self


class UUVPlatformState(MobilePlatformState):
    is_group_leader: bool = False
    master_connected: bool = False

    @model_validator(mode="after")
    def kind_is_uuv(self) -> UUVPlatformState:
        if self.capability.kind is not PlatformKind.UUV:
            raise ValueError("UUV state requires a UUV capability")
        return self


class CarrierPlatformState(StrictModel):
    carrier_id: str = Field(min_length=1)
    position_xy: tuple[float, float]
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: NonNegativeFloat
    support_radius_m: PositiveFloat
    onboard_platform_ids: tuple[str, ...]
    deployed_platform_ids: tuple[str, ...]
    returning_platform_ids: tuple[str, ...]

    @model_validator(mode="after")
    def relationship_lists_are_disjoint(self) -> CarrierPlatformState:
        relationships = (
            self.onboard_platform_ids,
            self.deployed_platform_ids,
            self.returning_platform_ids,
        )
        if any(len(values) != len(set(values)) for values in relationships):
            raise ValueError("carrier platform relationship lists must be unique and disjoint")
        groups = tuple(set(values) for values in relationships)
        if any(
            left & right
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("carrier platform relationship lists must be unique and disjoint")
        return self


class PlatformRoster(StrictModel):
    usvs: tuple[USVPlatformState, ...]
    uuvs: tuple[UUVPlatformState, ...]

    @model_validator(mode="after")
    def platform_ids_are_unique(self) -> PlatformRoster:
        ids = [platform.platform_id for platform in (*self.usvs, *self.uuvs)]
        if len(ids) != len(set(ids)):
            raise ValueError("platform IDs must be unique")
        for kind, platforms in (("USV", self.usvs), ("UUV", self.uuvs)):
            indices = [platform.platform_index for platform in platforms]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{kind} platform indices must be unique")
        return self


class CommunicationLink(StrictModel):
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    medium: Literal["surface", "acoustic"]
    distance_m: NonNegativeFloat


class PlatformSnapshot(StrictModel):
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    carrier: CarrierPlatformState
    roster: PlatformRoster
    communication_links: tuple[CommunicationLink, ...]
```

- [ ] **Step 4: Run focused domain tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/domain/test_platform_contracts.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Run domain regression and static checks**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/domain -q
PYTHONPATH=src .venv/bin/python -m ruff check src/underwater_tracking/domain/platforms.py tests/domain/test_platform_contracts.py
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking/domain/platforms.py
```

Expected: all commands exit `0`.

- [ ] **Step 6: Commit only Task 1 files**

```bash
git add src/underwater_tracking/domain/platforms.py tests/domain/test_platform_contracts.py
git commit -m "feat: add unified platform contracts"
```

---

### Task 2: Load the Explicit Platform-Core Configuration

**Files:**
- Create: `src/underwater_tracking/config/platform_core.py`
- Create: `configs/scenario/segmented_single_target.yaml`
- Create: `configs/environment.yaml`
- Create: `configs/platforms.yaml`
- Create: `configs/sensors.yaml`
- Create: `configs/communications.yaml`
- Create: `tests/config/test_platform_core_loader.py`
- Modify: `src/underwater_tracking/config/models.py:26-153`
- Modify: `src/underwater_tracking/config/loader.py:22-49`

**Interfaces:**
- Consumes: Task 1 `PlatformKind`.
- Produces: `PlatformCoreFiles`, `EnvironmentConfig`, `PlatformCatalogConfig`, `SensorCatalogConfig`, `CommunicationsConfig`, and optional `AppConfig.environment/platforms/sensors/communications` fields.

- [ ] **Step 1: Write failing loader tests**

Create `tests/config/test_platform_core_loader.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.config.loader import load_app_config


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


def test_explicit_platform_core_roster_loads() -> None:
    config = load_app_config(SCENARIO)

    assert config.scenario.scenario_id == "segmented-single-target"
    assert config.environment is not None
    assert config.platforms is not None
    assert config.sensors is not None
    assert config.communications is not None
    assert config.environment.carrier.platform_id == "carrier_01"
    assert len(config.environment.usvs) == 4
    assert len(config.environment.uuvs) == 12
    assert len(config.environment.submarines) == 1
    assert config.environment.decoys == ()


def test_every_roster_entry_resolves_capability_profiles() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    assert config.platforms is not None
    assert config.sensors is not None
    assert config.communications is not None

    for platform in (*config.environment.usvs, *config.environment.uuvs):
        assert platform.motion_profile in config.platforms.motion_profiles
        assert platform.sensor_profile in config.sensors.profiles
        assert platform.communication_profile in config.communications.profiles
    for submarine in config.environment.submarines:
        assert submarine.motion_profile in config.platforms.motion_profiles


def test_referenced_config_path_cannot_escape_configs(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    scenario_dir = config_root / "scenario"
    scenario_dir.mkdir(parents=True)
    (config_root / "tracking.yaml").write_text("group_min_size: 2\ngroup_max_size: 4\n", encoding="utf-8")
    scenario = scenario_dir / "bad.yaml"
    scenario.write_text(
        "scenario:\n"
        "  scenario_id: bad\n"
        "  duration_s: 60\n"
        "  seed: 1\n"
        "  platform_core:\n"
        "    environment: ../../outside.yaml\n"
        "    platforms: platforms.yaml\n"
        "    sensors: sensors.yaml\n"
        "    communications: communications.yaml\n"
        "timing:\n"
        "  physics_step_s: 10\n"
        "  observation_step_s: 30\n"
        "  group_report_s: 300\n"
        "  progress_report_s: 600\n"
        "  strategic_review_s: 900\n"
        "  prediction_horizon_s: 1800\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must stay below config root"):
        load_app_config(scenario)


def test_explicit_environment_rejects_duplicate_platform_ids() -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    duplicate = config.environment.model_dump()
    duplicate["uuvs"][0]["platform_id"] = duplicate["usvs"][0]["platform_id"]

    with pytest.raises(ValidationError, match="platform IDs must be unique"):
        type(config.environment).model_validate(duplicate)
```

- [ ] **Step 2: Run loader tests and verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/config/test_platform_core_loader.py -q
```

Expected: collection fails because `segmented_single_target.yaml` and `config.platform_core` do not exist.

- [ ] **Step 3: Add strict platform-core config models**

Create `src/underwater_tracking/config/platform_core.py` with these public contracts:

```python
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from underwater_tracking.domain.platforms import PlatformKind

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlatformCoreFiles(StrictConfig):
    environment: str = Field(min_length=1)
    platforms: str = Field(min_length=1)
    sensors: str = Field(min_length=1)
    communications: str = Field(min_length=1)


class RegionConfig(StrictConfig):
    region_id: str = Field(min_length=1)
    polygon_xy: tuple[tuple[float, float], ...] = Field(min_length=3)


class InitialPlatformConfig(StrictConfig):
    platform_id: str = Field(min_length=1)
    platform_index: int = Field(ge=0)
    kind: PlatformKind
    position_xy: tuple[float, float]
    heading_rad: float = Field(allow_inf_nan=False)
    energy_fraction: UnitFloat
    deployment_state: Literal["onboard", "deployed"]
    motion_profile: str = Field(min_length=1)
    sensor_profile: str = Field(min_length=1)
    communication_profile: str = Field(min_length=1)


class CarrierInitialConfig(StrictConfig):
    platform_id: str = Field(min_length=1)
    position_xy: tuple[float, float]
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: float = Field(ge=0, allow_inf_nan=False)
    support_radius_m: PositiveFloat
    patrol_route_xy: tuple[tuple[float, float], ...] = Field(min_length=2)


class SubmarineInitialConfig(StrictConfig):
    target_id: str = Field(min_length=1)
    position_xy: tuple[float, float]
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: PositiveFloat
    motion_profile: str = Field(min_length=1)
    task_region_id: str = Field(min_length=1)
    escape_region_ids: tuple[str, ...] = Field(min_length=1)


class EnvironmentConfig(StrictConfig):
    map_bounds_xy: tuple[float, float, float, float]
    carrier: CarrierInitialConfig
    usvs: tuple[InitialPlatformConfig, ...]
    uuvs: tuple[InitialPlatformConfig, ...]
    submarines: tuple[SubmarineInitialConfig, ...]
    decoys: tuple[str, ...]
    task_regions: tuple[RegionConfig, ...]
    escape_regions: tuple[RegionConfig, ...]

    @model_validator(mode="after")
    def validate_roster(self) -> EnvironmentConfig:
        if len(self.usvs) != 4 or len(self.uuvs) != 12 or len(self.submarines) != 1:
            raise ValueError("explicit scenario requires 4 USVs, 12 UUVs, and 1 submarine")
        if self.decoys:
            raise ValueError("explicit single-target scenario does not allow decoys")
        platforms = (*self.usvs, *self.uuvs)
        ids = [self.carrier.platform_id, *(platform.platform_id for platform in platforms)]
        if len(ids) != len(set(ids)):
            raise ValueError("platform IDs must be unique")
        if any(platform.kind is not PlatformKind.USV for platform in self.usvs):
            raise ValueError("usvs must contain only USV entries")
        if any(platform.kind is not PlatformKind.UUV for platform in self.uuvs):
            raise ValueError("uuvs must contain only UUV entries")
        return self


class MotionProfileConfig(StrictConfig):
    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_turn_rate_rad_s: PositiveFloat
    transit_energy_per_m: PositiveFloat
    hotel_energy_per_s: PositiveFloat


class PlatformCatalogConfig(StrictConfig):
    motion_profiles: dict[str, MotionProfileConfig]


class SensorProfileConfig(StrictConfig):
    passive_range_m: PositiveFloat
    passive_bearing_variance_rad2: PositiveFloat
    active_source_range_m: PositiveFloat
    active_receive_range_m: PositiveFloat
    active_range_sigma_m: PositiveFloat
    active_bearing_sigma_rad: PositiveFloat
    active_capable: bool
    ping_cooldown_s: int = Field(gt=0)
    ping_energy_cost_fraction: UnitFloat
    clutter_sensitivity: UnitFloat
    exposure_cost: UnitFloat


class SensorCatalogConfig(StrictConfig):
    profiles: dict[str, SensorProfileConfig]


class CommunicationProfileConfig(StrictConfig):
    surface_range_m: PositiveFloat
    acoustic_range_m: PositiveFloat


class CommunicationsConfig(StrictConfig):
    profiles: dict[str, CommunicationProfileConfig]
```

- [ ] **Step 4: Extend `ScenarioConfig` and `AppConfig`**

Modify `src/underwater_tracking/config/models.py` imports and models:

```python
from underwater_tracking.config.platform_core import (
    CommunicationsConfig,
    EnvironmentConfig,
    PlatformCatalogConfig,
    PlatformCoreFiles,
    SensorCatalogConfig,
)


class ScenarioConfig(StrictModel):
    scenario_id: str = "underwater-default"
    uuv_count: int = Field(12, ge=2)
    initial_target_count: int = Field(2, ge=1)
    max_target_count: int = Field(4, ge=1)
    duration_s: int = Field(28_800, gt=0)
    seed: int = 42
    initial_decoy_count: int = Field(default=0, ge=0)
    operational_scheme: OperationalScheme | None = None
    platform_core: PlatformCoreFiles | None = None


class AppConfig(StrictModel):
    scenario: ScenarioConfig
    timing: TimingConfig
    tracking: TrackingConfig
    agent: AgentConfig | None = None
    llm: LLMConfig | None = None
    environment: EnvironmentConfig | None = None
    platforms: PlatformCatalogConfig | None = None
    sensors: SensorCatalogConfig | None = None
    communications: CommunicationsConfig | None = None

    @model_validator(mode="after")
    def platform_core_is_complete(self) -> AppConfig:
        loaded = (self.environment, self.platforms, self.sensors, self.communications)
        if self.scenario.platform_core is None and all(value is None for value in loaded):
            return self
        if self.scenario.platform_core is None or any(value is None for value in loaded):
            raise ValueError("platform_core references and all loaded sections are required together")
        assert self.environment is not None
        assert self.platforms is not None
        assert self.sensors is not None
        assert self.communications is not None
        if self.scenario.uuv_count != len(self.environment.uuvs):
            raise ValueError("scenario uuv_count must equal explicit UUV roster size")
        if self.scenario.initial_target_count != len(self.environment.submarines):
            raise ValueError("scenario initial_target_count must equal explicit submarine roster size")
        if self.scenario.max_target_count != len(self.environment.submarines):
            raise ValueError("single-target max_target_count must equal explicit submarine roster size")
        for platform in (*self.environment.usvs, *self.environment.uuvs):
            if platform.motion_profile not in self.platforms.motion_profiles:
                raise ValueError(f"unknown motion profile {platform.motion_profile!r}")
            if platform.sensor_profile not in self.sensors.profiles:
                raise ValueError(f"unknown sensor profile {platform.sensor_profile!r}")
            if platform.communication_profile not in self.communications.profiles:
                raise ValueError(
                    f"unknown communication profile {platform.communication_profile!r}"
                )
        for submarine in self.environment.submarines:
            if submarine.motion_profile not in self.platforms.motion_profiles:
                raise ValueError(f"unknown submarine motion profile {submarine.motion_profile!r}")
        return self
```

Retain the current count fields only for the legacy scenario. The engine's new path must use the explicit roster, never those count fields.

- [ ] **Step 5: Resolve referenced YAML below the config root**

Add to `src/underwater_tracking/config/loader.py`:

```python
def _load_referenced_yaml(config_root: Path, relative_path: str) -> object:
    root = config_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"referenced config path {relative_path!r} must stay below config root")
    if not candidate.is_file():
        raise ValueError(f"referenced config file {relative_path!r} does not exist")
    return yaml.safe_load(candidate.read_text(encoding="utf-8"))
```

After loading `scenario_data`, insert this block before `AppConfig.model_validate(data)`:

```python
    scenario_section = scenario_data.get("scenario", {})
    file_refs = scenario_section.get("platform_core")
    if file_refs is not None:
        for section in ("environment", "platforms", "sensors", "communications"):
            relative_path = file_refs[section]
            data[section] = _load_referenced_yaml(config_root, relative_path)
```

- [ ] **Step 6: Add the explicit YAML files**

Create `configs/scenario/segmented_single_target.yaml`:

```yaml
scenario:
  scenario_id: segmented-single-target
  uuv_count: 12
  initial_target_count: 1
  max_target_count: 1
  duration_s: 28800
  seed: 42
  initial_decoy_count: 0
  platform_core:
    environment: environment.yaml
    platforms: platforms.yaml
    sensors: sensors.yaml
    communications: communications.yaml
timing:
  physics_step_s: 10
  observation_step_s: 30
  group_report_s: 300
  progress_report_s: 600
  strategic_review_s: 900
  prediction_horizon_s: 1800
```

Create `configs/environment.yaml` with the full roster. Use these exact IDs and make all UUVs initially onboard so later plans can demonstrate carrier launch:

```yaml
map_bounds_xy: [-12000.0, 12000.0, -12000.0, 12000.0]
carrier:
  platform_id: carrier_01
  position_xy: [-8000.0, -8000.0]
  heading_rad: 0.0
  speed_mps: 4.0
  support_radius_m: 16000.0
  patrol_route_xy:
    - [-8000.0, -8000.0]
    - [8000.0, -8000.0]
    - [8000.0, 8000.0]
    - [-8000.0, 8000.0]
usvs:
  - {platform_id: usv_00, platform_index: 0, kind: usv, position_xy: [-7600.0, -7800.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: deployed, motion_profile: usv_standard, sensor_profile: usv_multistatic, communication_profile: usv_relay}
  - {platform_id: usv_01, platform_index: 1, kind: usv, position_xy: [-8000.0, -7400.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: deployed, motion_profile: usv_standard, sensor_profile: usv_multistatic, communication_profile: usv_relay}
  - {platform_id: usv_02, platform_index: 2, kind: usv, position_xy: [-8400.0, -7800.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: deployed, motion_profile: usv_standard, sensor_profile: usv_multistatic, communication_profile: usv_relay}
  - {platform_id: usv_03, platform_index: 3, kind: usv, position_xy: [-8000.0, -8400.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: deployed, motion_profile: usv_standard, sensor_profile: usv_multistatic, communication_profile: usv_relay}
uuvs:
  - {platform_id: uuv_00, platform_index: 0, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_01, platform_index: 1, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_02, platform_index: 2, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_03, platform_index: 3, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_04, platform_index: 4, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_05, platform_index: 5, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_06, platform_index: 6, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_07, platform_index: 7, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_08, platform_index: 8, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_09, platform_index: 9, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_10, platform_index: 10, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
  - {platform_id: uuv_11, platform_index: 11, kind: uuv, position_xy: [-8000.0, -8000.0], heading_rad: 0.0, energy_fraction: 1.0, deployment_state: onboard, motion_profile: uuv_standard, sensor_profile: uuv_dual_sonar, communication_profile: uuv_acoustic}
submarines:
  - target_id: target_00
    position_xy: [-4500.0, -4500.0]
    heading_rad: 0.0
    speed_mps: 8.0
    motion_profile: submarine_standard
    task_region_id: mission_east
    escape_region_ids: [escape_north, escape_south]
decoys: []
task_regions:
  - region_id: mission_east
    polygon_xy: [[7000.0, -1500.0], [10000.0, -1500.0], [10000.0, 1500.0], [7000.0, 1500.0]]
escape_regions:
  - region_id: escape_north
    polygon_xy: [[2000.0, 7000.0], [5000.0, 7000.0], [5000.0, 10000.0], [2000.0, 10000.0]]
  - region_id: escape_south
    polygon_xy: [[2000.0, -10000.0], [5000.0, -10000.0], [5000.0, -7000.0], [2000.0, -7000.0]]
```

Create `configs/platforms.yaml`:

```yaml
motion_profiles:
  usv_standard: {max_speed_mps: 8.0, max_acceleration_mps2: 0.25, max_turn_rate_rad_s: 0.035, transit_energy_per_m: 0.0000008, hotel_energy_per_s: 0.00000005}
  uuv_standard: {max_speed_mps: 4.0, max_acceleration_mps2: 0.10, max_turn_rate_rad_s: 0.05235987755982988, transit_energy_per_m: 0.000002, hotel_energy_per_s: 0.0000001}
  submarine_standard: {max_speed_mps: 14.0, max_acceleration_mps2: 0.08, max_turn_rate_rad_s: 0.010471975511965976, transit_energy_per_m: 0.0000001, hotel_energy_per_s: 0.00000001}
```

Create `configs/sensors.yaml`:

```yaml
profiles:
  usv_multistatic: {passive_range_m: 5500.0, passive_bearing_variance_rad2: 0.009, active_source_range_m: 5000.0, active_receive_range_m: 6000.0, active_range_sigma_m: 15.0, active_bearing_sigma_rad: 0.003, active_capable: true, ping_cooldown_s: 30, ping_energy_cost_fraction: 0.0002, clutter_sensitivity: 0.20, exposure_cost: 0.35}
  uuv_dual_sonar: {passive_range_m: 4500.0, passive_bearing_variance_rad2: 0.010, active_source_range_m: 3500.0, active_receive_range_m: 4500.0, active_range_sigma_m: 18.0, active_bearing_sigma_rad: 0.004, active_capable: true, ping_cooldown_s: 30, ping_energy_cost_fraction: 0.0003, clutter_sensitivity: 0.25, exposure_cost: 0.60}
```

Create `configs/communications.yaml`:

```yaml
profiles:
  usv_relay: {surface_range_m: 12000.0, acoustic_range_m: 5000.0}
  uuv_acoustic: {surface_range_m: 1.0, acoustic_range_m: 4000.0}
```

- [ ] **Step 7: Run focused and legacy loader tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/config/test_platform_core_loader.py tests/config/test_loader.py tests/agent/test_agent_loader.py -q
```

Expected: all tests pass; the legacy loader assertions remain valid.

- [ ] **Step 8: Run static checks and commit**

```bash
PYTHONPATH=src .venv/bin/python -m ruff check src/underwater_tracking/config tests/config/test_platform_core_loader.py
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking/config
git add src/underwater_tracking/config/models.py src/underwater_tracking/config/loader.py src/underwater_tracking/config/platform_core.py configs/scenario/segmented_single_target.yaml configs/environment.yaml configs/platforms.yaml configs/sensors.yaml configs/communications.yaml tests/config/test_platform_core_loader.py
git commit -m "feat: load explicit platform core configuration"
```

Expected: Ruff and Mypy exit `0`; commit contains only listed files.

---

### Task 3: Introduce Shared Continuous 2-D Kinematics

**Files:**
- Create: `src/underwater_tracking/simulation/kinematics.py`
- Create: `src/underwater_tracking/simulation/usv.py`
- Modify: `src/underwater_tracking/simulation/uuv.py:1-35`
- Modify: `src/underwater_tracking/simulation/carrier.py:15-80`
- Modify: `src/underwater_tracking/simulation/target.py:100-171`
- Modify: `tests/simulation/test_kinematics.py`
- Test: `tests/simulation/test_carrier.py`

**Interfaces:**
- Consumes: Task 1 `MotionLimits`.
- Produces: `MotionState`, `MotionCommand`, `advance_motion(state, command, limits, dt_s)`, and `USVEntity`.

- [ ] **Step 1: Add failing bounded-motion tests**

Append to `tests/simulation/test_kinematics.py`:

```python
from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import MotionCommand, MotionState, advance_motion
from underwater_tracking.simulation.usv import USVEntity


LIMITS = MotionLimits(
    max_speed_mps=8.0,
    max_acceleration_mps2=0.2,
    max_turn_rate_rad_s=0.03,
)


def test_shared_motion_limits_acceleration_and_turn_rate() -> None:
    start = MotionState(position_xy=(0.0, 0.0), heading_rad=0.0, speed_mps=2.0)
    command = MotionCommand(desired_heading_rad=1.0, desired_speed_mps=8.0)

    end = advance_motion(start, command, LIMITS, dt_s=10.0)

    assert end.speed_mps == pytest.approx(4.0)
    assert end.heading_rad == pytest.approx(0.3)
    assert end.position_xy[0] == pytest.approx(4.0 * 10.0 * cos(0.3))
    assert end.position_xy[1] == pytest.approx(4.0 * 10.0 * sin(0.3))


def test_usv_entity_uses_shared_motion_and_monotonic_energy() -> None:
    usv = USVEntity(
        usv_id="usv_00",
        platform_index=0,
        motion=MotionState(position_xy=(0.0, 0.0), heading_rad=0.0, speed_mps=0.0),
        energy_fraction=1.0,
        limits=LIMITS,
        transit_energy_per_m=8e-7,
        hotel_energy_per_s=5e-8,
    )
    usv.set_motion_command(MotionCommand(desired_heading_rad=0.0, desired_speed_mps=6.0))

    usv.step(10.0)

    assert 0.0 < usv.motion.speed_mps <= 2.0
    assert usv.motion.position_xy[0] > 0.0
    assert 0.0 < usv.energy_fraction < 1.0


def test_target_intent_change_no_longer_instantly_rotates_velocity() -> None:
    target = TargetEntity(
        "T1",
        (0.0, 0.0),
        (8.0, 0.0),
        HiddenIntent.TRANSIT,
        intent_speed_mps={intent: 8.0 for intent in HiddenIntent},
        max_acceleration_mps2=0.08,
        max_turn_rate_rad_s=0.01,
    )
    target.apply_evasive_maneuver(pi / 2)

    target.step(10.0, random.Random(3))

    heading = atan2(target.velocity_xy[1], target.velocity_xy[0])
    assert 0.0 < heading <= 0.1
```

Add imports for `atan2`, `cos`, `sin`, and `pytest` if they are not already present.

- [ ] **Step 2: Run the focused tests and verify missing interfaces**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/simulation/test_kinematics.py -q
```

Expected: collection fails because `simulation.kinematics` and `simulation.usv` do not exist.

- [ ] **Step 3: Implement the shared integrator**

Create `src/underwater_tracking/simulation/kinematics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from underwater_tracking.domain.platforms import MotionLimits


def wrap_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi


@dataclass(frozen=True, slots=True)
class MotionState:
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True, slots=True)
class MotionCommand:
    desired_heading_rad: float
    desired_speed_mps: float


def advance_motion(
    state: MotionState,
    command: MotionCommand,
    limits: MotionLimits,
    dt_s: float,
) -> MotionState:
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")
    desired_speed = min(limits.max_speed_mps, max(0.0, command.desired_speed_mps))
    max_speed_delta = limits.max_acceleration_mps2 * dt_s
    speed_delta = max(-max_speed_delta, min(max_speed_delta, desired_speed - state.speed_mps))
    speed = min(limits.max_speed_mps, max(0.0, state.speed_mps + speed_delta))
    max_heading_delta = limits.max_turn_rate_rad_s * dt_s
    heading_error = wrap_angle(command.desired_heading_rad - state.heading_rad)
    heading_delta = max(-max_heading_delta, min(max_heading_delta, heading_error))
    heading = wrap_angle(state.heading_rad + heading_delta)
    distance = speed * dt_s
    return MotionState(
        position_xy=(
            state.position_xy[0] + distance * cos(heading),
            state.position_xy[1] + distance * sin(heading),
        ),
        heading_rad=heading,
        speed_mps=speed,
    )
```

- [ ] **Step 4: Implement `USVEntity`**

Create `src/underwater_tracking/simulation/usv.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import MotionCommand, MotionState, advance_motion


@dataclass(slots=True)
class USVEntity:
    usv_id: str
    platform_index: int
    motion: MotionState
    energy_fraction: float
    limits: MotionLimits
    transit_energy_per_m: float
    hotel_energy_per_s: float
    command: MotionCommand | None = None

    def set_motion_command(self, command: MotionCommand) -> None:
        self.command = command

    def step(self, dt_s: float) -> None:
        command = self.command or MotionCommand(
            desired_heading_rad=self.motion.heading_rad,
            desired_speed_mps=0.0,
        )
        before = self.motion.position_xy
        self.motion = advance_motion(self.motion, command, self.limits, dt_s)
        dx = self.motion.position_xy[0] - before[0]
        dy = self.motion.position_xy[1] - before[1]
        distance = (dx * dx + dy * dy) ** 0.5
        used = distance * self.transit_energy_per_m + dt_s * self.hotel_energy_per_s
        self.energy_fraction = max(0.0, self.energy_fraction - used)
```

- [ ] **Step 5: Adapt UUV, carrier, and target without breaking public constructors**

In `simulation/uuv.py`, retain `UUVEntity` fields and `set_waypoints`, append `platform_index` and `speed_mps` defaults so existing positional calls remain valid, then delegate motion to the shared integrator:

```python
    platform_index: int = 0
    speed_mps: float = 0.0

    def step(
        self,
        dt_s: float,
        max_speed_mps: float,
        max_turn_rate_rad_s: float,
        max_acceleration_mps2: float | None = None,
    ) -> None:
        if not self.waypoints or self.energy_fraction <= 0:
            self.speed_mps = 0.0
            return
        wx, wy = self.waypoints[0]
        desired = atan2(wy - self.position_xy[1], wx - self.position_xy[0])
        limits = MotionLimits(
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=max_acceleration_mps2 or max_speed_mps,
            max_turn_rate_rad_s=max_turn_rate_rad_s,
        )
        end = advance_motion(
            MotionState(self.position_xy, self.heading_rad, self.speed_mps),
            MotionCommand(desired, max_speed_mps),
            limits,
            dt_s,
        )
        distance = hypot(end.position_xy[0] - self.position_xy[0], end.position_xy[1] - self.position_xy[1])
        self.position_xy = end.position_xy
        self.heading_rad = end.heading_rad
        self.speed_mps = end.speed_mps
        self.energy_fraction = max(0.0, self.energy_fraction - distance * 2e-6 - dt_s * 1e-7)
        if hypot(wx - self.position_xy[0], wy - self.position_xy[1]) < 1.0:
            self.waypoints.pop(0)
```

In `simulation/carrier.py`, change `CarrierEntity.__init__` to accept optional explicit values while preserving current defaults:

```python
    def __init__(
        self,
        *,
        carrier_id: str = _CARRIER_ID,
        position_xy: tuple[float, float] = _PATROL_CORNERS[0],
        speed_mps: float = _PATROL_SPEED_MPS,
        patrol_route_xy: tuple[tuple[float, float], ...] = _PATROL_CORNERS,
        support_radius_m: float = 16000.0,
    ) -> None:
        if len(patrol_route_xy) < 2:
            raise ValueError("carrier patrol route requires at least two points")
        self.carrier_id = carrier_id
        self.position_xy = position_xy
        self.speed_mps = speed_mps
        self.support_radius_m = support_radius_m
        self._patrol_route_xy = patrol_route_xy
        self._next_corner_index = 1
        self.heading_rad = self._heading_to_next_corner()
```

Replace `_PATROL_CORNERS` lookups in `step` and `_heading_to_next_corner` with `self._patrol_route_xy`, and emit `self.carrier_id` in `state_for`.

In `simulation/target.py`, import `field`, `MotionLimits`, and the shared kinematics types, then replace `TargetEntity` with this constructor-compatible bounded implementation:

```python
@dataclass(slots=True)
class TargetEntity:
    target_id: str
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    intent: HiddenIntent
    bounds_xy: tuple[float, float, float, float] = DEFAULT_BOUNDS_XY
    intent_speed_mps: dict[HiddenIntent, float] | None = None
    max_speed_mps: float = 14.0
    max_acceleration_mps2: float = 0.08
    max_turn_rate_rad_s: float = math.pi / 300.0
    _desired_heading_rad: float = field(init=False, repr=False)
    _desired_speed_mps: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._desired_heading_rad = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self._desired_speed_mps = math.hypot(*self.velocity_xy)

    def step(self, dt_s: float, rng: random.Random) -> None:
        next_intent = self._sample_intent(rng)
        if next_intent is not self.intent:
            self.intent = next_intent
            direction = INTENT_VELOCITIES[next_intent]
            self._desired_heading_rad = math.atan2(direction[1], direction[0])
            self._desired_speed_mps = self._intent_speed(next_intent)
        current_speed = math.hypot(*self.velocity_xy)
        current_heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        limits = MotionLimits(
            max_speed_mps=self.max_speed_mps,
            max_acceleration_mps2=self.max_acceleration_mps2,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
        )
        end = advance_motion(
            MotionState(self.position_xy, current_heading, current_speed),
            MotionCommand(self._desired_heading_rad, self._desired_speed_mps),
            limits,
            dt_s,
        )
        self.position_xy = end.position_xy
        self.velocity_xy = (
            end.speed_mps * math.cos(end.heading_rad),
            end.speed_mps * math.sin(end.heading_rad),
        )
        self._reflect_into_bounds()

    def apply_evasive_maneuver(self, turn_angle_rad: float) -> None:
        current_heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0])
        self.intent = HiddenIntent.EVADE
        self._desired_heading_rad = wrap_angle(current_heading + turn_angle_rad)
        self._desired_speed_mps = self._intent_speed(HiddenIntent.EVADE)

    def _scaled_velocity(self, intent: HiddenIntent, heading: float) -> tuple[float, float]:
        speed = self._intent_speed(intent)
        return (speed * math.cos(heading), speed * math.sin(heading))

    def _intent_velocity(self, intent: HiddenIntent) -> tuple[float, float]:
        dx, dy = INTENT_VELOCITIES[intent]
        scale = self._intent_speed(intent) / max(math.hypot(dx, dy), 1e-9)
        return (dx * scale, dy * scale)

    def _intent_speed(self, intent: HiddenIntent) -> float:
        if self.intent_speed_mps is None:
            return math.hypot(*INTENT_VELOCITIES[intent])
        return self.intent_speed_mps.get(intent, 8.0)

    def public_kinematics(self) -> dict[str, object]:
        return {"target_id": self.target_id}

    def _sample_intent(self, rng: random.Random) -> HiddenIntent:
        row = TRANSITION_PROBABILITIES[self.intent]
        draw = rng.random()
        cumulative = 0.0
        for next_intent, probability in row.items():
            cumulative += probability
            if draw < cumulative:
                return next_intent
        return self.intent

    def _reflect_into_bounds(self) -> None:
        x, y = self.position_xy
        vx, vy = self.velocity_xy
        x_min, x_max, y_min, y_max = self.bounds_xy
        if x < x_min:
            x, vx = x_min + (x_min - x), -vx
        elif x > x_max:
            x, vx = x_max - (x - x_max), -vx
        if y < y_min:
            y, vy = y_min + (y_min - y), -vy
        elif y > y_max:
            y, vy = y_max - (y - y_max), -vy
        self.position_xy = (x, y)
        self.velocity_xy = (vx, vy)
        reflected_heading = math.atan2(vy, vx)
        self._desired_heading_rad = reflected_heading
```

Use `wrap_angle` from `simulation.kinematics`; retain constants and Markov transition tables unchanged. The explicit engine path passes the configured submarine acceleration and turn-rate limits, while legacy constructors continue to use these defaults.

- [ ] **Step 6: Run kinematics and carrier tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/simulation/test_kinematics.py tests/simulation/test_carrier.py -q
```

Expected: all tests pass, including the new bounded-motion tests.

- [ ] **Step 7: Run simulation regression and static checks**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/simulation -m 'not real_llm' -q
PYTHONPATH=src .venv/bin/python -m ruff check src/underwater_tracking/simulation tests/simulation/test_kinematics.py
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking/simulation
```

Expected: all commands exit `0`.

- [ ] **Step 8: Commit Task 3 files**

```bash
git add src/underwater_tracking/simulation/kinematics.py src/underwater_tracking/simulation/usv.py src/underwater_tracking/simulation/uuv.py src/underwater_tracking/simulation/carrier.py src/underwater_tracking/simulation/target.py tests/simulation/test_kinematics.py
git commit -m "feat: add shared continuous platform kinematics"
```

---

### Task 4: Build the Distance-Only Communication Graph

**Files:**
- Create: `src/underwater_tracking/simulation/connectivity.py`
- Create: `tests/simulation/test_connectivity.py`

**Interfaces:**
- Consumes: Task 1 `CommunicationLink`, `PlatformKind`, and platform capabilities.
- Produces: `ConnectivityNode`, `ConnectivitySnapshot`, `build_connectivity(carrier_id, carrier_xy, nodes)`, and `has_path(snapshot, source_id, target_id)`.

- [ ] **Step 1: Write failing connectivity tests**

Create `tests/simulation/test_connectivity.py`:

```python
from underwater_tracking.domain.platforms import PlatformKind
from underwater_tracking.simulation.connectivity import (
    ConnectivityNode,
    build_connectivity,
    has_path,
)


def node(
    platform_id: str,
    kind: PlatformKind,
    position_xy: tuple[float, float],
    surface_range_m: float,
    acoustic_range_m: float,
) -> ConnectivityNode:
    return ConnectivityNode(
        platform_id=platform_id,
        kind=kind,
        position_xy=position_xy,
        surface_range_m=surface_range_m,
        acoustic_range_m=acoustic_range_m,
    )


def test_usv_mesh_connects_carrier_to_uuv_over_multiple_hops() -> None:
    snapshot = build_connectivity(
        carrier_id="carrier_01",
        carrier_xy=(0.0, 0.0),
        nodes=(
            node("usv_00", PlatformKind.USV, (5000.0, 0.0), 6000.0, 2500.0),
            node("usv_01", PlatformKind.USV, (10000.0, 0.0), 6000.0, 2500.0),
            node("uuv_11", PlatformKind.UUV, (11000.0, 1000.0), 1.0, 2500.0),
        ),
    )

    assert has_path(snapshot, "carrier_01", "uuv_11")
    assert [(link.source_id, link.target_id, link.medium) for link in snapshot.links] == [
        ("carrier_01", "usv_00", "surface"),
        ("usv_00", "usv_01", "surface"),
        ("usv_01", "uuv_11", "acoustic"),
    ]


def test_distance_break_disconnects_group_leader() -> None:
    snapshot = build_connectivity(
        carrier_id="carrier_01",
        carrier_xy=(0.0, 0.0),
        nodes=(
            node("usv_00", PlatformKind.USV, (7000.0, 0.0), 6000.0, 2000.0),
            node("uuv_11", PlatformKind.UUV, (7000.0, 1000.0), 1.0, 2000.0),
        ),
    )

    assert not has_path(snapshot, "carrier_01", "uuv_11")
    assert [(link.source_id, link.target_id, link.medium) for link in snapshot.links] == [
        ("usv_00", "uuv_11", "acoustic")
    ]


def test_uuv_to_uuv_uses_acoustic_range() -> None:
    snapshot = build_connectivity(
        carrier_id="carrier_01",
        carrier_xy=(10000.0, 10000.0),
        nodes=(
            node("uuv_10", PlatformKind.UUV, (0.0, 0.0), 1.0, 1500.0),
            node("uuv_11", PlatformKind.UUV, (1000.0, 0.0), 1.0, 1500.0),
        ),
    )

    assert has_path(snapshot, "uuv_10", "uuv_11")
```

- [ ] **Step 2: Run tests and verify the missing module failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/simulation/test_connectivity.py -q
```

Expected: collection fails with missing `simulation.connectivity`.

- [ ] **Step 3: Implement deterministic link construction and BFS reachability**

Create `src/underwater_tracking/simulation/connectivity.py`:

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot

from underwater_tracking.domain.platforms import CommunicationLink, PlatformKind


@dataclass(frozen=True, slots=True)
class ConnectivityNode:
    platform_id: str
    kind: PlatformKind
    position_xy: tuple[float, float]
    surface_range_m: float
    acoustic_range_m: float


@dataclass(frozen=True, slots=True)
class ConnectivitySnapshot:
    links: tuple[CommunicationLink, ...]


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def build_connectivity(
    *,
    carrier_id: str,
    carrier_xy: tuple[float, float],
    nodes: tuple[ConnectivityNode, ...],
) -> ConnectivitySnapshot:
    links: list[CommunicationLink] = []
    ordered = tuple(sorted(nodes, key=lambda node: node.platform_id))
    for node in ordered:
        if node.kind is PlatformKind.USV:
            distance = _distance(carrier_xy, node.position_xy)
            if distance <= node.surface_range_m:
                links.append(
                    CommunicationLink(
                        source_id=carrier_id,
                        target_id=node.platform_id,
                        medium="surface",
                        distance_m=distance,
                    )
                )
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            distance = _distance(left.position_xy, right.position_xy)
            if left.kind is PlatformKind.USV and right.kind is PlatformKind.USV:
                medium = "surface"
                limit = min(left.surface_range_m, right.surface_range_m)
            else:
                medium = "acoustic"
                limit = min(left.acoustic_range_m, right.acoustic_range_m)
            if distance <= limit:
                links.append(
                    CommunicationLink(
                        source_id=left.platform_id,
                        target_id=right.platform_id,
                        medium=medium,
                        distance_m=distance,
                    )
                )
    return ConnectivitySnapshot(
        links=tuple(sorted(links, key=lambda link: (link.source_id, link.target_id)))
    )


def has_path(snapshot: ConnectivitySnapshot, source_id: str, target_id: str) -> bool:
    if source_id == target_id:
        return True
    adjacency: dict[str, set[str]] = {}
    for link in snapshot.links:
        adjacency.setdefault(link.source_id, set()).add(link.target_id)
        adjacency.setdefault(link.target_id, set()).add(link.source_id)
    queue = deque([source_id])
    visited = {source_id}
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor == target_id:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
```

- [ ] **Step 4: Run focused tests and static checks**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/simulation/test_connectivity.py -q
PYTHONPATH=src .venv/bin/python -m ruff check src/underwater_tracking/simulation/connectivity.py tests/simulation/test_connectivity.py
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking/simulation/connectivity.py
```

Expected: `3 passed`; Ruff and Mypy exit `0`.

- [ ] **Step 5: Commit Task 4 files**

```bash
git add src/underwater_tracking/simulation/connectivity.py tests/simulation/test_connectivity.py
git commit -m "feat: add distance communication graph"
```

---

### Task 5: Add Passive and Multistatic Sonar Contracts and Simulation

**Files:**
- Create: `src/underwater_tracking/domain/observations.py`
- Create: `src/underwater_tracking/simulation/sonar.py`
- Create: `tests/domain/test_observation_contracts.py`
- Create: `tests/simulation/test_multistatic_sonar.py`

**Interfaces:**
- Consumes: Task 1 `PlatformKind` and `SonarCapability`.
- Produces: `PassiveSonarObservation`, `ActiveTransmission`, `MultistaticObservation`, `make_passive_observation`, and `make_multistatic_observations`.

- [ ] **Step 1: Write failing observation contract tests**

Create `tests/domain/test_observation_contracts.py`:

```python
import pytest
from pydantic import ValidationError

from underwater_tracking.domain.observations import (
    ActiveTransmission,
    MultistaticObservation,
    PassiveSonarObservation,
)


def test_observation_contracts_use_generic_platform_ids() -> None:
    passive = PassiveSonarObservation(
        observation_id="passive:usv_00:target_00:30",
        scenario_id="single-target-relay",
        sim_time_s=30,
        observer_id="usv_00",
        target_id="target_00",
        azimuth_rad=0.2,
        variance_rad2=0.01,
        detection_confidence=0.8,
        snr_db=6.0,
    )
    transmission = ActiveTransmission(
        transmission_id="ping:usv_00:target_00:30",
        scenario_id="single-target-relay",
        sim_time_s=30,
        emitter_id="usv_00",
        target_id="target_00",
    )
    active = MultistaticObservation(
        observation_id="active:usv_00:uuv_00:target_00:30",
        transmission_id=transmission.transmission_id,
        scenario_id="single-target-relay",
        sim_time_s=30,
        emitter_id="usv_00",
        receiver_id="uuv_00",
        target_id="target_00",
        bistatic_range_m=3000.0,
        receiver_azimuth_rad=0.3,
        range_variance_m2=225.0,
        bearing_variance_rad2=9e-6,
        detection_confidence=0.9,
    )

    assert passive.observer_id == "usv_00"
    assert active.receiver_id == "uuv_00"
    assert "position" not in active.model_dump()


def test_observations_reject_non_finite_measurements() -> None:
    with pytest.raises(ValidationError):
        PassiveSonarObservation(
            observation_id="bad",
            scenario_id="scenario",
            sim_time_s=0,
            observer_id="uuv_00",
            target_id="target_00",
            azimuth_rad=float("nan"),
            variance_rad2=0.01,
            detection_confidence=1.0,
            snr_db=0.0,
        )
```

- [ ] **Step 2: Write failing multistatic simulation tests**

Create `tests/simulation/test_multistatic_sonar.py`:

```python
import random

from underwater_tracking.domain.platforms import SonarCapability
from underwater_tracking.simulation.sonar import (
    SonarNode,
    make_multistatic_observations,
    make_passive_observation,
)


CAPABILITY = SonarCapability(
    passive_range_m=5000.0,
    passive_bearing_variance_rad2=0.01,
    active_source_range_m=4000.0,
    active_receive_range_m=5000.0,
    active_range_sigma_m=15.0,
    active_bearing_sigma_rad=0.003,
    active_capable=True,
    ping_cooldown_s=30,
    ping_energy_cost_fraction=0.001,
    clutter_sensitivity=0.2,
    exposure_cost=0.4,
)


def test_passive_observation_respects_detection_range() -> None:
    observer = SonarNode("uuv_00", (0.0, 0.0), CAPABILITY)
    near = make_passive_observation(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(3000.0, 0.0),
        rng=random.Random(1),
    )
    far = make_passive_observation(
        scenario_id="scenario",
        sim_time_s=30,
        observer=observer,
        target_id="target_00",
        target_xy=(6000.0, 0.0),
        rng=random.Random(1),
    )

    assert near is not None
    assert far is None


def test_one_emitter_produces_observations_for_all_in_range_receivers() -> None:
    emitter = SonarNode("usv_00", (0.0, 0.0), CAPABILITY)
    receivers = (
        SonarNode("uuv_00", (2000.0, 0.0), CAPABILITY),
        SonarNode("uuv_01", (0.0, 2000.0), CAPABILITY),
        SonarNode("uuv_far", (9000.0, 0.0), CAPABILITY),
    )

    transmission, observations = make_multistatic_observations(
        scenario_id="scenario",
        sim_time_s=60,
        emitter=emitter,
        receivers=receivers,
        target_id="target_00",
        target_xy=(1000.0, 1000.0),
        rng=random.Random(5),
    )

    assert transmission.emitter_id == "usv_00"
    assert [observation.receiver_id for observation in observations] == ["uuv_00", "uuv_01"]
    assert all(observation.bistatic_range_m > 0.0 for observation in observations)
```

- [ ] **Step 3: Run both files and verify missing modules**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/domain/test_observation_contracts.py tests/simulation/test_multistatic_sonar.py -q
```

Expected: collection fails because `domain.observations` and `simulation.sonar` do not exist.

- [ ] **Step 4: Implement strict observation contracts**

Create `src/underwater_tracking/domain/observations.py`:

```python
from __future__ import annotations

from math import pi
from typing import Annotated

from pydantic import Field, field_validator

from underwater_tracking.domain.models import StrictModel

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PassiveSonarObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    observer_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    azimuth_rad: float = Field(allow_inf_nan=False)
    variance_rad2: PositiveFloat
    detection_confidence: UnitFloat
    snr_db: float = Field(allow_inf_nan=False)

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_azimuth(cls, value: float) -> float:
        return (value + pi) % (2.0 * pi) - pi


class ActiveTransmission(StrictModel):
    transmission_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    emitter_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class MultistaticObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    transmission_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    emitter_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    bistatic_range_m: PositiveFloat
    receiver_azimuth_rad: float = Field(allow_inf_nan=False)
    range_variance_m2: PositiveFloat
    bearing_variance_rad2: PositiveFloat
    detection_confidence: UnitFloat

    @field_validator("receiver_azimuth_rad")
    @classmethod
    def wrap_azimuth(cls, value: float) -> float:
        return (value + pi) % (2.0 * pi) - pi
```

- [ ] **Step 5: Implement passive and multistatic simulation functions**

Create `src/underwater_tracking/simulation/sonar.py`:

```python
from __future__ import annotations

import random
from dataclasses import dataclass
from math import atan2, hypot, pi

from underwater_tracking.domain.observations import (
    ActiveTransmission,
    MultistaticObservation,
    PassiveSonarObservation,
)
from underwater_tracking.domain.platforms import SonarCapability


@dataclass(frozen=True, slots=True)
class SonarNode:
    platform_id: str
    position_xy: tuple[float, float]
    capability: SonarCapability


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _wrapped_noisy_bearing(
    origin: tuple[float, float],
    target: tuple[float, float],
    sigma_rad: float,
    rng: random.Random,
) -> float:
    bearing = atan2(target[1] - origin[1], target[0] - origin[0])
    return (bearing + rng.gauss(0.0, sigma_rad) + pi) % (2.0 * pi) - pi


def make_passive_observation(
    *,
    scenario_id: str,
    sim_time_s: int,
    observer: SonarNode,
    target_id: str,
    target_xy: tuple[float, float],
    rng: random.Random,
) -> PassiveSonarObservation | None:
    distance = _distance(observer.position_xy, target_xy)
    if distance > observer.capability.passive_range_m:
        return None
    confidence = max(0.0, min(1.0, 1.0 - distance / observer.capability.passive_range_m))
    snr_db = 20.0 * confidence - 10.0
    return PassiveSonarObservation(
        observation_id=f"passive:{observer.platform_id}:{target_id}:{sim_time_s}",
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        observer_id=observer.platform_id,
        target_id=target_id,
        azimuth_rad=_wrapped_noisy_bearing(
            observer.position_xy,
            target_xy,
            observer.capability.passive_bearing_variance_rad2**0.5,
            rng,
        ),
        variance_rad2=observer.capability.passive_bearing_variance_rad2,
        detection_confidence=confidence,
        snr_db=snr_db,
    )


def make_multistatic_observations(
    *,
    scenario_id: str,
    sim_time_s: int,
    emitter: SonarNode,
    receivers: tuple[SonarNode, ...],
    target_id: str,
    target_xy: tuple[float, float],
    rng: random.Random,
) -> tuple[ActiveTransmission, tuple[MultistaticObservation, ...]]:
    if not emitter.capability.active_capable:
        raise ValueError(f"platform {emitter.platform_id!r} cannot emit active sonar")
    emitter_leg = _distance(emitter.position_xy, target_xy)
    if emitter_leg > emitter.capability.active_source_range_m:
        raise ValueError("target is outside emitter active-source range")
    transmission = ActiveTransmission(
        transmission_id=f"ping:{emitter.platform_id}:{target_id}:{sim_time_s}",
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        emitter_id=emitter.platform_id,
        target_id=target_id,
    )
    observations: list[MultistaticObservation] = []
    for receiver in sorted(receivers, key=lambda node: node.platform_id):
        receiver_leg = _distance(receiver.position_xy, target_xy)
        if receiver_leg > receiver.capability.active_receive_range_m:
            continue
        range_sigma = receiver.capability.active_range_sigma_m
        bearing_sigma = receiver.capability.active_bearing_sigma_rad
        confidence = max(
            0.0,
            min(1.0, 1.0 - receiver_leg / receiver.capability.active_receive_range_m),
        )
        observations.append(
            MultistaticObservation(
                observation_id=(
                    f"active:{emitter.platform_id}:{receiver.platform_id}:"
                    f"{target_id}:{sim_time_s}"
                ),
                transmission_id=transmission.transmission_id,
                scenario_id=scenario_id,
                sim_time_s=sim_time_s,
                emitter_id=emitter.platform_id,
                receiver_id=receiver.platform_id,
                target_id=target_id,
                bistatic_range_m=max(
                    1e-6,
                    emitter_leg + receiver_leg + rng.gauss(0.0, range_sigma),
                ),
                receiver_azimuth_rad=_wrapped_noisy_bearing(
                    receiver.position_xy,
                    target_xy,
                    bearing_sigma,
                    rng,
                ),
                range_variance_m2=range_sigma**2,
                bearing_variance_rad2=bearing_sigma**2,
                detection_confidence=confidence,
            )
        )
    return transmission, tuple(observations)
```

- [ ] **Step 6: Run focused tests and static checks**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/domain/test_observation_contracts.py tests/simulation/test_multistatic_sonar.py -q
PYTHONPATH=src .venv/bin/python -m ruff check src/underwater_tracking/domain/observations.py src/underwater_tracking/simulation/sonar.py tests/domain/test_observation_contracts.py tests/simulation/test_multistatic_sonar.py
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking/domain/observations.py src/underwater_tracking/simulation/sonar.py
```

Expected: `4 passed`; Ruff and Mypy exit `0`.

- [ ] **Step 7: Commit Task 5 files**

```bash
git add src/underwater_tracking/domain/observations.py src/underwater_tracking/simulation/sonar.py tests/domain/test_observation_contracts.py tests/simulation/test_multistatic_sonar.py
git commit -m "feat: add multistatic sonar observation core"
```

---

### Task 6: Integrate the Explicit Roster into `SimulationEngine`

**Files:**
- Modify: `src/underwater_tracking/simulation/engine.py:163-219,241-303,338-403,875-942,1283-1350`
- Modify: `tests/simulation/test_engine.py`
- Create: `tests/integration/test_platform_core_scenario.py`

**Interfaces:**
- Consumes: Tasks 1-5 public interfaces.
- Produces: `SimulationEngine.platform_snapshot() -> PlatformSnapshot`; additive frame keys `usvs`, `communication_links`, and `platform_core`; legacy `SituationSnapshot` remains UUV-only for existing agents.

- [ ] **Step 1: Add failing integration tests for the explicit world**

Create `tests/integration/test_platform_core_scenario.py`:

```python
from pathlib import Path

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


def test_explicit_platform_core_world_spawns_from_yaml(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    snapshot = engine.platform_snapshot()

    assert snapshot.scenario_id == "segmented-single-target"
    assert snapshot.carrier.carrier_id == "carrier_01"
    assert [usv.platform_id for usv in snapshot.roster.usvs] == [
        "usv_00", "usv_01", "usv_02", "usv_03"
    ]
    assert [uuv.platform_id for uuv in snapshot.roster.uuvs] == [
        f"uuv_{index:02d}" for index in range(12)
    ]
    assert snapshot.carrier.onboard_platform_ids == tuple(
        f"uuv_{index:02d}" for index in range(12)
    )


def test_explicit_frame_exposes_usvs_and_distance_links(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    frame = engine.step()
    for _ in range(2):
        frame = engine.step()

    assert frame["platform_core"] is True
    assert len(frame["usvs"]) == 4
    assert frame["uuvs"][0]["deployment_state"] == "onboard"
    assert any(link["medium"] == "surface" for link in frame["communication_links"])
    assert frame["sonar_observations"]


def test_platform_snapshot_never_contains_target_truth(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    payload = engine.platform_snapshot().model_dump()
    frame = engine.step()

    snapshot_rendered = repr(payload).lower()
    frame_rendered = repr(frame).lower()
    assert "target_00" not in snapshot_rendered
    assert "truth" not in snapshot_rendered
    assert "true_position" not in snapshot_rendered
    assert "true_position" not in frame_rendered
    assert "target_truth" not in frame_rendered
    assert "ground_truth" not in frame_rendered


def test_usvs_remain_inside_carrier_support_radius_during_smoke_run(
    tmp_path: Path,
) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    for _ in range(12):
        engine.step()
        snapshot = engine.platform_snapshot()
        assert all(
            usv.distance_to_carrier_m <= snapshot.carrier.support_radius_m
            for usv in snapshot.roster.usvs
        )
```

Append to `tests/simulation/test_engine.py`:

```python
def test_legacy_default_frame_remains_backward_compatible(tmp_path):
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=42, output_dir=tmp_path)

    frame = engine.step()

    assert frame["platform_core"] is False
    assert frame["usvs"] == []
    assert frame["communication_links"] == []
    assert len(frame["uuvs"]) == 12
```

- [ ] **Step 2: Run integration tests and verify the missing engine interface**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_platform_core_scenario.py tests/simulation/test_engine.py::test_legacy_default_frame_remains_backward_compatible -q
```

Expected: failures because `platform_snapshot`, explicit roster spawning, and additive frame keys do not exist.

- [ ] **Step 3: Add explicit-world imports and constructor state**

Add these imports to `simulation/engine.py`:

```python
from underwater_tracking.config.platform_core import EnvironmentConfig, InitialPlatformConfig
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    USVPlatformState,
    UUVPlatformState,
)
from underwater_tracking.simulation.connectivity import (
    ConnectivityNode,
    ConnectivitySnapshot,
    build_connectivity,
)
from underwater_tracking.simulation.kinematics import MotionCommand, MotionState
from underwater_tracking.simulation.sonar import SonarNode, make_passive_observation
from underwater_tracking.simulation.usv import USVEntity
```

In `SimulationEngine.__init__`, use `config.scenario.scenario_id`, initialize all explicit-world state before `_spawn_world()`, and construct the carrier from YAML only on the explicit path:

```python
        self._scenario_id = config.scenario.scenario_id
        self._platform_core_enabled = config.environment is not None
        self._usvs: dict[str, USVEntity] = {}
        self._usv_deployment_states: dict[str, DeploymentState] = {}
        self._usv_capabilities: dict[str, PlatformCapability] = {}
        self._uuv_platform_capabilities: dict[str, PlatformCapability] = {}
        self._uuv_motion_limits: dict[str, MotionLimits] = {}
        self._connectivity = ConnectivitySnapshot(links=())
        self._platform_observations: tuple[PassiveSonarObservation, ...] = ()
        environment = config.environment
        if environment is None:
            self._carrier_entity = CarrierEntity()
        else:
            carrier = environment.carrier
            self._carrier_entity = CarrierEntity(
                carrier_id=carrier.platform_id,
                position_xy=carrier.position_xy,
                speed_mps=carrier.speed_mps,
                patrol_route_xy=carrier.patrol_route_xy,
                support_radius_m=carrier.support_radius_m,
            )
```

Remove the old unconditional `self._carrier_entity = CarrierEntity()`. After `_spawn_world()`, retain manager/report dictionary initialization but call `_allocate_and_create_groups()` only for the legacy path:

```python
        if not self._platform_core_enabled:
            self._allocate_and_create_groups()
```

- [ ] **Step 4: Resolve profiles and spawn the explicit roster**

Add this exact profile adapter:

```python
    def _platform_capability(self, platform: InitialPlatformConfig) -> PlatformCapability:
        catalog = self._config.platforms
        sensors = self._config.sensors
        communications = self._config.communications
        assert catalog is not None and sensors is not None and communications is not None
        motion = catalog.motion_profiles[platform.motion_profile]
        sonar = sensors.profiles[platform.sensor_profile]
        communication = communications.profiles[platform.communication_profile]
        return PlatformCapability(
            kind=platform.kind,
            motion=MotionLimits(
                max_speed_mps=motion.max_speed_mps,
                max_acceleration_mps2=motion.max_acceleration_mps2,
                max_turn_rate_rad_s=motion.max_turn_rate_rad_s,
            ),
            sonar=SonarCapability(**sonar.model_dump()),
            communications=CommunicationCapability(**communication.model_dump()),
        )
```

Rename the current `_spawn_world` implementation to `_spawn_legacy_world`, then add the dispatcher and explicit implementation:

```python
    def _spawn_world(self) -> None:
        environment = self._config.environment
        if environment is None:
            self._spawn_legacy_world()
            return
        self._spawn_explicit_world(environment)

    def _spawn_explicit_world(self, environment: EnvironmentConfig) -> None:
        if environment.decoys or len(environment.submarines) != 1:
            raise ValueError("platform-core world requires one submarine and no decoys")
        catalog = self._config.platforms
        assert catalog is not None
        for initial in environment.usvs:
            capability = self._platform_capability(initial)
            motion_profile = catalog.motion_profiles[initial.motion_profile]
            self._usv_capabilities[initial.platform_id] = capability
            self._usv_deployment_states[initial.platform_id] = DeploymentState(
                initial.deployment_state
            )
            self._usvs[initial.platform_id] = USVEntity(
                usv_id=initial.platform_id,
                platform_index=initial.platform_index,
                motion=MotionState(initial.position_xy, initial.heading_rad, 0.0),
                energy_fraction=initial.energy_fraction,
                limits=capability.motion,
                transit_energy_per_m=motion_profile.transit_energy_per_m,
                hotel_energy_per_s=motion_profile.hotel_energy_per_s,
            )
        for initial in environment.uuvs:
            capability = self._platform_capability(initial)
            self._uuv_platform_capabilities[initial.platform_id] = capability
            self._uuv_motion_limits[initial.platform_id] = capability.motion
            self._uuvs[initial.platform_id] = UUVEntity(
                uuv_id=initial.platform_id,
                position_xy=initial.position_xy,
                heading_rad=initial.heading_rad,
                energy_fraction=initial.energy_fraction,
                capability=SurveillanceCapability(
                    passive_range_m=capability.sonar.passive_range_m,
                    active_range_m=capability.sonar.active_source_range_m,
                    bearing_variance_rad2=capability.sonar.passive_bearing_variance_rad2,
                    active_sonar_available=capability.sonar.active_capable,
                    max_speed_mps=capability.motion.max_speed_mps,
                    max_turn_rate_rad_s=capability.motion.max_turn_rate_rad_s,
                ),
                platform_index=initial.platform_index,
            )
            self._uuv_speeds[initial.platform_id] = 0.0
            self._deployment_states[initial.platform_id] = DeploymentState(
                initial.deployment_state
            )
        submarine = environment.submarines[0]
        submarine_motion = catalog.motion_profiles[submarine.motion_profile]
        self._targets[submarine.target_id] = TargetEntity(
            target_id=submarine.target_id,
            position_xy=submarine.position_xy,
            velocity_xy=(
                submarine.speed_mps * cos(submarine.heading_rad),
                submarine.speed_mps * sin(submarine.heading_rad),
            ),
            intent=HiddenIntent.TRANSIT,
            bounds_xy=environment.map_bounds_xy,
            intent_speed_mps={
                intent: (
                    submarine_motion.max_speed_mps
                    if intent is HiddenIntent.EVADE
                    else submarine.speed_mps
                )
                for intent in HiddenIntent
            },
            max_speed_mps=submarine_motion.max_speed_mps,
            max_acceleration_mps2=submarine_motion.max_acceleration_mps2,
            max_turn_rate_rad_s=submarine_motion.max_turn_rate_rad_s,
        )
        self._contact_state[submarine.target_id] = {
            "classification": ContactClassification.SUBMARINE,
            "evidence": (),
            "position_xy": None,
        }
        self._rebuild_connectivity()
```

In the target constructor retained inside `_spawn_legacy_world`, also pass `max_speed_mps=tracking.submarine_sprint_speed_mps` and `max_turn_rate_rad_s=tracking.submarine_turn_rate_rad_s`; this preserves configurable legacy sprint and turn limits under the bounded integrator.

- [ ] **Step 5: Advance explicit platforms and rebuild distance connectivity**

At the beginning of `_advance_world`, advance the carrier, then run `_advance_usvs(dt_s)` when platform core is enabled. In the existing UUV loop, select configured limits on the explicit path:

```python
            limits = self._uuv_motion_limits.get(uuv_id)
            uuv.step(
                dt_s,
                limits.max_speed_mps if limits else tracking.uuv_max_speed_mps,
                limits.max_turn_rate_rad_s if limits else tracking.uuv_max_turn_rate_rad_s,
                limits.max_acceleration_mps2 if limits else None,
            )
```

After targets, decoys, and ping processing have advanced, rebuild connectivity again so the emitted links describe the current positions. Add:

```python
    def _advance_usvs(self, dt_s: float) -> None:
        carrier_xy = self._carrier_entity.position_xy
        for usv_id in sorted(self._usvs):
            if self._usv_deployment_states[usv_id] is not DeploymentState.DEPLOYED:
                continue
            usv = self._usvs[usv_id]
            dx = carrier_xy[0] - usv.motion.position_xy[0]
            dy = carrier_xy[1] - usv.motion.position_xy[1]
            distance = hypot(dx, dy)
            desired_speed = min(self._carrier_entity.speed_mps, usv.limits.max_speed_mps)
            if distance > 0.9 * self._carrier_entity.support_radius_m:
                desired_speed = usv.limits.max_speed_mps
            usv.set_motion_command(
                MotionCommand(
                    desired_heading_rad=atan2(dy, dx),
                    desired_speed_mps=desired_speed,
                )
            )
            usv.step(dt_s)

    def _rebuild_connectivity(self) -> None:
        nodes = tuple(
            ConnectivityNode(
                platform_id=state.platform_id,
                kind=state.capability.kind,
                position_xy=state.position_xy,
                surface_range_m=state.capability.communications.surface_range_m,
                acoustic_range_m=state.capability.communications.acoustic_range_m,
            )
            for state in (*self._usv_platform_states(), *self._uuv_platform_states())
            if state.deployment_state == "deployed"
        )
        self._connectivity = build_connectivity(
            carrier_id=self._carrier_entity.carrier_id,
            carrier_xy=self._carrier_entity.position_xy,
            nodes=nodes,
        )
```

- [ ] **Step 6: Add explicit platform state adapters**

Add exact USV/UUV adapters and make `platform_snapshot` explicit-scenario-only:

```python
    def _usv_platform_states(self) -> tuple[USVPlatformState, ...]:
        return tuple(
            USVPlatformState(
                platform_id=usv_id,
                platform_index=usv.platform_index,
                position_xy=usv.motion.position_xy,
                heading_rad=usv.motion.heading_rad,
                speed_mps=usv.motion.speed_mps,
                energy_fraction=usv.energy_fraction,
                deployment_state=self._usv_deployment_states[usv_id].value,
                capability=self._usv_capabilities[usv_id],
                sensor_mode=self._sensor_modes.get(usv_id, "passive"),
                distance_to_carrier_m=hypot(
                    usv.motion.position_xy[0] - self._carrier_entity.position_xy[0],
                    usv.motion.position_xy[1] - self._carrier_entity.position_xy[1],
                ),
            )
            for usv_id, usv in sorted(self._usvs.items())
        )

    def _uuv_platform_states(self) -> tuple[UUVPlatformState, ...]:
        if not self._platform_core_enabled:
            return ()
        return tuple(
            UUVPlatformState(
                platform_id=uuv_id,
                platform_index=uuv.platform_index,
                position_xy=uuv.position_xy,
                heading_rad=uuv.heading_rad,
                speed_mps=uuv.speed_mps,
                energy_fraction=uuv.energy_fraction,
                deployment_state=self._deployment_states[uuv_id].value,
                capability=self._uuv_platform_capabilities[uuv_id],
                group_id=self._uuv_groups.get(uuv_id),
                sensor_mode=self._sensor_modes.get(uuv_id, "passive"),
                is_group_leader=False,
                master_connected=False,
            )
            for uuv_id, uuv in sorted(self._uuvs.items())
        )

    def _carrier_platform_state(self) -> CarrierPlatformState:
        states = (*self._usv_platform_states(), *self._uuv_platform_states())
        by_state = {
            deployment: tuple(
                sorted(
                    state.platform_id
                    for state in states
                    if state.deployment_state == deployment
                )
            )
            for deployment in ("onboard", "deployed", "returning")
        }
        return CarrierPlatformState(
            carrier_id=self._carrier_entity.carrier_id,
            position_xy=self._carrier_entity.position_xy,
            heading_rad=self._carrier_entity.heading_rad,
            speed_mps=self._carrier_entity.speed_mps,
            support_radius_m=self._carrier_entity.support_radius_m,
            onboard_platform_ids=by_state["onboard"],
            deployed_platform_ids=by_state["deployed"],
            returning_platform_ids=by_state["returning"],
        )

    def platform_snapshot(self) -> PlatformSnapshot:
        if not self._platform_core_enabled:
            raise RuntimeError("platform_snapshot requires an explicit platform-core scenario")
        return PlatformSnapshot(
            scenario_id=self._scenario_id,
            sim_time_s=self._clock.sim_time_s,
            carrier=self._carrier_platform_state(),
            roster=PlatformRoster(
                usvs=self._usv_platform_states(),
                uuvs=self._uuv_platform_states(),
            ),
            communication_links=self._connectivity.links,
        )
```

`SimulationClock.sim_time_s` is already public; do not modify `clock.py`.

- [ ] **Step 7: Separate explicit observations from the legacy group cycle**

Rename the current `_observation_cycle` body to `_legacy_observation_cycle`, then add this dispatcher and passive platform-core cycle:

```python
    def _observation_cycle(self, sim_time_s: int) -> None:
        if self._platform_core_enabled:
            self._platform_core_observation_cycle(sim_time_s)
            return
        self._legacy_observation_cycle(sim_time_s)

    def _platform_core_observation_cycle(self, sim_time_s: int) -> None:
        states = (*self._usv_platform_states(), *self._uuv_platform_states())
        nodes = tuple(
            SonarNode(state.platform_id, state.position_xy, state.capability.sonar)
            for state in states
            if state.deployment_state == "deployed"
        )
        observations: list[PassiveSonarObservation] = []
        for target_id, target in sorted(self._targets.items()):
            for node in nodes:
                rng_key = f"platform:{target_id}:{node.platform_id}"
                rng = self._observer_rngs.setdefault(
                    rng_key,
                    random.Random(self._seed ^ _stable_int(rng_key)),
                )
                observation = make_passive_observation(
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    observer=node,
                    target_id=target_id,
                    target_xy=target.position_xy,
                    rng=rng,
                )
                if observation is not None:
                    observations.append(observation)
        self._platform_observations = tuple(observations)
```

This branch deliberately does not call `GroupManager` or the legacy carrier hook: all UUVs are onboard and mixed task groups belong to the next subproject. It also prevents the current `_latest_reports[target_id]` lookup from failing in the explicit world.

In `_build_frame`, select explicit or legacy carrier/UUV payloads and always add the new keys:

```python
        explicit_uuvs = self._uuv_platform_states()
        frame_uuvs = (
            [state.model_dump() for state in explicit_uuvs]
            if self._platform_core_enabled
            else [uuv.model_dump() for uuv in uuvs]
        )
        carrier = (
            self._carrier_platform_state().model_dump()
            if self._platform_core_enabled
            else self._carrier_entity.state_for(uuvs).model_dump()
        )
```

Use `frame_uuvs` and `carrier` for the existing `"uuvs"` and `"carrier"` fields, then add:

```python
            "platform_core": self._platform_core_enabled,
            "usvs": [state.model_dump() for state in self._usv_platform_states()],
            "communication_links": [link.model_dump() for link in self._connectivity.links],
            "sonar_observations": [
                observation.model_dump() for observation in self._platform_observations
            ],
```

Leave `_build_situation` on the existing UUV-only `SituationSnapshot` contract. The explicit observation branch must not invoke it until the mixed-group/master-slave subproject provides its replacement.

- [ ] **Step 8: Run focused engine tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_platform_core_scenario.py tests/simulation/test_engine.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Run the complete non-LLM Python regression**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -m 'not real_llm' -q
```

Expected: all selected tests pass; no current agent/planning worktree changes are required.

- [ ] **Step 10: Run static checks and commit**

```bash
PYTHONPATH=src .venv/bin/python -m ruff check src tests
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking
git add src/underwater_tracking/simulation/engine.py tests/simulation/test_engine.py tests/integration/test_platform_core_scenario.py
git commit -m "feat: integrate explicit USV UUV platform world"
```

Before committing, run `git diff --cached --name-only` and verify no pre-existing dirty agent/planning file is staged.

---

### Task 7: Verify the Platform-Core Exit Contract

**Files:**
- Modify: `docs/superpowers/specs/2026-08-16-hierarchical-adversarial-segmented-tracking-design.md` only if implementation reveals a contract discrepancy requiring user approval; otherwise no production file changes.
- Test: all Task 1-6 test files.

**Interfaces:**
- Consumes: all platform-core interfaces from Tasks 1-6.
- Produces: a verified first-subproject baseline for the mixed-group/master-slave plan.

- [ ] **Step 1: Run the new focused suite together**

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/domain/test_platform_contracts.py \
  tests/domain/test_observation_contracts.py \
  tests/config/test_platform_core_loader.py \
  tests/simulation/test_kinematics.py \
  tests/simulation/test_connectivity.py \
  tests/simulation/test_multistatic_sonar.py \
  tests/integration/test_platform_core_scenario.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run legacy regression, lint, and type checks**

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -m 'not real_llm' -q
PYTHONPATH=src .venv/bin/python -m ruff check src tests
PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking
```

Expected: all commands exit `0`.

- [ ] **Step 3: Run the explicit headless smoke**

```bash
PYTHONPATH=src .venv/bin/python -m underwater_tracking.cli simulate --config configs/scenario/segmented_single_target.yaml --steps 12 --seed 42
```

Expected: exit `0`; generated frames contain `platform_core: true`, four USVs, twelve UUVs, and non-empty surface communication links.

- [ ] **Step 4: Verify frame truth isolation through the exact integration test**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_platform_core_scenario.py::test_platform_snapshot_never_contains_target_truth -q
```

Expected: pass. This checks both `PlatformSnapshot` and the actual frame returned by `SimulationEngine.step()` without relying on a glob over unrelated historical output directories. Target truth remains available only to the separate evaluation sink.

- [ ] **Step 5: Verify the worktree and commit history**

```bash
git status --short
git log -7 --oneline
```

Expected: the pre-existing agent/planning modifications remain unstaged and identifiable; all Task 1-6 files are committed. Task 7 is verification-only and creates no commit. Any failure returns to the owning Task 1-6 step, where the exact production and test files are corrected and that task's verification is rerun.
