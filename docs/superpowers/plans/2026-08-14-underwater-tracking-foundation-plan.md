# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan:** Deterministic Underwater Tracking Foundation

**Goal:** Build a deterministic headless simulation that turns noisy multi-UUV bearings into robust target beliefs, quality-aware elastic groups, and observability-optimized waypoint commands.

**Architecture:** A 10-second simulation kernel publishes only operational observations to independent per-target group graphs. Group graphs run IMM-UIF estimation and quality calculation; deterministic planning services allocate 2–4 UUVs and issue receding-horizon waypoints using lexicographic economics and robust FIM scoring.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy, SciPy, PyYAML, LangGraph 1.x, Pytest, Hypothesis, Ruff, MyPy.

---

## File map

- `pyproject.toml`: package metadata, runtime dependencies, test/lint/type settings.
- `configs/scenario/default.yaml`: 12-UUV, 2-target default scenario and timing.
- `configs/tracking.yaml`: filter, quality, group, and waypoint defaults.
- `src/underwater_tracking/config/models.py`: typed configuration models.
- `src/underwater_tracking/config/loader.py`: YAML loading and merge logic.
- `src/underwater_tracking/domain/models.py`: cross-module Pydantic contracts.
- `src/underwater_tracking/simulation/{clock,target,uuv,bearing,engine}.py`: hidden world and operational observation gateway.
- `src/underwater_tracking/tracking/{angles,initialization,models,uif,imm,quality}.py`: bearing-only estimation and quality.
- `src/underwater_tracking/prediction/{bspline,features}.py`: trajectory prediction and historical feature extraction.
- `src/underwater_tracking/planning/{fim,allocation,waypoints,validator}.py`: deterministic plan construction.
- `src/underwater_tracking/groups/{state,nodes,graph,manager}.py`: per-target group runtime.
- `src/underwater_tracking/persistence/frame_log.py`: JSONL operational frame log.
- `src/underwater_tracking/cli.py`: headless command.

### Task 1: Scaffold the package and typed configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/underwater_tracking/__init__.py`
- Create: `src/underwater_tracking/config/__init__.py`
- Create: `src/underwater_tracking/config/models.py`
- Create: `src/underwater_tracking/config/loader.py`
- Create: `configs/scenario/default.yaml`
- Create: `configs/tracking.yaml`
- Test: `tests/config/test_loader.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from underwater_tracking.config.loader import load_app_config


def test_default_config_has_confirmed_multirate_defaults():
    config = load_app_config("configs/scenario/default.yaml")
    assert config.scenario.uuv_count == 12
    assert config.scenario.initial_target_count == 2
    assert config.timing.physics_step_s == 10
    assert config.timing.observation_step_s == 30
    assert config.timing.group_report_s == 300
    assert config.timing.strategic_review_s == 900
    assert config.tracking.group_min_size == 2
    assert config.tracking.group_max_size == 4
```

- [ ] **Step 2: Run the test and verify the package is missing**

Run: `python -m pytest tests/config/test_loader.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'underwater_tracking'`.

- [ ] **Step 3: Add package metadata and configuration models**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "underwater-tracking"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "langgraph>=1.1,<2",
  "numpy>=2.1,<3",
  "pydantic>=2.10,<3",
  "PyYAML>=6,<7",
  "scipy>=1.14,<2",
]

[project.optional-dependencies]
dev = ["hypothesis>=6,<7", "mypy>=1.14,<2", "pytest>=8,<9", "ruff>=0.9,<1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
```

```python
# src/underwater_tracking/config/models.py
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimingConfig(StrictModel):
    physics_step_s: int = 10
    observation_step_s: int = 30
    group_report_s: int = 300
    progress_report_s: int = 600
    strategic_review_s: int = 900
    prediction_horizon_s: int = 1800


class ScenarioConfig(StrictModel):
    uuv_count: int = Field(12, ge=2)
    initial_target_count: int = Field(2, ge=1)
    max_target_count: int = Field(4, ge=1)
    duration_s: int = Field(28_800, gt=0)
    seed: int = 42


class TrackingConfig(StrictModel):
    group_min_size: int = 2
    group_max_size: int = 4
    quality_warning: float = 0.65
    quality_critical: float = 0.40
    quality_release: float = 0.75
    quality_window_s: int = 300
    release_hold_s: int = 600

    @model_validator(mode="after")
    def validate_group_sizes(self):
        if self.group_min_size > self.group_max_size:
            raise ValueError("group_min_size must not exceed group_max_size")
        return self


class AppConfig(StrictModel):
    scenario: ScenarioConfig
    timing: TimingConfig
    tracking: TrackingConfig
```

```python
# src/underwater_tracking/config/loader.py
from pathlib import Path
import yaml
from underwater_tracking.config.models import AppConfig


def load_app_config(path: str | Path) -> AppConfig:
    scenario_path = Path(path)
    scenario_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    tracking_path = scenario_path.parents[1] / "tracking.yaml"
    tracking_data = yaml.safe_load(tracking_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate({**scenario_data, "tracking": tracking_data})
```

```yaml
# configs/scenario/default.yaml
scenario:
  uuv_count: 12
  initial_target_count: 2
  max_target_count: 4
  duration_s: 28800
  seed: 42
timing:
  physics_step_s: 10
  observation_step_s: 30
  group_report_s: 300
  progress_report_s: 600
  strategic_review_s: 900
  prediction_horizon_s: 1800
```

```yaml
# configs/tracking.yaml
group_min_size: 2
group_max_size: 4
quality_warning: 0.65
quality_critical: 0.40
quality_release: 0.75
quality_window_s: 300
release_hold_s: 600
```

- [ ] **Step 4: Install editable dependencies and run the test**

Run: `python -m pip install -e ".[dev]"`

Expected: exit `0`.

Run: `python -m pytest tests/config/test_loader.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit the scaffold**

```powershell
git add pyproject.toml configs src/underwater_tracking tests/config/test_loader.py
git commit -m "chore: scaffold typed underwater tracking project"
```

### Task 2: Define strict domain contracts and truth separation

**Files:**
- Create: `src/underwater_tracking/domain/__init__.py`
- Create: `src/underwater_tracking/domain/models.py`
- Create: `src/underwater_tracking/domain/truth.py`
- Test: `tests/domain/test_models.py`
- Test: `tests/domain/test_truth_boundary.py`

- [ ] **Step 1: Write failing schema tests**

```python
import pytest
from pydantic import ValidationError
from underwater_tracking.domain.models import BearingObservation, SituationSnapshot


def test_bearing_rejects_unknown_fields_and_normalizes_angle():
    observation = BearingObservation(
        observation_id="O1", scenario_id="S1", sim_time_s=30,
        uuv_id="U1", target_id="T1", azimuth_rad=7.0,
        variance_rad2=0.01, detection_confidence=0.9,
    )
    assert -3.141592653589793 <= observation.azimuth_rad < 3.141592653589793
    with pytest.raises(ValidationError):
        BearingObservation(**observation.model_dump(), target_truth=[1.0, 2.0])


def test_operational_snapshot_has_no_truth_field():
    assert "truth" not in SituationSnapshot.model_fields
    assert "true_targets" not in SituationSnapshot.model_fields
```

- [ ] **Step 2: Run and verify the missing models failure**

Run: `python -m pytest tests/domain -v`

Expected: FAIL importing `underwater_tracking.domain.models`.

- [ ] **Step 3: Implement the core contracts**

```python
# src/underwater_tracking/domain/models.py
from __future__ import annotations
from enum import StrEnum
from math import pi
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventLevel(StrEnum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    INFORMATIONAL = "informational"


class UUVStatus(StrEnum):
    AVAILABLE = "available"
    TRACKING = "tracking"
    RETURNING = "returning"
    FAILED = "failed"


class BearingObservation(StrictModel):
    observation_id: str
    scenario_id: str
    sim_time_s: int = Field(ge=0)
    uuv_id: str
    target_id: str
    azimuth_rad: float
    variance_rad2: float = Field(gt=0)
    detection_confidence: float = Field(ge=0, le=1)

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi


class UUVState(StrictModel):
    uuv_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    status: UUVStatus
    group_id: str | None = None


class TargetBelief(StrictModel):
    target_id: str
    sim_time_s: int
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    model_probabilities: dict[str, float]
    source_observation_ids: tuple[str, ...] = ()
    fim_min_eigenvalue: float = 0.0
    fim_condition: float = float("inf")


class GroupQuality(StrictModel):
    instant: float = Field(ge=0, le=1)
    window_mean: float = Field(ge=0, le=1)
    ewma: float = Field(ge=0, le=1)
    components: dict[str, float]
    hard_guard_reasons: tuple[str, ...] = ()


class GroupReport(StrictModel):
    group_id: str
    target_id: str
    sim_time_s: int
    member_ids: tuple[str, ...]
    belief: TargetBelief
    quality: GroupQuality
    plan_revision: int
    event_types: tuple[str, ...] = ()


class RuntimeEvent(StrictModel):
    event_id: str
    scenario_id: str
    sim_time_s: int
    event_type: str
    entity_id: str | None = None
    level: EventLevel
    payload: dict[str, Any] = {}


class SituationSnapshot(StrictModel):
    scenario_id: str
    snapshot_revision: int
    sim_time_s: int
    uuvs: tuple[UUVState, ...]
    group_reports: tuple[GroupReport, ...]
    pending_events: tuple[RuntimeEvent, ...]
    active_plan_id: str | None = None
    active_plan_revision: int | None = None
```

```python
# src/underwater_tracking/domain/truth.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetTruth:
    target_id: str
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    intent_label: str
```

- [ ] **Step 4: Add an import-boundary test**

```python
# tests/domain/test_truth_boundary.py
from pathlib import Path


def test_operational_modules_do_not_import_truth():
    roots = [Path("src/underwater_tracking/tracking"), Path("src/underwater_tracking/groups")]
    offenders = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "domain.truth" in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
    assert offenders == []
```

- [ ] **Step 5: Run domain tests**

Run: `python -m pytest tests/domain -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit domain contracts**

```powershell
git add src/underwater_tracking/domain tests/domain
git commit -m "feat: define strict operational domain contracts"
```

### Task 3: Implement deterministic target and UUV kinematics

**Files:**
- Create: `src/underwater_tracking/simulation/clock.py`
- Create: `src/underwater_tracking/simulation/target.py`
- Create: `src/underwater_tracking/simulation/uuv.py`
- Test: `tests/simulation/test_kinematics.py`

- [ ] **Step 1: Write failing kinematics tests**

```python
from math import pi
from underwater_tracking.simulation.target import HiddenIntent, TargetEntity
from underwater_tracking.simulation.uuv import UUVEntity


def test_uuv_respects_turn_rate_and_energy_monotonicity():
    uuv = UUVEntity("U1", (0.0, 0.0), 0.0, energy_fraction=1.0)
    uuv.set_waypoints([(0.0, 1000.0)])
    previous_energy = uuv.energy_fraction
    uuv.step(dt_s=10, max_speed_mps=3.0, max_turn_rate_rad_s=pi / 60)
    assert 0.0 < uuv.heading_rad <= pi / 6
    assert uuv.energy_fraction < previous_energy


def test_hidden_intent_is_not_exposed_by_public_state():
    target = TargetEntity("T1", (0.0, 0.0), (2.0, 0.0), HiddenIntent.TRANSIT)
    assert "intent" not in target.public_kinematics()
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/simulation/test_kinematics.py -v`

Expected: FAIL importing simulation entities.

- [ ] **Step 3: Implement clock and entities**

```python
# src/underwater_tracking/simulation/clock.py
from dataclasses import dataclass


@dataclass(slots=True)
class SimulationClock:
    step_s: int = 10
    sim_time_s: int = 0

    def tick(self) -> int:
        self.sim_time_s += self.step_s
        return self.sim_time_s
```

```python
# src/underwater_tracking/simulation/uuv.py
from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin


def wrap(value: float) -> float:
    return (value + pi) % (2 * pi) - pi


@dataclass(slots=True)
class UUVEntity:
    uuv_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    energy_fraction: float
    waypoints: list[tuple[float, float]] = field(default_factory=list)

    def set_waypoints(self, points: list[tuple[float, float]]) -> None:
        self.waypoints = list(points)

    def step(self, dt_s: float, max_speed_mps: float, max_turn_rate_rad_s: float) -> None:
        if not self.waypoints or self.energy_fraction <= 0:
            return
        x, y = self.position_xy
        wx, wy = self.waypoints[0]
        desired = atan2(wy - y, wx - x)
        turn = max(-max_turn_rate_rad_s * dt_s, min(max_turn_rate_rad_s * dt_s, wrap(desired - self.heading_rad)))
        self.heading_rad = wrap(self.heading_rad + turn)
        distance = min(max_speed_mps * dt_s, hypot(wx - x, wy - y))
        self.position_xy = (x + distance * cos(self.heading_rad), y + distance * sin(self.heading_rad))
        self.energy_fraction = max(0.0, self.energy_fraction - distance * 2e-6 - dt_s * 1e-7)
        if hypot(wx - self.position_xy[0], wy - self.position_xy[1]) < 1.0:
            self.waypoints.pop(0)
```

```python
# src/underwater_tracking/simulation/target.py
from dataclasses import dataclass
from enum import StrEnum


class HiddenIntent(StrEnum):
    TRANSIT = "transit"
    PATROL = "patrol"
    LOITER = "loiter"
    EVADE = "evade"
    APPROACH = "approach"
    WITHDRAW = "withdraw"


@dataclass(slots=True)
class TargetEntity:
    target_id: str
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    intent: HiddenIntent

    def step(self, dt_s: float) -> None:
        x, y = self.position_xy
        vx, vy = self.velocity_xy
        self.position_xy = (x + vx * dt_s, y + vy * dt_s)

    def public_kinematics(self) -> dict[str, object]:
        return {"target_id": self.target_id}
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/simulation/test_kinematics.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Add seeded intent transitions and boundary reflection tests**

Add a test that two `TargetEntity` instances initialized with the same `random.Random(42)` produce identical state transitions for 1,000 seconds, and that every position remains inside configured bounds. Implement transition probabilities as explicit per-intent tables in `target.py`; pass the RNG into `step()` rather than reading global randomness.

Run: `python -m pytest tests/simulation/test_kinematics.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit kinematics**

```powershell
git add src/underwater_tracking/simulation tests/simulation/test_kinematics.py
git commit -m "feat: add deterministic underwater kinematics"
```

### Task 4: Generate bearing observations and initialize a track

**Files:**
- Create: `src/underwater_tracking/tracking/angles.py`
- Create: `src/underwater_tracking/simulation/bearing.py`
- Create: `src/underwater_tracking/tracking/initialization.py`
- Test: `tests/tracking/test_initialization.py`

- [ ] **Step 1: Write the failing triangulation test**

```python
import numpy as np
from underwater_tracking.tracking.initialization import initialize_from_bearings


def test_gauss_newton_initialization_recovers_crossing_bearings():
    result = initialize_from_bearings(
        origins=np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]]),
        bearings=np.array([0.7853981634, 2.3561944902, -0.7853981634]),
        variances=np.full(3, 1e-4),
        prior=np.array([400.0, 600.0]),
    )
    np.testing.assert_allclose(result.position_xy, [500.0, 500.0], atol=2.0)
    assert np.all(np.linalg.eigvalsh(result.covariance_xy) > 0)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/tracking/test_initialization.py -v`

Expected: FAIL because `initialize_from_bearings` is missing.

- [ ] **Step 3: Implement angle helpers, sensor, and weighted least squares**

```python
# src/underwater_tracking/tracking/angles.py
import numpy as np


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi
```

```python
# src/underwater_tracking/tracking/initialization.py
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from underwater_tracking.tracking.angles import wrap_angle


@dataclass(frozen=True)
class InitializationResult:
    position_xy: np.ndarray
    covariance_xy: np.ndarray
    residual_norm: float


def initialize_from_bearings(origins, bearings, variances, prior):
    origins = np.asarray(origins, dtype=float)
    bearings = np.asarray(bearings, dtype=float)
    variances = np.asarray(variances, dtype=float)

    def residual(position):
        predicted = np.arctan2(position[1] - origins[:, 1], position[0] - origins[:, 0])
        return wrap_angle(predicted - bearings) / np.sqrt(variances)

    fit = least_squares(residual, np.asarray(prior, dtype=float), method="trf")
    information = fit.jac.T @ fit.jac
    covariance = np.linalg.pinv(information)
    return InitializationResult(fit.x, covariance, float(np.linalg.norm(fit.fun)))
```

```python
# src/underwater_tracking/simulation/bearing.py
from math import atan2, pi
import numpy as np
from underwater_tracking.domain.models import BearingObservation


def make_bearing_observation(*, scenario_id, sim_time_s, uuv_id, uuv_xy, target_id,
                             target_xy, variance_rad2, rng):
    truth = atan2(target_xy[1] - uuv_xy[1], target_xy[0] - uuv_xy[0])
    measured = (truth + rng.normal(0.0, variance_rad2 ** 0.5) + pi) % (2 * pi) - pi
    return BearingObservation(
        observation_id=f"{target_id}:{uuv_id}:{sim_time_s}", scenario_id=scenario_id,
        sim_time_s=sim_time_s, uuv_id=uuv_id, target_id=target_id,
        azimuth_rad=measured, variance_rad2=variance_rad2, detection_confidence=1.0,
    )
```

- [ ] **Step 4: Add near-parallel rejection**

Add `minimum_crossing_sine: float = 0.15` to `initialize_from_bearings`. Before solving, compute every bearing-pair `abs(sin(b_i-b_j))`; raise `InsufficientGeometryError` when no pair reaches the threshold. Add a test using three equal bearings and assert the exception.

- [ ] **Step 5: Run tracking initialization tests**

Run: `python -m pytest tests/tracking/test_initialization.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit bearing initialization**

```powershell
git add src/underwater_tracking/simulation/bearing.py src/underwater_tracking/tracking tests/tracking/test_initialization.py
git commit -m "feat: add robust bearing track initialization"
```

### Task 5: Implement the IMM-UIF estimator and robust measurement update

**Files:**
- Create: `src/underwater_tracking/tracking/models.py`
- Create: `src/underwater_tracking/tracking/uif.py`
- Create: `src/underwater_tracking/tracking/imm.py`
- Test: `tests/tracking/test_imm_uif.py`

- [ ] **Step 1: Write a failing consistency test**

```python
import numpy as np
from underwater_tracking.tracking.imm import build_default_imm


def test_imm_uif_reduces_position_covariance_with_crossing_bearings():
    imm = build_default_imm(
        mean=np.array([500.0, 500.0, 2.0, 0.0, 0.0]),
        covariance=np.diag([40_000.0, 40_000.0, 25.0, 25.0, 0.01]),
    )
    before = np.trace(imm.mixed_covariance[:2, :2])
    imm.predict(30.0)
    imm.update(
        observer_positions=np.array([[0.0, 0.0], [1000.0, 0.0]]),
        bearings=np.array([0.785398, 2.356194]), variances=np.array([1e-3, 1e-3]),
    )
    assert np.trace(imm.mixed_covariance[:2, :2]) < before
    assert abs(sum(imm.model_probabilities) - 1.0) < 1e-12
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/tracking/test_imm_uif.py -v`

Expected: FAIL importing `build_default_imm`.

- [ ] **Step 3: Implement shared motion and measurement functions**

```python
# src/underwater_tracking/tracking/models.py
import numpy as np


def constant_turn(state: np.ndarray, dt: float, commanded_turn: float) -> np.ndarray:
    x, y, vx, vy, _ = state
    angle = commanded_turn * dt
    c, s = np.cos(angle), np.sin(angle)
    nvx, nvy = c * vx - s * vy, s * vx + c * vy
    return np.array([x + nvx * dt, y + nvy * dt, nvx, nvy, commanded_turn])


def bearing_measurement(state: np.ndarray, observer_xy: np.ndarray) -> float:
    return float(np.arctan2(state[1] - observer_xy[1], state[0] - observer_xy[0]))
```

- [ ] **Step 4: Implement an unscented information filter**

Create `UnscentedInformationFilter` with `sigma_points()`, `predict()`, and `update_bearings()`. Use scaled unscented parameters `alpha=0.3`, `beta=2`, `kappa=0`; compute circular bearing means with `atan2(sum(w*sin z), sum(w*cos z))`; wrap all bearing residuals; gate each scalar measurement at NIS `6.635`; measurements between the Huber threshold `2.5` and the gate are variance-inflated by `abs(normalized_residual)/2.5`. Convert the posterior back to information form with `Y = pinv(P)` and `y = Y @ x` after every update.

The public API must expose constructor `UnscentedInformationFilter(mean: np.ndarray, covariance: np.ndarray, process_noise: np.ndarray)`, method `predict(transition, dt: float) -> None`, and method `update_bearings(observer_positions: np.ndarray, bearings: np.ndarray, variances: np.ndarray) -> list[float]`.

Add focused tests for angle wrapping across `-pi/pi`, rejection above the NIS gate, and covariance inflation after a missed update.

- [ ] **Step 5: Implement IMM interaction and mixing**

Create three filters named `cv`, `left_turn`, and `right_turn`, with commanded turns `0`, `+0.002`, and `-0.002` rad/s. Use transition matrix:

```python
np.array([
    [0.94, 0.03, 0.03],
    [0.08, 0.88, 0.04],
    [0.08, 0.04, 0.88],
])
```

Implement standard IMM mixing probabilities, state/covariance mixing, per-model likelihoods from accepted innovations, normalized posterior model probabilities, and public `mixed_mean`, `mixed_covariance`, and `model_probabilities` properties.

- [ ] **Step 6: Run estimator tests and a 200-step synthetic maneuver**

Run: `python -m pytest tests/tracking/test_imm_uif.py -v`

Expected: all tests PASS.

Add `test_synthetic_turn_track_has_finite_consistent_outputs` using two moving observers, one target that starts turning at step 100, and assert finite state/covariance, positive covariance eigenvalues, and normalized model probabilities at every step.

- [ ] **Step 7: Commit estimator**

```powershell
git add src/underwater_tracking/tracking tests/tracking/test_imm_uif.py
git commit -m "feat: add robust multi-model unscented information filter"
```

### Task 6: Fit B-spline predictions and extract intent features

**Files:**
- Create: `src/underwater_tracking/prediction/bspline.py`
- Create: `src/underwater_tracking/prediction/features.py`
- Test: `tests/prediction/test_bspline.py`

- [ ] **Step 1: Write failing prediction tests**

```python
import numpy as np
from underwater_tracking.prediction.bspline import predict_track


def test_weighted_bspline_extrapolates_straight_track_with_bounded_speed():
    times = np.arange(0.0, 1200.0, 30.0)
    positions = np.column_stack([2.0 * times, 0.5 * times])
    covariances = np.repeat(np.eye(2)[None, :, :] * 25.0, len(times), axis=0)
    prediction = predict_track(times, positions, covariances, horizon_s=1800,
                               sample_step_s=30, max_speed_mps=3.0,
                               max_turn_rate_rad_s=0.01)
    assert prediction.points_xy.shape == (60, 2)
    speed = np.linalg.norm(np.diff(prediction.points_xy, axis=0), axis=1) / 30.0
    assert np.max(speed) <= 3.0 + 1e-9
    assert np.all(np.diff(prediction.corridor_radius_m) >= 0)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/prediction/test_bspline.py -v`

Expected: FAIL importing prediction functions.

- [ ] **Step 3: Implement weighted cubic splines with bounded extrapolation**

Use `scipy.interpolate.UnivariateSpline` for `x(t)` and `y(t)` with weights `1 / sqrt(max(trace(P_xy), 1e-6))`. Require at least 8 history points and 240 seconds of span; otherwise raise `InsufficientHistoryError`. Sample the next horizon, clip each displacement to `max_speed_mps * sample_step_s`, clip heading change to `max_turn_rate_rad_s * sample_step_s`, and construct corridor radius as:

```python
base_sigma = np.sqrt(np.maximum(np.trace(covariances[:, :2, :2], axis1=1, axis2=2), 1e-9))
residual_rms = np.sqrt(np.mean(np.sum((positions - fitted_history) ** 2, axis=1)))
corridor = base_sigma[-1] + residual_rms * np.sqrt(1.0 + future_seconds / history_span)
```

Return a frozen `TrackPrediction` dataclass with `times_s`, `points_xy`, `corridor_radius_m`, and `fallback_used`.

- [ ] **Step 4: Implement deterministic historical features**

Create `extract_motion_features(times, positions)` returning a dictionary with mean/max speed, heading change, signed turn-rate mean, curvature quantiles, net displacement, path efficiency, dwell fraction, and last-window acceleration. Add tests for straight transit, circular loiter, and sharp evasion tracks.

- [ ] **Step 5: Run prediction tests**

Run: `python -m pytest tests/prediction -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit prediction**

```powershell
git add src/underwater_tracking/prediction tests/prediction
git commit -m "feat: add bounded bspline prediction and motion features"
```

### Task 7: Compute FIM and group quality with hard guards

**Files:**
- Create: `src/underwater_tracking/planning/fim.py`
- Create: `src/underwater_tracking/tracking/quality.py`
- Test: `tests/planning/test_fim.py`
- Test: `tests/tracking/test_quality.py`

- [ ] **Step 1: Write failing FIM geometry tests**

```python
import numpy as np
from underwater_tracking.planning.fim import bearing_fim, fim_metrics


def test_crossing_geometry_has_more_information_than_collinear_geometry():
    target = np.array([0.0, 0.0])
    crossing = bearing_fim(target, np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.full(2, 1e-3))
    collinear = bearing_fim(target, np.array([[1000.0, 0.0], [2000.0, 0.0]]), np.full(2, 1e-3))
    assert fim_metrics(crossing).min_eigenvalue > fim_metrics(collinear).min_eigenvalue
```

- [ ] **Step 2: Implement FIM and normalized metrics**

For each observer, use bearing Jacobian `H = [-dy/r^2, dx/r^2]` and accumulate `H.T @ H / variance`. Return min eigenvalue, `logdet` using `slogdet`, and condition number; clamp small negative eigenvalues caused by floating-point error to zero.

- [ ] **Step 3: Write and implement the quality test**

```python
from underwater_tracking.tracking.quality import QualityCalculator, QualityInputs


def test_quality_is_bounded_and_hard_guard_bypasses_average():
    calculator = QualityCalculator(window_s=300, ewma_alpha=0.2)
    result = calculator.update(30, QualityInputs(
        covariance_trace=100.0, fim_min_eigenvalue=0.0, fim_condition=1e12,
        detection_rate=1.0, normalized_nis=1.0, age_s=0.0,
    ))
    assert 0 <= result.instant <= 1
    assert "fim_degenerate" in result.hard_guard_reasons
```

Implement the approved weights `0.30/0.25/0.20/0.15/0.10`; normalize with explicit reference scales from `TrackingConfig`; keep timestamped values in a deque; calculate a time-bounded mean and EWMA; emit guards for no accepted observation, low FIM eigenvalue, high FIM condition, and covariance overflow.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/planning/test_fim.py tests/tracking/test_quality.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit information and quality metrics**

```powershell
git add src/underwater_tracking/planning/fim.py src/underwater_tracking/tracking/quality.py tests/planning/test_fim.py tests/tracking/test_quality.py
git commit -m "feat: add observability and group quality metrics"
```

### Task 8: Solve lexicographic elastic group allocation

**Files:**
- Create: `src/underwater_tracking/planning/allocation.py`
- Create: `src/underwater_tracking/planning/validator.py`
- Test: `tests/planning/test_allocation.py`

- [ ] **Step 1: Write a failing economic allocation test**

```python
from underwater_tracking.planning.allocation import AllocationInput, allocate_groups


def test_allocator_uses_two_members_when_quality_is_feasible():
    problem = AllocationInput.synthetic(uuv_count=6, target_count=2, feasible_pair_quality=0.8)
    solution = allocate_groups(problem)
    assert all(len(members) == 2 for members in solution.members_by_target.values())
    assert len(solution.reserve_ids) == 2
    assert solution.hard_violations == ()
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/planning/test_allocation.py -v`

Expected: FAIL importing the allocator.

- [ ] **Step 3: Implement MILP variables and lexicographic passes**

Define binary `x[u,t]` for membership and `a[u]` for active state. Hard constraints: each target receives 2–4 members, unavailable/returning/failed UUVs have zero assignments, each UUV belongs to at most one target, and infeasible range/energy pairs have zero assignments. Solve three passes with `scipy.optimize.milp`:

1. find any hard-feasible solution;
2. constrain feasibility and minimize `sum(a)`;
3. fix the minimum active count and minimize energy + travel + reassignment + rotation costs.

Use stable UUV/target ID sorting before building matrices. Return `AllocationSolution` with member mapping, reserve IDs, objective breakdown, solver status, and hard violations.

- [ ] **Step 4: Implement deterministic fallback and validator**

For solver failure, enumerate target group sizes from 2 to 4 and candidate combinations in stable cost order, prune any partial assignment that reuses a UUV or cannot fill remaining targets, and stop at the first lexicographically optimal complete assignment. Add `validate_allocation()` that independently re-checks every hard constraint.

- [ ] **Step 5: Add target-growth, failure, and hysteresis tests**

Test 12 UUVs with targets growing from 2 to 4; test one failed member; test that a prior 3-member group remains unchanged while quality is between warning and release thresholds; test release only after the configured hold time.

Run: `python -m pytest tests/planning/test_allocation.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit allocation**

```powershell
git add src/underwater_tracking/planning/allocation.py src/underwater_tracking/planning/validator.py tests/planning/test_allocation.py
git commit -m "feat: add economic elastic group allocation"
```

### Task 9: Generate robust receding-horizon waypoints

**Files:**
- Create: `src/underwater_tracking/planning/waypoints.py`
- Test: `tests/planning/test_waypoints.py`

- [ ] **Step 1: Write a failing observability waypoint test**

```python
import numpy as np
from underwater_tracking.planning.waypoints import plan_group_waypoints


def test_two_uuv_waypoints_avoid_collinear_geometry():
    result = plan_group_waypoints(
        uuv_positions=np.array([[-1000.0, 0.0], [-2000.0, 0.0]]),
        target_sigma_points=np.array([[0.0, 0.0], [100.0, 50.0], [-100.0, -50.0]]),
        previous_waypoints=None, max_step_m=900.0, min_separation_m=300.0,
        bearing_variance=1e-3, beam_width=16,
    )
    vectors = result.waypoints_xy - np.array([0.0, 0.0])
    cosine = abs(float(vectors[0] @ vectors[1])) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    assert cosine < 0.8
```

- [ ] **Step 2: Implement candidate lattice and robust beam search**

Generate candidates for relative bearings every 15 degrees and standoff radii from configured sensor minimum to maximum range. Reject candidates beyond `max_step_m`, outside scenario bounds, or violating minimum separation. Score joint choices by tuple:

```python
score = (
    worst_case_fim_min_eigenvalue,
    worst_case_logdet,
    -energy_cost,
    -waypoint_change_cost,
)
```

Evaluate every target sigma point and use the minimum information metrics. Beam search expands one UUV at a time, keeps `beam_width` candidates by stable score and coordinate tie-break, and returns a short waypoint sequence whose first point is committed.

- [ ] **Step 3: Add motion, boundary, and permutation tests**

Use Hypothesis to generate translated/rotated scenarios and assert equivalent relative geometry. Assert all waypoints satisfy step length, bounds, and separation. Permute UUV input order and compare results by UUV ID rather than array index.

- [ ] **Step 4: Run waypoint tests**

Run: `python -m pytest tests/planning/test_waypoints.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit waypoint planning**

```powershell
git add src/underwater_tracking/planning/waypoints.py tests/planning/test_waypoints.py
git commit -m "feat: add robust fim waypoint planner"
```

### Task 10: Assemble one stateful group graph per target

**Files:**
- Create: `src/underwater_tracking/groups/state.py`
- Create: `src/underwater_tracking/groups/nodes.py`
- Create: `src/underwater_tracking/groups/graph.py`
- Create: `src/underwater_tracking/groups/manager.py`
- Test: `tests/groups/test_group_graph.py`

- [ ] **Step 1: Write a failing group graph test**

```python
from underwater_tracking.groups.graph import build_group_graph
from underwater_tracking.groups.state import GroupState


def test_group_graph_updates_belief_quality_and_report(two_uuv_observations):
    graph = build_group_graph()
    output = graph.invoke(
        GroupState.initial("S1", "G-T1", "T1", ("U1", "U2"), coarse_prior=(500.0, 500.0)),
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    output = graph.invoke({"new_observations": two_uuv_observations},
                          config={"configurable": {"thread_id": "S1:T1"}})
    assert output["belief"].target_id == "T1"
    assert 0 <= output["quality"].ewma <= 1
    assert output["report"].member_ids == ("U1", "U2")
```

- [ ] **Step 2: Define focused group state**

`GroupState` contains only scenario/group/target IDs, member IDs, filter serialization, last observations, quality history, current plan revision, pending command, last report, and emitted events. It does not contain global resources, LLM messages, or truth.

- [ ] **Step 3: Implement nodes and graph**

Add nodes `ingest_observations`, `ensure_initialized`, `predict_and_update`, `calculate_quality`, `apply_plan_command`, `build_report`, and `emit_events`. Route initialization failure to an inflated-prior belief; route quality hard guards to event emission. Compile with a checkpointer supplied by the caller so each target uses an independent thread.

```python
builder = StateGraph(GroupState)
builder.add_edge(START, "ingest_observations")
builder.add_edge("ingest_observations", "ensure_initialized")
builder.add_edge("ensure_initialized", "predict_and_update")
builder.add_edge("predict_and_update", "calculate_quality")
builder.add_edge("calculate_quality", "apply_plan_command")
builder.add_edge("apply_plan_command", "build_report")
builder.add_edge("build_report", "emit_events")
builder.add_edge("emit_events", END)
```

- [ ] **Step 4: Implement `GroupManager`**

Create, invoke, complete, and list group runtimes by target ID. Invoke separate compiled graph instances or separate calls with unique `thread_id`s; never invoke the same stateful subgraph twice inside one parent superstep.

- [ ] **Step 5: Test state isolation and member failure**

Create two groups, feed different observations, and assert beliefs cannot cross. Mark one member failed, apply a plan command with a replacement, and assert the new member set and plan revision are reported.

Run: `python -m pytest tests/groups -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit group runtime**

```powershell
git add src/underwater_tracking/groups tests/groups
git commit -m "feat: add stateful target group graphs"
```

### Task 11: Build the headless simulation engine and operational frame log

**Files:**
- Create: `src/underwater_tracking/simulation/engine.py`
- Create: `src/underwater_tracking/persistence/frame_log.py`
- Create: `src/underwater_tracking/cli.py`
- Create: `src/underwater_tracking/__main__.py`
- Test: `tests/integration/test_headless_loop.py`

- [ ] **Step 1: Write a failing deterministic-loop test**

```python
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


def test_default_engine_runs_multirate_loop_without_truth_leak(tmp_path):
    engine = SimulationEngine(load_app_config("configs/scenario/default.yaml"), seed=42,
                              output_dir=tmp_path)
    frames = [engine.step() for _ in range(36)]
    assert frames[-1]["sim_time_s"] == 360
    assert len(frames[-1]["uuvs"]) == 12
    assert "target_truth" not in frames[-1]
    assert frames[-1]["group_reports"]
```

- [ ] **Step 2: Implement explicit multirate scheduling**

At every 10-second step, advance entities and energy. At multiples of 30 seconds, create observations and invoke group graphs. At multiples of 300 seconds, publish `GroupReport`. Construct an operational frame containing public UUV states, group reports, estimated tracks, quality, current assignments, events, and waypoint commands. Keep truth only on an `EvaluationSink` callback that defaults to a no-op.

- [ ] **Step 3: Implement JSONL logging**

`FrameLogger.write(frame)` serializes one UTF-8 JSON object per line, flushes after each write, retries transient `PermissionError` up to 20 times with 50-ms delay as in the reference project, and exposes `path` and `count`.

- [ ] **Step 4: Implement the CLI**

```python
# src/underwater_tracking/cli.py
import argparse
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--steps", type=int, required=True)
    simulate.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    config = load_app_config(args.config)
    engine = SimulationEngine(config, seed=args.seed)
    for _ in range(args.steps):
        engine.step()
    return 0
```

- [ ] **Step 5: Verify deterministic normalized output**

Run the same 360-step seed twice into separate temporary directories. Normalize run IDs and output paths, then assert SHA-256 hashes of the normalized JSONL match. Run seed 43 and assert its hash differs.

Run: `python -m pytest tests/integration/test_headless_loop.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the headless runtime**

```powershell
git add src/underwater_tracking/simulation/engine.py src/underwater_tracking/persistence src/underwater_tracking/cli.py src/underwater_tracking/__main__.py tests/integration/test_headless_loop.py
git commit -m "feat: assemble deterministic headless tracking loop"
```

### Task 12: Verify the foundation exit criteria

**Files:**
- Modify: `README.md`
- Create: `tests/property/test_foundation_invariants.py`

- [ ] **Step 1: Add property tests for the approved invariants**

Use Hypothesis to test angle wrapping, FIM positive semidefiniteness, quality bounds, stable allocation under input permutation, waypoint bounds, and absence of `TargetTruth` imports in operational packages.

- [ ] **Step 2: Document the runnable headless command**

Add exact install, simulation, output, and test commands to `README.md`, including:

```powershell
python -m pip install -e ".[dev]"
python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42
python -m pytest -q
```

- [ ] **Step 3: Run the full foundation verification**

Run:

```powershell
python -m pytest tests/config tests/domain tests/simulation tests/tracking tests/prediction tests/planning tests/groups tests/integration tests/property -q
python -m ruff check src/underwater_tracking tests
python -m mypy src/underwater_tracking
python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42
```

Expected: every command exits `0`; Pytest reports no failures; CLI writes a non-empty JSONL run.

- [ ] **Step 4: Commit foundation documentation and invariants**

```powershell
git add README.md tests/property
git commit -m "test: verify deterministic tracking foundation"
```
