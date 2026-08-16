# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute tasks strictly in order T1 → T6; each task is self-contained (verbatim code, exact commands, exact commit messages). Do not skip the failing-test round of any step.

**Plan:** Underwater Tracking Tactical Realism (requirements R1–R5 as tasks T1–T6)

**Goal:** (R1) make the LLM API key and every client knob explicit in `configs/llm.yaml`; (R2) correct the motion realism inversion — UUV max 4.0 m/s, submarine cruise 8.0 / sprint 14.0 m/s, scaled turn rates; (R3) add LLM trajectory segmentation for multi-group relay tracking with a deterministic uniform time-split fallback; (R4) add a human-in-the-loop assignment mode — reserved UUVs are excluded from the LLM-allocatable set, driven by a bearing-pursuit controller, and exempt from LLM commands except their own; (R5) add the active-sonar ping protocol state machine (idle → verifying → classified → in_position → all passive) with decoy entities, classification, promotion/drop, and a `sensor_mode` on `PlanCommand`.

**Architecture:** The LangGraph carrier central graph gains one deterministic node (`active_verification`) between the question branch and the three-tier routing. The simulation engine gains decoy entities, an active-sonar ping model, per-UUV sensor modes, contact state, and a reservation set. The allocator's `_available()` gate learns `reserved_uuv_ids` so reservations propagate through MILP zero-rows, the fallback enumeration, the reserve pool, and the health term with one line. `StrategyProposal` gains `segment_plan`; the optimizer seeds waypoint lattices from segment intercepts. The runtime's `apply_directive` records assignments into a shared `ReservationRegistry`; the loop syncs it into the engine every observation cycle. All new logic is deterministic; LLM involvement is limited to the strategy layer exactly as the protocol requires.

**Tech Stack:** Python 3.11 (`.venv`), LangGraph (central graph), Pydantic `StrictModel`, scipy MILP allocator, B-spline prediction port, pytest + Vitest/Testing Library.

**Prerequisites:** The repository is at HEAD `f2264cd` with the foundation, spec 8/10/14/15/17 tasks, and the UI plan (Tasks 1–10) completed. Task T6 additionally requires the UI plan Tasks 5–9 to be done (they precede it in the UI plan; implement this plan's T1–T5 first, then T6). Live LLM tests call the real LongCat provider; they carry a module-level `skipif` and `@pytest.mark.real_llm`. The venv is `.venv` (Python 3.11). pytest runs ONLY via `PYTHONPATH=. .venv/bin/python -m pytest`; ruff ignores `I001` repo-wide with line-length 100; mypy strict on `src`.

## Global constraints (binding)

1. **pytest only via `PYTHONPATH=. .venv/bin/python -m pytest`.**
2. **No mocks substitute real LLM functionality.** Every test drives the real network or never invokes the client. The `api_key` string is never printed, logged, persisted, or asserted by value anywhere.
3. **The LongCat API key string does NOT appear in this document.** In T1 Step 2 the executor inserts the exact key supplied in the task dispatch message. The key enters git history in the T1 commit — the user explicitly required that and accepted the exposure risk.
4. **Truth isolation holds.** Operational packages never import `underwater_tracking.domain.truth` (AST-enforced); frame schemas never contain the tokens `truth`, `true_position`, `target_truth`, `ground_truth`. The new `ContactClassification` is an *operational measurement* and contains no truth-kind field; decoy truth stays truth-side (`_truth`'s `"decoys"` list).
5. **ruff**: line-length 100, `I001` ignored; **mypy strict on `src`**.
6. Every task ends with a commit whose exact message is given.
7. Default scenario behavior is unchanged: `configs/scenario/default.yaml` gains only the additive `initial_decoy_count: 0`; decoy tests construct custom configs directly.

## Amendments to spec

Spec wins everywhere except the following sections, amended by requirements R1–R5:

- **§5.1 仿真与观测层** (R2, R3, R5): submarine/UUV motion parameters corrected (UUV 4.0 m/s; submarine cruise 8.0 / sprint 14.0 m/s; turn rates `π/60` and `π/300`); simulation gains decoy entities, an active-sonar ping model, per-UUV `sensor_mode`, and contact classification. Targets now outrun observers: intent understanding + trajectory prediction are the core of tracking feasibility.
- **§6.7 StrategyProposal** (R3): gains additive optional `segment_plan: SegmentPlan | None = None` (with `Segment`/`SegmentPlan` models); the strategy payload gains a `predicted_tracks` summary.
- **§6.8 TrackingPlan / PlanCommand** (R3, R5): `TrackingPlan` gains additive `segment_plan`; `PlanCommand` gains `sensor_mode: Literal["passive", "active"] = "passive"`; the verify marker scan exempts the `group_id` key inside `segment_plan` (never any other segment field).
- **§8.1 中枢主图** (R5): a deterministic `active_verification` node runs between `question_branch` and the tier routing; `CarrierState` gains `verification_commands` / `verification_states` / `verification_pingers` channels; `CarrierDependencies` gains `reservations` and `in_position_gate_m`.
- **§10.1 专家批注** (R4): `ExpertDirective` gains `directive_type: Literal["constraint", "assignment"] = "constraint"`, `assignment_target_id`, `assignment_uuv_ids`; `apply_directive` records assignments into a `ReservationRegistry` exposed via `CarrierRuntime.reservations()`; new typed shortcut `assign_target_uuvs`.
- **§11.1 观测模型** (R5): active-sonar model (range, noise, ping cadence, energy cost, ping-heard probability, classification probabilities) with `Contact` / `ContactClassification`; passive bearing-only tracking unchanged.
- **§12 意图模型**: spec wins; the evasive maneuver (R2) only feeds the hidden intent chain (EVADE), which the intent analysis already observes.
- **§14.1/14.2/14.3 分配层** (R3, R4): `AllocationInput` gains `reserved_uuv_ids`; `_available()` returns `False` for reserved UUVs (MILP zero-rows, fallback eligibility, reserve pool, and health term all follow); waypoint lattices are seeded from segment intercept points; the previous-plan-infeasible check treats reserved members as unusable.
- **§17.2/17.3 UI** (R4, R5, R3): human control-mode toggle + assignment panel (自动调度/人为接入), sonar status and classification badges (主动声纳/被动声纳/未验证/已确认潜艇/诱饵), trajectory-segment overlay (段1/段2 …); truth isolation unchanged.
- **§19.2 测试**: live LongCat tests only; module `skipif` conditions change from "env var absent" to "neither config key nor env key present" (`has_live_api_key()`); all knobs explicit.
- **§22 LLM 配置** (R1): the API key is stored in `configs/llm.yaml` (`api_key`) with env override (env wins); `max_tokens`, `max_retries`, `backoff_base_s`, `backoff_max_s` become explicit config fields; the key enters git history (user-accepted risk).

## File map

```
configs/llm.yaml                                 T1   api_key (executor-inserted) + max_tokens/max_retries/backoff
configs/tracking.yaml                            T2,T3  motion + active-sonar + decoy fields
configs/scenario/default.yaml                    T3   initial_decoy_count: 0
src/underwater_tracking/config/models.py         T1,T2,T3  LLMConfig / TrackingConfig / ScenarioConfig
src/underwater_tracking/agent/llm.py             T1   api_key param, env wins, error message
src/underwater_tracking/cli.py                   T1,T2,T5  _build_llm, predictor knobs, loop wiring
tests/conftest.py                                T1   make_live_llm / has_live_api_key
tests/agent/test_llm_port.py                     T1
tests/{agent,integration}/*  (8 live modules)    T1   skipif → has_live_api_key()
src/underwater_tracking/simulation/target.py     T2,T3  INTENT_SPEED_MPS, intent_speed_mps, apply_evasive_maneuver
src/underwater_tracking/prediction/port.py       T2   defaults 6.0 → 4.0
src/underwater_tracking/simulation/decoy.py      T3   NEW DecoyEntity
src/underwater_tracking/simulation/engine.py     T2,T3,T5  motion/config, decoys, pings, sensor modes, reservations, pursuit, promote/drop
src/underwater_tracking/domain/models.py         T3   Contact, ContactClassification, UUVState.sensor_mode/reserved, SituationSnapshot.contacts
src/underwater_tracking/agent/nodes/event_monitor.py  T3  taxonomy + active_ping/contact_classified
tests/simulation/test_active_sonar.py            T3   NEW
tests/simulation/test_kinematics.py              T2   NEW motion tests
src/underwater_tracking/domain/agent_models.py   T4,T5  Segment/SegmentPlan, VerificationCommand, PlanCommand.sensor_mode, ExpertDirective assignment fields
src/underwater_tracking/planning/segmentation.py T4   NEW default_segment_plan + initial_intercept
src/underwater_tracking/agent/nodes/strategy.py  T4   predicted_tracks payload
src/underwater_tracking/agent/nodes/verify.py    T4   segment scan + semantic checks
src/underwater_tracking/agent/nodes/optimize.py  T4,T5  predictions/intercept seeding, reserved_uuv_ids
src/underwater_tracking/agent/nodes/commit.py    T4   _check_segments, build_commands sensor_mode
src/underwater_tracking/agent/prompts.py         T4,T5  strategy-v2, directive-v2
tests/agent/test_segmentation.py                 T4   NEW
tests/integration/test_llm_real_api.py           T4   NEW live segmentation test
src/underwater_tracking/planning/allocation.py   T5   reserved_uuv_ids
src/underwater_tracking/agent/reservations.py    T5   NEW ReservationRegistry
src/underwater_tracking/agent/nodes/directives.py T5  assignment validation + assign_target_uuvs
src/underwater_tracking/agent/runtime.py         T5   reservations() + apply path
src/underwater_tracking/agent/state.py           T5   verification channels
src/underwater_tracking/agent/nodes/active_verification.py  T5  NEW
src/underwater_tracking/agent/graphs/central.py  T5   node wiring
tests/planning/test_allocation.py                T5   reserved exclusion
tests/agent/test_reservations.py                 T5   NEW
tests/agent/test_assignment_directives.py        T5   NEW
tests/agent/test_active_verification.py          T5   NEW
src/underwater_tracking/domain/ui_models.py      T6   sensor_mode/reserved/classification/last_ping_s/segment_plan
tests/api/test_frame_contracts.py                T6   additive field tests
ui/src/components/assistant/AssignmentPanel.tsx (+ test)  T6  NEW
ui/src/components/map/SonarBadges.tsx (+ test)   T6   NEW
ui/src/components/map/SegmentOverlay.tsx (+ test) T6  NEW
ui/src/components/CommandShell.tsx / RightSidebar.tsx / BottomDrawer.tsx  T6  additive wiring
```

---

## Task T1 — Explicit LLM configuration (R1)

**Files:**
- `configs/llm.yaml`
- `src/underwater_tracking/config/models.py`
- `src/underwater_tracking/agent/llm.py`
- `src/underwater_tracking/cli.py`
- `tests/conftest.py`
- `tests/agent/test_llm_port.py`
- `tests/agent/test_verify_graph.py`, `tests/agent/test_semantic_nodes.py`, `tests/agent/test_questions.py`, `tests/agent/test_agent_loader.py`, `tests/agent/test_directives.py`, `tests/agent/test_central_graph.py`, `tests/integration/test_llm_real_api.py`, `tests/integration/test_agent_loop.py`

**Step 1 — Failing tests.** Append to `tests/agent/test_llm_port.py`, and update its imports to `from underwater_tracking.agent.llm import HTTPStructuredLLM, LLMConfigError, TransientLLMError`:

```python
def test_llm_config_points_at_longcat_provider():
    """The shipped llm.yaml wires the OpenAI-compatible LongCat provider.

    Pure config check, no network: the key is present in the config file
    (referenced by value, never compared or printed) and every client knob
    is explicit.
    """
    config_path = (
        Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"
    )
    config = load_app_config(config_path)
    assert config.llm is not None
    assert config.llm.base_url == "https://api.longcat.chat/openai/v1"
    assert config.llm.model == "LongCat-2.0"
    assert config.llm.api_key_env == "UNDERWATER_TRACKING_API_KEY"
    # The key itself is never compared or printed; only its presence and the
    # explicit knob values are asserted.
    assert config.llm.api_key is not None
    assert config.llm.max_tokens == 4096
    assert config.llm.max_retries == 3
    assert config.llm.backoff_base_s == 1.0
    assert config.llm.backoff_max_s == 60.0


def test_constructor_lands_config_defaults():
    """The shipped llm.yaml defaults land on the client (no network).

    These are constructor-time mirrors of ``configs/llm.yaml`` (the same
    values ``agent-run`` passes); the client is never invoked, so neither
    the API key nor the network is involved.
    """
    client = make_live_llm()
    try:
        assert client._base_url == "https://api.longcat.chat/openai/v1"
        assert client._model == "LongCat-2.0"
        assert client._api_key_env == "UNDERWATER_TRACKING_API_KEY"
        assert client._temperature == 0.2
        assert client._max_tokens == 4096
        assert client._max_attempts == 3
        assert client._api_key is not None
    finally:
        client.close()


def test_config_api_key_bypasses_the_environment_variable_check():
    """A configured ``api_key`` supplies the bearer token without the env var.

    The base URL is unroutable, so a working key path surfaces as a
    connection ``TransientLLMError`` instead of the call-time key check
    (``LLMConfigError``) — the same ordering argument the missing-key test
    makes in reverse. The token value itself is never asserted.
    """
    client = HTTPStructuredLLM(
        base_url="http://127.0.0.1:1/v1/chat/completions",
        model="LongCat-2.0",
        api_key_env=MISSING_KEY_ENV,
        api_key="test-secret-for-the-ordering-proof",
        connect_timeout_s=0.2,
        request_timeout_s=0.5,
        max_retries=1,
    )
    try:
        with pytest.raises(TransientLLMError):
            client.invoke_structured("intent", {}, IntentHypothesis)
    finally:
        client.close()
```

Run and confirm it fails:

```
PYTHONPATH=. .venv/bin/python -m pytest tests/agent/test_llm_port.py -q
```

**Step 2 — Implement.** `configs/llm.yaml` becomes (replace the whole file; the executor substitutes the exact key from the task dispatch, and the comment states the source of truth):

```yaml
# Provider-neutral LLM client settings (spec 22, R1), validated against
# LLMConfig. Loaded additively by load_app_config() when present next to
# tracking.yaml.
#
# R1: the API key lives in this file (api_key) and every client knob is
# explicit. The environment variable api_key_env OVERRIDES api_key when
# set (env wins); a missing env var falls back to the configured key.
model: "LongCat-2.0"
base_url: "https://api.longcat.chat/openai/v1"
api_key: "<EXACT KEY SUPPLIED IN THE TASK DISPATCH — insert it here verbatim>"
api_key_env: "UNDERWATER_TRACKING_API_KEY"
temperature: 0.2
request_timeout_s: 60.0
connect_timeout_s: 10.0
max_tokens: 4096
max_retries: 3
backoff_base_s: 1.0
backoff_max_s: 60.0
```

`src/underwater_tracking/config/models.py` — replace the `LLMConfig` class with:

```python
class LLMConfig(StrictModel):
    """Provider-neutral LLM client settings (spec 22, R1).

    ``api_key`` is the explicit LongCat key from ``configs/llm.yaml``;
    ``api_key_env`` names the environment variable that OVERRIDES it when
    set (env wins). ``max_tokens``, ``max_retries``, ``backoff_base_s``
    and ``backoff_max_s`` make the client's hidden defaults explicit.
    """

    model: str = "underwater-assistant-model"
    base_url: str = "https://api.example.com/v1"
    api_key: str | None = None
    api_key_env: str = "UNDERWATER_TRACKING_API_KEY"
    temperature: float = Field(default=0.2, ge=0, le=2)
    request_timeout_s: float = Field(default=60.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    max_retries: int = Field(default=3, ge=0)
    backoff_base_s: float = Field(default=1.0, gt=0)
    backoff_max_s: float = Field(default=60.0, gt=0)
```

`src/underwater_tracking/agent/llm.py`:
- Module docstring line about the bearer token becomes: "with the bearer token read at call time from the configured ``api_key`` (``configs/llm.yaml``) or the environment variable configured via ``api_key_env`` — the environment wins when both are set."
- `__init__` gains `api_key: str | None = None` immediately after `api_key_env: str,` and stores `self._api_key = api_key`.
- Class docstring lines "The bearer token is read at call time from the configured environment variable and is never stored on the instance." become: "The bearer token is read at call time from the configured ``api_key`` or the configured environment variable (env wins) and is never stored in the ledger."
- `_request_once` key check becomes:

```python
        token = os.environ.get(self._api_key_env) or self._api_key
        if token is None:
            raise LLMConfigError(
                f"neither environment variable {self._api_key_env!r} nor a "
                "configured api_key is set"
            )
```

`src/underwater_tracking/cli.py` — replace `_build_llm` with:

```python
def _build_llm(config: AppConfig) -> HTTPStructuredLLM:
    """The real LongCat HTTP client, failing clearly when it cannot run.

    ``agent-run`` has no mock fallback: the bearer token is read at call
    time from the configured api_key (``configs/llm.yaml``) or the
    configured environment variable (env wins), so ``agent-run`` fails up
    front, naming the two sources, only when neither exists.
    """
    llm_config = config.llm
    if llm_config is None:
        print(
            "agent-run requires an 'llm' section in the config file",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if os.environ.get(llm_config.api_key_env) is None and llm_config.api_key is None:
        print(
            f"agent-run requires the {llm_config.api_key_env} environment variable"
            " or a configured api_key in the llm config",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return HTTPStructuredLLM(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key_env=llm_config.api_key_env,
        api_key=llm_config.api_key,
        request_timeout_s=llm_config.request_timeout_s,
        connect_timeout_s=llm_config.connect_timeout_s,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=llm_config.max_retries,
        backoff_base_s=llm_config.backoff_base_s,
        backoff_max_s=llm_config.backoff_max_s,
    )
```

Also update the module docstring lines "the API key is read from the configured environment variable at call time; ``agent-run`` fails with a message naming the variable when it is missing" to "the API key is read at call time from the configured api_key or environment variable (env wins); ``agent-run`` fails with a message naming both sources when neither exists".

`tests/conftest.py` — replace `make_live_llm` and `has_live_api_key`, and `REAL_LLM_SKIP_REASON`:

```python
# The reason string every live module reports when no key is available: it
# names the environment variable and the config key, never a value.
REAL_LLM_SKIP_REASON = (
    "neither the UNDERWATER_TRACKING_API_KEY environment variable nor a "
    "configured api_key is set; the live LongCat API tests are skipped"
)


def make_live_llm(**kwargs: object) -> HTTPStructuredLLM:
    """A real HTTP client over the shipped LongCat config (key from config or env)."""
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None, "configs/scenario/default.yaml must load llm.yaml"
    llm_config = config.llm
    return HTTPStructuredLLM(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key_env=llm_config.api_key_env,
        api_key=llm_config.api_key,
        request_timeout_s=llm_config.request_timeout_s,
        connect_timeout_s=llm_config.connect_timeout_s,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=llm_config.max_retries,
        backoff_base_s=llm_config.backoff_base_s,
        backoff_max_s=llm_config.backoff_max_s,
        **kwargs,
    )


def has_live_api_key() -> bool:
    """True when a key is available: the env var, or the shipped config's key."""
    if os.environ.get("UNDERWATER_TRACKING_API_KEY"):
        return True
    config = load_app_config(CONFIG_PATH)
    return bool(config.llm is not None and config.llm.api_key)
```

In `tests/integration/test_llm_real_api.py`, the module `pytestmark` becomes `pytest.mark.skipif(not has_live_api_key(), reason=REAL_LLM_SKIP_REASON)` with the import updated to `from tests.conftest import CONFIG_PATH, REAL_LLM_SKIP_REASON, has_live_api_key` (drop `import os` if it becomes unused); the outbound wire test's client construction gains `api_key=config.llm.api_key` (so it works when only the config key is set).

**Step 3 — Replace every module-level env-var skipif.** In each of the eight live modules listed in Files (`test_verify_graph.py`, `test_semantic_nodes.py`, `test_questions.py`, `test_agent_loader.py`, `test_directives.py`, `test_central_graph.py`, `test_llm_real_api.py`, `test_agent_loop.py`), replace the block

```python
pytestmark = pytest.mark.skipif(
    not os.environ.get("UNDERWATER_TRACKING_API_KEY"),
    reason="UNDERWATER_TRACKING_API_KEY is not set; the live LongCat API tests are skipped",
)
```

with

```python
pytestmark = pytest.mark.skipif(not has_live_api_key(), reason=REAL_LLM_SKIP_REASON)
```

extend the existing `from tests.conftest import ...` line (or add one) with `has_live_api_key, REAL_LLM_SKIP_REASON`, and drop any `import os` that becomes unused.

**Step 4 — Run and confirm it passes:**

```
PYTHONPATH=. .venv/bin/python -m pytest tests/agent/test_llm_port.py tests/planning tests/simulation tests/domain tests/config tests/api tests/property tests/groups tests/tracking tests/prediction -q
PYTHONPATH=. .venv/bin/ruff check src tests
PYTHONPATH=. .venv/bin/mypy src
```

Verify no module still reads the key from the environment directly: `grep -rn "UNDERWATER_TRACKING_API_KEY" tests --include="*.py"` must only match `tests/conftest.py` (the env check inside `has_live_api_key`) and `tests/agent/test_llm_port.py` (`MISSING_KEY_ENV`).

**Step 5 — Commit.**

```
git add -A
git commit -m "feat: explicit llm configuration with api key and all client knobs"
```

---

## Task T2 — Realistic submarine and UUV motion (R2)

**Files:**
- `src/underwater_tracking/config/models.py`
- `configs/tracking.yaml`
- `src/underwater_tracking/simulation/target.py`
- `src/underwater_tracking/simulation/engine.py`
- `src/underwater_tracking/prediction/port.py`
- `src/underwater_tracking/cli.py`
- `tests/simulation/test_kinematics.py`
- `tests/integration/test_agent_loop.py` (test `AgentLoop._deps` predictor)

**Step 1 — Failing tests.** Append to `tests/simulation/test_kinematics.py`:

```python
import math

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


def test_uuv_motion_respects_configured_max_speed(tmp_path):
    """R2: the configured UUV maximum speed caps actual motion."""
    base = load_app_config(CONFIG_PATH)
    config = base.model_copy(
        update={
            "tracking": base.tracking.model_copy(
                update={"uuv_max_speed_mps": 1.0}
            )
        }
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    speeds: list[float] = []
    for _ in range(12):
        frame = engine.step()
        speeds.extend(float(uuv["speed_mps"]) for uuv in frame["uuvs"])
    assert max(speeds) <= 1.0 + 1e-6


def test_submarine_cruise_outruns_uuv_max_speed(tmp_path):
    """R2: a cruising submarine is faster than the fastest UUV."""
    config = load_app_config(CONFIG_PATH)
    truths: list[dict[str, object]] = []
    engine = SimulationEngine(
        config, seed=42, output_dir=tmp_path, evaluation_sink=truths.append
    )
    for _ in range(3):
        engine.step()
    assert truths
    target = truths[-1]["targets"][0]
    vx, vy = target["velocity_xy"]
    assert math.hypot(vx, vy) >= config.tracking.submarine_cruise_speed_mps - 1e-6
    assert config.tracking.uuv_max_speed_mps < config.tracking.submarine_cruise_speed_mps
```

Run and confirm the first fails (UUV speeds reach 6.0 > 1.0) and the second fails (`2.0 < 8.0 - 1e-6`):

```
PYTHONPATH=. .venv/bin/python -m pytest tests/simulation/test_kinematics.py -q
```

**Step 2 — Implement.**

`src/underwater_tracking/config/models.py`: add `from math import pi` to the imports; extend `TrackingConfig` after `release_hold_s: int = 600` with:

```python
    # Motion realism (spec 5.1 amendment, R2): the UUV fleet tops out at
    # 4 m/s with a 3 deg/s turn rate, while submarines cruise at 8 m/s and
    # sprint at 14 m/s during evasion — the submarine pulls away from the
    # observer, so intent understanding + trajectory prediction drive
    # tracking feasibility.
    uuv_max_speed_mps: float = Field(default=4.0, gt=0)
    uuv_max_turn_rate_rad_s: float = Field(default=pi / 60.0, gt=0)
    submarine_cruise_speed_mps: float = Field(default=8.0, gt=0)
    submarine_sprint_speed_mps: float = Field(default=14.0, gt=0)
    submarine_turn_rate_rad_s: float = Field(default=pi / 300.0, gt=0)
```

`configs/tracking.yaml`: append

```yaml
# Motion realism (spec 5.1 amendment, R2).
uuv_max_speed_mps: 4.0
uuv_max_turn_rate_rad_s: 0.05235987755982988  # pi / 60
submarine_cruise_speed_mps: 8.0
submarine_sprint_speed_mps: 14.0
submarine_turn_rate_rad_s: 0.010471975511965976  # pi / 300
```

`src/underwater_tracking/simulation/target.py`: add `import math`; after `INTENT_VELOCITIES` add:

```python
# Cruise/sprint speeds (m/s) adopted when the target transitions into each
# intent (spec 5.1 amendment, R2): submarines cruise at 8 m/s and sprint at
# 14 m/s while evading — significantly faster than the 4 m/s UUV, so intent
# understanding and trajectory prediction are the core of tracking
# feasibility. ``INTENT_VELOCITIES`` above stays as the DIRECTION table.
INTENT_SPEED_MPS: dict[HiddenIntent, float] = {
    HiddenIntent.TRANSIT: 8.0,
    HiddenIntent.PATROL: 8.0,
    HiddenIntent.LOITER: 8.0,
    HiddenIntent.EVADE: 14.0,
    HiddenIntent.APPROACH: 8.0,
    HiddenIntent.WITHDRAW: 8.0,
}
```

`TargetEntity` gains a trailing field `intent_speed_mps: dict[HiddenIntent, float] | None = None` and the new methods; `step` uses them:

```python
    def step(self, dt_s: float, rng: random.Random) -> None:
        next_intent = self._sample_intent(rng)
        if next_intent is not self.intent:
            self.intent = next_intent
            self.velocity_xy = self._intent_velocity(next_intent)
        x, y = self.position_xy
        vx, vy = self.velocity_xy
        self.position_xy = (x + vx * dt_s, y + vy * dt_s)
        self._reflect_into_bounds()

    def _intent_velocity(self, intent: HiddenIntent) -> tuple[float, float]:
        """Normalized INTENT_VELOCITIES direction scaled by the intent speed."""
        dx, dy = INTENT_VELOCITIES[intent]
        scale = self._intent_speed(intent) / max(math.hypot(dx, dy), 1e-9)
        return (dx * scale, dy * scale)

    def _intent_speed(self, intent: HiddenIntent) -> float:
        if self.intent_speed_mps is None:
            return math.hypot(*INTENT_VELOCITIES[intent])
        return self.intent_speed_mps.get(intent, 8.0)
```

`src/underwater_tracking/simulation/engine.py`:
- Delete the second `_UUV_MAX_SPEED_MPS = 6.0` (line 115) and its comment block; replace with:

```python
# Fleet kinematics (spec 5.1 amendment, R2): the configured UUV maximum
# speed and turn rate replace the old module constants. The submarine now
# cruises faster than the UUV (8 vs 4 m/s), so closing a drifting standoff
# ring is no longer an option: intent understanding and trajectory
# prediction — not raw pursuit speed — are what keep a track alive.
```

- Delete `_UUV_MAX_SPEED_MPS = 6.0` and `_UUV_MAX_TURN_RATE_RAD_S = pi / 60.0` from the block near line 106 (the block keeps `_SENSOR_MIN_RANGE_M`, `_UUV_DEPLOY_RADIUS_M`, `_TARGET_SPAWN_SPAN_M`).
- `_spawn_world`: create each target with the configured speed:

```python
        tracking = self._config.tracking
        for index in range(scenario.initial_target_count):
            target_id = f"target_{index:02d}"
            x = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            y = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            self._targets[target_id] = TargetEntity(
                target_id,
                (float(x), float(y)),
                (tracking.submarine_cruise_speed_mps, 0.0),
                HiddenIntent.TRANSIT,
                intent_speed_mps=self._intent_speed_mps(),
            )
```

- `_advance_world`: replace `uuv.step(dt_s, _UUV_MAX_SPEED_MPS, _UUV_MAX_TURN_RATE_RAD_S)` with `uuv.step(dt_s, self._config.tracking.uuv_max_speed_mps, self._config.tracking.uuv_max_turn_rate_rad_s)`.
- `_situation_uuv_state`: `return state.model_copy(update={"speed_mps": float(self._config.tracking.uuv_max_speed_mps)})`.
- Add the helper:

```python
    def _intent_speed_mps(self) -> dict[HiddenIntent, float]:
        """Configured per-intent target speeds (cruise/sprint, R2)."""
        tracking = self._config.tracking
        return {
            HiddenIntent.TRANSIT: tracking.submarine_cruise_speed_mps,
            HiddenIntent.PATROL: tracking.submarine_cruise_speed_mps,
            HiddenIntent.LOITER: tracking.submarine_cruise_speed_mps,
            HiddenIntent.EVADE: tracking.submarine_sprint_speed_mps,
            HiddenIntent.APPROACH: tracking.submarine_cruise_speed_mps,
            HiddenIntent.WITHDRAW: tracking.submarine_cruise_speed_mps,
        }
```

`src/underwater_tracking/prediction/port.py`: change `_DEFAULT_MAX_SPEED_MPS = 6.0` to `_DEFAULT_MAX_SPEED_MPS = 4.0` and update its comment ("mirror the simulation constants: 4 m/s max speed") to reference the configured `uuv_max_speed_mps`; change the `make_snapshot_predictor` default comment accordingly.

`src/underwater_tracking/cli.py` `_deps` predictor gains the config motion knobs:

```python
            predictor=make_snapshot_predictor(
                belief_history=self._belief_history,
                horizon_s=config.timing.prediction_horizon_s,
                sample_step_s=config.timing.observation_step_s,
                max_speed_mps=config.tracking.uuv_max_speed_mps,
                max_turn_rate_rad_s=config.tracking.uuv_max_turn_rate_rad_s,
            ),
```

`tests/integration/test_agent_loop.py` `AgentLoop._deps`: apply the identical predictor change.

**Step 3 — Run and confirm it passes:**

```
PYTHONPATH=. .venv/bin/python -m pytest tests/simulation tests/prediction tests/agent -q
PYTHONPATH=. .venv/bin/ruff check src tests
PYTHONPATH=. .venv/bin/mypy src
```

**Step 4 — Commit.**

```
git add -A
git commit -m "feat: realistic submarine and uuv motion parameters"
```

---

## Task T3 — Active sonar model and decoy entities (R5, R3)

**Files:**
- `src/underwater_tracking/domain/models.py`
- `src/underwater_tracking/config/models.py`
- `configs/tracking.yaml`
- `configs/scenario/default.yaml`
- `src/underwater_tracking/simulation/decoy.py` (NEW)
- `src/underwater_tracking/simulation/target.py` (add `apply_evasive_maneuver`)
- `src/underwater_tracking/simulation/engine.py`
- `src/underwater_tracking/agent/nodes/event_monitor.py`
- `tests/simulation/test_active_sonar.py` (NEW)

**Step 1 — Failing tests.** Create `tests/simulation/test_active_sonar.py`:

```python
# tests/simulation/test_active_sonar.py
"""Active-sonar probe model and decoy entities (spec 5.1/11.1 amendment, R5).

The engine emits decoys with the same passive bearing observations as
submarines; classification comes exclusively from active pings. All tests
construct custom configs (decoy behavior is off by default), are
deterministic under the fixed seed, and never touch truth-boundary gates.
"""

from math import hypot, pi

import pytest

from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import ContactClassification, EventLevel
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


def _decoy_config(**overrides: object) -> object:
    base = load_app_config(CONFIG_PATH)
    tracking = base.tracking.model_copy(update=overrides)
    scenario = base.scenario.model_copy(update={"initial_decoy_count": 1})
    return base.model_copy(update={"tracking": tracking, "scenario": scenario})


def _run(config: object, steps: int, *, tmp_path, sink=None) -> list[dict[str, object]]:
    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path, evaluation_sink=sink.append if sink else None
    )
    frames: list[dict[str, object]] = []
    for _ in range(steps):
        frames.append(engine.step())
    return frames


def test_active_sonar_event_types_classify():
    monitor = EventMonitor()
    assert monitor.classify("active_ping") is EventLevel.INFORMATIONAL
    assert monitor.classify("contact_classified") is EventLevel.TACTICAL


def test_decoy_spawns_unverified_contact(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    assert "decoy_00" in contacts
    assert contacts["decoy_00"]["classification"] == "unverified"
    assert set(frames[0].keys()) <= {"contacts", "reservations", "run_id", "scenario_id",
                                     "sim_time_s", "step_index", "uuvs", "group_reports",
                                     "tracks", "quality", "assignments", "events",
                                     "waypoint_commands"}


def test_decoy_is_passively_indistinguishable_from_a_submarine(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    assert len(contacts["decoy_00"]["bearing_rays"]) == 12  # every observer


def test_truth_reports_decoys(tmp_path):
    config = _decoy_config()
    truths: list[dict[str, object]] = []
    _run(config, 1, tmp_path=tmp_path, sink=truths)
    assert truths
    decoys = truths[-1]["decoys"]
    assert [d["decoy_id"] for d in decoys] == ["decoy_00"]


def test_decoy_drift_speed_is_configured(tmp_path):
    config = _decoy_config()
    truths: list[dict[str, object]] = []
    _run(config, 3, tmp_path=tmp_path, sink=truths)
    first = truths[0]["decoys"][0]["position_xy"]
    last = truths[-1]["decoys"][0]["position_xy"]
    delta = hypot(last[0] - first[0], last[1] - first[1])
    assert delta == pytest.approx(0.5 * 20.0, abs=1e-9)  # 0.5 m/s * 2 steps


def test_active_ping_classifies_and_drains_energy(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=0.0,  # decoys ALWAYS classify submarine
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    assert contacts["decoy_00"]["classification"] == "submarine"
    assert contacts["decoy_00"]["estimated_position_xy"] is not None
    uuvs = {u["uuv_id"]: u for u in frame["uuvs"]}
    assert uuvs["uuv_00"]["energy_fraction"] == pytest.approx(1.0 - 2e-4)
    assert any(e["event_type"] == "contact_classified" for e in frame["events"])
    assert any(e["event_type"] == "active_ping" for e in frame["events"])


def test_heard_ping_triggers_evasive_sprint(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_submarine_prob=1.0,
    )
    truths: list[dict[str, object]] = []
    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path, evaluation_sink=truths.append
    )
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    engine.step()
    target = truths[-1]["targets"][0]
    vx, vy = target["velocity_xy"]
    assert hypot(vx, vy) == pytest.approx(14.0, abs=1e-6)  # EVADE sprint
    assert target["intent_label"] == "evade"


def test_drop_contact_removes_the_decoy(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.step()
    engine.drop_contact("decoy_00")
    frame = engine.step()
    assert "decoy_00" not in {c["contact_id"] for c in frame["contacts"]}


def test_promote_contact_creates_target_and_group(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=0.0,  # ALWAYS submarine
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.step()  # ping classifies decoy_00 as submarine
    engine.promote_contact("decoy_00")
    frame = engine.step()  # observation cycle creates the group
    reports = {r["target_id"]: r for r in frame["group_reports"]}
    assert "decoy_00" in reports
    assert 2 <= len(reports["decoy_00"]["member_ids"]) <= 4


def test_reserved_uuv_is_skipped_from_decoy_observation(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_reservations({"target_00": ("uuv_00",)})
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    rays = contacts["decoy_00"]["bearing_rays"]
    assert len(rays) == 11
    assert "uuv_00" not in {r["uuv_id"] for r in rays}
```

Run and confirm it fails on import of `ContactClassification` / `engine.set_sensor_mode`:

```
PYTHONPATH=. .venv/bin/python -m pytest tests/simulation/test_active_sonar.py -q
```

**Step 2 — Implement.**

`src/underwater_tracking/domain/models.py`: add `from typing import Any, Literal` (extend the existing `from typing import Any`); after `UUVStatus` add:

```python
class ContactClassification(StrEnum):
    UNVERIFIED = "unverified"
    SUBMARINE = "submarine"
    DECOY = "decoy"


class Contact(StrictModel):
    """One operational sonar contact (spec 11.1 amendment, R5).

    The classification is the operational measurement produced by active
    pings; it is not truth (decoy truth stays truth-side only). Targets
    being tracked enter classified SUBMARINE (already dispatched); decoys
    enter UNVERIFIED and are pinged by the active-verification protocol.
    """

    contact_id: str
    sim_time_s: int = Field(ge=0)
    bearing_rays: tuple[BearingObservation, ...] = ()
    classification: ContactClassification = ContactClassification.UNVERIFIED
    classification_evidence: tuple[str, ...] = ()
    estimated_position_xy: tuple[float, float] | None = None
```

`UUVState` gains trailing fields `sensor_mode: Literal["passive", "active"] = "passive"` and `reserved: bool = False`. `SituationSnapshot` gains trailing field `contacts: tuple[Contact, ...] = ()`.

`src/underwater_tracking/config/models.py`: `ScenarioConfig` gains `initial_decoy_count: int = Field(default=0, ge=0)`; `TrackingConfig` gains:

```python
    # Active-sonar probe model (spec 5.1/11.1 amendment, R5): range, noise,
    # ping cadence, energy cost, ping-heard probability, and the
    # classification probabilities of a contacted submarine vs a decoy.
    sensor_active_range_m: float = Field(default=3000.0, gt=0)
    sensor_active_range_sigma_m: float = Field(default=15.0, gt=0)
    sensor_active_bearing_sigma_rad: float = Field(default=0.003, gt=0)
    sensor_ping_interval_s: int = Field(default=30, gt=0)
    sensor_ping_energy_cost: float = Field(default=2e-4, gt=0)
    sensor_ping_heard_probability: float = Field(default=0.6, ge=0, le=1)
    sensor_active_classify_submarine_prob: float = Field(default=0.95, ge=0, le=1)
    sensor_active_classify_decoy_prob: float = Field(default=0.90, ge=0, le=1)
    # Decoy drift (spec 5.1 amendment, R5): a slow heading random walk.
    decoy_drift_speed_mps: float = Field(default=0.5, gt=0)
    decoy_heading_noise_rad_per_s: float = Field(default=0.02, gt=0)
```

`configs/tracking.yaml` appends the same eleven fields with the same values. `configs/scenario/default.yaml` `scenario:` gains `initial_decoy_count: 0`.

Create `src/underwater_tracking/simulation/decoy.py`:

```python
# src/underwater_tracking/simulation/decoy.py
"""Passive-sonar decoy entity (spec 5.1 amendment, R5).

A decoy drifts slowly with a heading random walk and is indistinguishable
from a submarine to passive sonar: the engine emits the same bearing
observations for it. Its true nature stays truth-side (``_truth`` only);
the operational ``ContactClassification`` of a contact comes exclusively
from active-sonar pings, never from the truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import cos, sin


@dataclass(slots=True)
class DecoyEntity:
    decoy_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    drift_speed_mps: float
    heading_noise_rad_per_s: float

    def step(self, dt_s: float, rng: random.Random) -> None:
        self.heading_rad = self.heading_rad + rng.gauss(
            0.0, self.heading_noise_rad_per_s
        ) * dt_s
        distance = self.drift_speed_mps * dt_s
        self.position_xy = (
            self.position_xy[0] + distance * cos(self.heading_rad),
            self.position_xy[1] + distance * sin(self.heading_rad),
        )
```

`src/underwater_tracking/simulation/target.py`: add `apply_evasive_maneuver` to `TargetEntity`:

```python
    def apply_evasive_maneuver(self, turn_angle_rad: float) -> None:
        """Evasive turn when the target detects an active ping (R2/R5).

        The target switches to EVADE and rotates its velocity vector by
        ``turn_angle_rad``; subsequent steps re-sample the intent chain
        from EVADE.
        """
        heading = math.atan2(self.velocity_xy[1], self.velocity_xy[0]) + turn_angle_rad
        self.intent = HiddenIntent.EVADE
        self.velocity_xy = self._scaled_velocity(HiddenIntent.EVADE, heading)

    def _scaled_velocity(self, intent: HiddenIntent, heading: float) -> tuple[float, float]:
        speed = self._intent_speed(intent)
        return (speed * math.cos(heading), speed * math.sin(heading))
```

`src/underwater_tracking/simulation/engine.py`:
- Imports: add `from collections.abc import Callable, Mapping, Sequence`, `from typing import Any, Literal`, and `from underwater_tracking.domain.models import Contact, ContactClassification` (extend the existing import), and `from underwater_tracking.simulation.decoy import DecoyEntity`.
- `__init__`, immediately before `self._spawn_world()`, add:

```python
        self._decoys: dict[str, DecoyEntity] = {}
        self._decoy_observations: dict[str, tuple[BearingObservation, ...]] = {}
        self._contact_state: dict[str, dict[str, Any]] = {}
        self._sensor_modes: dict[str, str] = {}
        self._ping_targets: dict[str, str | None] = {}
        self._last_ping_times: dict[tuple[str, str], int] = {}
        self._reserved_by_target: dict[str, tuple[str, ...]] = {}
        self._reserved_uuvs: frozenset[str] = frozenset()
```

- `_spawn_world`: after the UUV loop, decoys spawn with their contact state; after the targets loop, every tracked target's contact state is `SUBMARINE`:

```python
        tracking = self._config.tracking
        for index in range(scenario.initial_decoy_count):
            decoy_id = f"decoy_{index:02d}"
            x = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            y = self._master_rng.uniform(-_TARGET_SPAWN_SPAN_M, _TARGET_SPAWN_SPAN_M)
            heading = self._master_rng.uniform(-pi, pi)
            self._decoys[decoy_id] = DecoyEntity(
                decoy_id,
                (float(x), float(y)),
                heading,
                tracking.decoy_drift_speed_mps,
                tracking.decoy_heading_noise_rad_per_s,
            )
            self._contact_state[decoy_id] = {
                "classification": ContactClassification.UNVERIFIED,
                "evidence": (),
                "position_xy": None,
            }
        for target_id in self._targets:
            self._contact_state[target_id] = {
                "classification": ContactClassification.SUBMARINE,
                "evidence": (),
                "position_xy": None,
            }
```

- Replace `_advance_world` entirely:

```python
    def _advance_world(self, sim_time_s: int) -> None:
        dt_s = float(self._clock.step_s)
        tracking = self._config.tracking
        for uuv_id in sorted(self._uuvs):
            if self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE) is UUVStatus.FAILED:
                continue
            uuv = self._uuvs[uuv_id]
            if uuv_id in self._reserved_uuvs:
                # Bearing pursuit (R4): steer straight at the reserved
                # target's belief mean every physics step; the UUV is
                # exempt from plan commands.
                reserved_target = next(
                    (
                        target_id
                        for target_id, uuv_ids in self._reserved_by_target.items()
                        if uuv_id in uuv_ids
                    ),
                    None,
                )
                if reserved_target is not None:
                    report = self._latest_reports.get(reserved_target)
                    if report is not None:
                        uuv.set_waypoints(
                            [
                                (
                                    float(report.belief.mean[0]),
                                    float(report.belief.mean[1]),
                                )
                            ]
                        )
            before = uuv.position_xy
            uuv.step(
                dt_s, tracking.uuv_max_speed_mps, tracking.uuv_max_turn_rate_rad_s
            )
            after = uuv.position_xy
            self._uuv_speeds[uuv_id] = (
                hypot(after[0] - before[0], after[1] - before[1]) / dt_s
            )
        for target_id in sorted(self._targets):
            self._targets[target_id].step(dt_s, self._target_rng(target_id))
        for decoy_id in sorted(self._decoys):
            self._decoys[decoy_id].step(dt_s, self._decoy_rng(decoy_id))
        self._process_pings(sim_time_s)
```

- Add the ping processor and helpers (place after `_make_observation`):

```python
    def _process_pings(self, sim_time_s: int) -> None:
        """Execute every active-sonar ping due this physics step (R5).

        A UUV in ``active`` mode with a ping contact pings every
        ``sensor_ping_interval_s`` seconds; the ping-heard and
        classification draws use a seeded per-(uuv, contact) rng, so the
        whole protocol is deterministic per seed. Contacts keep their
        first classification (sticky); a heard ping drains the pinger's
        energy and, for a contacted submarine, triggers an evasive sprint.
        """
        tracking = self._config.tracking
        for uuv_id, contact_id in sorted(self._ping_targets.items()):
            if self._sensor_modes.get(uuv_id) != "active" or contact_id is None:
                continue
            last = self._last_ping_times.get((uuv_id, contact_id))
            if last is not None and sim_time_s - last < tracking.sensor_ping_interval_s:
                continue
            uuv = self._uuvs.get(uuv_id)
            contact = self._contact_state.get(contact_id)
            if uuv is None or contact is None:
                continue
            contact_xy = contact.get("position_xy")
            if contact_xy is None:
                report = self._latest_reports.get(contact_id)
                contact_xy = (
                    (float(report.belief.mean[0]), float(report.belief.mean[1]))
                    if report is not None
                    else None
                )
            if contact_xy is None:
                continue
            self._last_ping_times[(uuv_id, contact_id)] = sim_time_s
            range_m = hypot(
                contact_xy[0] - uuv.position_xy[0],
                contact_xy[1] - uuv.position_xy[1],
            )
            if range_m > tracking.sensor_active_range_m or range_m < _SENSOR_MIN_RANGE_M:
                continue
            azimuth_rad = atan2(
                contact_xy[1] - uuv.position_xy[1],
                contact_xy[0] - uuv.position_xy[0],
            )
            rng = self._entity_rngs.setdefault(
                f"active:{uuv_id}:{contact_id}",
                random.Random(
                    self._seed ^ _stable_int(f"active:{uuv_id}:{contact_id}")
                ),
            )
            is_decoy = contact_id in self._decoys
            if not is_decoy:
                self._contact_state[contact_id]["position_xy"] = (
                    float(contact_xy[0]),
                    float(contact_xy[1]),
                )
            if rng.random() > tracking.sensor_ping_heard_probability:
                self._events.append(
                    RuntimeEvent(
                        event_id=f"active_ping:{uuv_id}:{contact_id}:{sim_time_s}",
                        scenario_id=self._scenario_id,
                        sim_time_s=sim_time_s,
                        event_type="active_ping",
                        entity_id=contact_id,
                        level=EventLevel.INFORMATIONAL,
                        payload={
                            "uuv_id": uuv_id,
                            "contact_id": contact_id,
                            "range_m": round(range_m, 1),
                            "azimuth_rad": round(azimuth_rad, 6),
                        },
                    )
                )
                continue
            self._uuvs[uuv_id].energy_fraction = max(
                0.0, uuv.energy_fraction - tracking.sensor_ping_energy_cost
            )
            state = self._contact_state[contact_id]
            classification = state.get(
                "classification", ContactClassification.UNVERIFIED
            )
            if classification is ContactClassification.UNVERIFIED:
                classify_prob = (
                    tracking.sensor_active_classify_decoy_prob
                    if is_decoy
                    else tracking.sensor_active_classify_submarine_prob
                )
                classification = (
                    ContactClassification.SUBMARINE
                    if rng.random() < classify_prob
                    else ContactClassification.DECOY
                )
                state["classification"] = classification
                evidence = state.get("evidence", ())
                state["evidence"] = (
                    *evidence,
                    f"ping:{uuv_id}:{contact_id}:{sim_time_s}",
                )
            if classification is ContactClassification.SUBMARINE and not is_decoy:
                target = self._targets[contact_id]
                target.apply_evasive_maneuver(
                    tracking.submarine_turn_rate_rad_s
                    * tracking.sensor_ping_interval_s
                )
            self._events.append(
                RuntimeEvent(
                    event_id=f"contact_classified:{contact_id}:{uuv_id}:{sim_time_s}",
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="contact_classified",
                    entity_id=contact_id,
                    level=EventLevel.TACTICAL,
                    payload={
                        "contact_id": contact_id,
                        "classification": classification.value,
                        "uuv_id": uuv_id,
                    },
                )
            )

    def _decoy_rng(self, decoy_id: str) -> random.Random:
        rng = self._entity_rngs.get(decoy_id)
        if rng is None:
            rng = random.Random(self._seed ^ _stable_int(decoy_id))
            self._entity_rngs[decoy_id] = rng
        return rng

    def _contacts(self) -> tuple[Contact, ...]:
        """Operational contacts: dispatched tracked targets plus decoys."""
        contacts: list[Contact] = []
        for target_id in sorted(self._targets):
            report = self._latest_reports.get(target_id)
            sim_time_s = report.sim_time_s if report is not None else 0
            state = self._contact_state.get(target_id, {})
            contacts.append(
                Contact(
                    contact_id=target_id,
                    sim_time_s=sim_time_s,
                    bearing_rays=self._target_rays.get(target_id, ()),
                    classification=state.get(
                        "classification", ContactClassification.SUBMARINE
                    ),
                    classification_evidence=tuple(state.get("evidence", ())),
                    estimated_position_xy=state.get("position_xy"),
                )
            )
        for decoy_id, rays in sorted(self._decoy_observations.items()):
            state = self._contact_state.get(decoy_id, {})
            contacts.append(
                Contact(
                    contact_id=decoy_id,
                    sim_time_s=0,
                    bearing_rays=rays,
                    classification=state.get(
                        "classification", ContactClassification.UNVERIFIED
                    ),
                    classification_evidence=tuple(state.get("evidence", ())),
                    estimated_position_xy=state.get("position_xy"),
                )
            )
        return tuple(contacts)
```

- `_observation_cycle`: the per-target member loop excludes reserved UUVs, and decoy observations are collected (use the existing `_sensor_observation`):

```python
    def _observation_cycle(self, sim_time_s: int) -> None:
        for target_id in sorted(self._targets):
            report = self._latest_reports[target_id]
            members = tuple(
                member
                for member in report.member_ids
                if member not in self._reserved_uuvs
            )
            target = self._targets[target_id]
            observations = tuple(
                observation
                for uuv_id in members
                if (
                    observation := self._sensor_observation(
                        target_id, uuv_id, sim_time_s, target.position_xy
                    )
                )
                is not None
            )
            fresh = self._manager.invoke(
                target_id,
                observations=observations,
                member_positions={
                    member: self._uuvs[member].position_xy for member in members
                },
                command=self._pending_group_commands.pop(target_id, None),
            )
            self._latest_reports[target_id] = fresh
            self._events.extend(self._guard_events(fresh))
        self._decoy_observations = self._observe_decoys(sim_time_s)
        self._record_belief_history()
        self._plan_waypoints()
        if self._carrier is not None:
            self._carrier(self._build_situation(sim_time_s))

    def _observe_decoys(
        self, sim_time_s: int
    ) -> dict[str, tuple[BearingObservation, ...]]:
        """Passive bearing observations of every decoy by the fleet."""
        rays_by_decoy: dict[str, tuple[BearingObservation, ...]] = {}
        for decoy_id in sorted(self._decoys):
            rays: list[BearingObservation] = []
            for uuv_id in sorted(self._uuvs):
                if uuv_id in self._reserved_uuvs:
                    continue
                observation = self._sensor_observation(
                    decoy_id, uuv_id, sim_time_s, self._decoys[decoy_id].position_xy
                )
                if observation is not None:
                    rays.append(observation)
            rays_by_decoy[decoy_id] = tuple(rays)
        return rays_by_decoy
```

- `_build_frame` gains `"contacts"` and `"reservations"` entries (after `"assignments"`):

```python
            "contacts": [contact.model_dump() for contact in self._contacts()],
            "reservations": {
                target_id: list(uuv_ids)
                for target_id, uuv_ids in sorted(self._reserved_by_target.items())
            },
```

- `_uuv_state` gains `sensor_mode=self._sensor_modes.get(uuv_id, "passive")` and `reserved=uuv_id in self._reserved_uuvs`.
- `_build_situation` gains `contacts=tuple(self._contacts())`.
- `_truth` gains decoys:

```python
        decoys = [
            {
                "decoy_id": decoy_id,
                "position_xy": [
                    float(decoy.position_xy[0]),
                    float(decoy.position_xy[1]),
                ],
                "heading_rad": float(decoy.heading_rad),
            }
            for decoy_id, decoy in sorted(self._decoys.items())
        ]
        return {"sim_time_s": sim_time_s, "targets": targets, "decoys": decoys}
```

- Add the public protocol API (after `fail_uuv`):

```python
    def set_sensor_mode(
        self,
        uuv_id: str,
        mode: Literal["passive", "active"],
        ping_contact_id: str | None = None,
    ) -> None:
        """Set one UUV's sensor mode; ``active`` targets ``ping_contact_id`` (R5)."""
        if uuv_id not in self._uuvs:
            raise ValueError(f"unknown uuv {uuv_id!r}")
        self._sensor_modes[uuv_id] = mode
        self._ping_targets[uuv_id] = ping_contact_id

    def drop_contact(self, contact_id: str) -> None:
        """Drop a decoy-classified contact from the operational picture (R5)."""
        if contact_id not in self._decoys:
            return
        self._decoys.pop(contact_id, None)
        self._decoy_observations.pop(contact_id, None)
        self._contact_state.pop(contact_id, None)
        for uuv_id in sorted(self._uuvs):
            if self._ping_targets.get(uuv_id) == contact_id:
                self._ping_targets[uuv_id] = None

    def set_reservations(self, reservations: Mapping[str, Sequence[str]]) -> None:
        """Replace the human reservation set (R4): target -> reserved uuv ids."""
        self._reserved_by_target = {
            target_id: tuple(sorted(uuv_ids))
            for target_id, uuv_ids in reservations.items()
        }
        self._reserved_uuvs = frozenset(
            uuv for uuv_ids in self._reserved_by_target.values() for uuv in uuv_ids
        )
```

- `apply_plan_command`: after the report lookup, create the group for promoted contacts that could not allocate at promotion time, and apply the command's sensor mode:

```python
        report = self._latest_reports.get(command.target_id)
        if report is None:
            report = self._create_missing_group(command)
            if report is None:
                return
```

and, at the end of the method (after staging the group command), add:

```python
        for member in command.member_ids:
            self.set_sensor_mode(member, command.sensor_mode)
```

- Add `_create_missing_group` (after `apply_plan_command`):

```python
    def _create_missing_group(self, command: PlanCommand) -> GroupReport | None:
        """Create the group of a promoted contact not yet allocated.

        The R5 protocol promotes a submarine-classified contact through
        ``promote_contact``; when fewer than two free UUVs existed at
        promotion time, the committed plan command is the first chance to
        materialize the group. The group takes the coarse prior from the
        active-sonar measurement.
        """
        if command.target_id not in self._targets:
            return None
        contact = self._contact_state.get(command.target_id, {})
        prior = contact.get("position_xy") or self._targets[command.target_id].position_xy
        report = self._manager.create(
            command.target_id,
            scenario_id=self._scenario_id,
            group_id=command.group_id,
            member_ids=tuple(command.member_ids),
            coarse_prior=prior,
            member_positions={
                member: self._uuvs[member].position_xy for member in command.member_ids
            },
        )
        self._latest_reports[command.target_id] = report
        self._last_guard_reasons[command.target_id] = report.quality.hard_guard_reasons
        self._event_counters[command.target_id] = 0
        self._assignments[command.target_id] = tuple(command.member_ids)
        for member in command.member_ids:
            self._uuv_groups[member] = command.target_id
        return report
```

- Add `promote_contact` (after `drop_contact`):

```python
    def promote_contact(self, contact_id: str) -> None:
        """Promote a submarine-classified contact into a tracked target (R5).

        Creates the target and, when at least two free UUVs exist (not in
        a group, not reserved, not failed), its group through the existing
        allocation. The group takes the coarse prior from the active-sonar
        measurement; with fewer than two free UUVs the promotion leaves a
        target without a group and the committed plan's command creates it
        at the next observation cycle.
        """
        if contact_id in self._targets:
            return
        state = self._contact_state.get(contact_id)
        if state is None or state.get("position_xy") is None:
            return
        tracking = self._config.tracking
        measured = state["position_xy"]
        self._targets[contact_id] = TargetEntity(
            contact_id,
            measured,
            (tracking.submarine_cruise_speed_mps, 0.0),
            HiddenIntent.TRANSIT,
            intent_speed_mps=self._intent_speed_mps(),
        )
        state["classification"] = ContactClassification.SUBMARINE
        self._decoys.pop(contact_id, None)
        self._decoy_observations.pop(contact_id, None)
        free = tuple(
            sorted(
                uuv_id
                for uuv_id in self._uuvs
                if uuv_id not in self._uuv_groups
                and uuv_id not in self._reserved_uuvs
                and self._uuv_statuses.get(uuv_id, UUVStatus.AVAILABLE)
                is not UUVStatus.FAILED
            )
        )
        if len(free) < 2:
            return
        solution = allocate_groups(
            AllocationInput(
                uuv_ids=free,
                target_ids=(contact_id,),
                quality_by_target={contact_id: _INITIAL_QUALITY},
                uuv_available={uuv_id: True for uuv_id in free},
                uuv_energy_fraction={
                    uuv_id: self._uuvs[uuv_id].energy_fraction for uuv_id in free
                },
                quality_warning=tracking.quality_warning,
                quality_release=tracking.quality_release,
                release_hold_s=tracking.release_hold_s,
            )
        )
        members = solution.members_by_target.get(contact_id, ())
        if len(members) < 2:
            return
        report = self._manager.create(
            contact_id,
            scenario_id=self._scenario_id,
            group_id=f"G-{contact_id}",
            member_ids=tuple(members),
            coarse_prior=measured,
            member_positions={member: self._uuvs[member].position_xy for member in members},
        )
        self._latest_reports[contact_id] = report
        self._last_guard_reasons[contact_id] = report.quality.hard_guard_reasons
        self._event_counters[contact_id] = 0
        self._assignments[contact_id] = tuple(members)
        for member in members:
            self._uuv_groups[member] = contact_id
```

- Note: `_allocate_and_create_groups` (initial spawn) keeps using `_COARSE_PRIOR` — no change.

`src/underwater_tracking/agent/nodes/event_monitor.py`: add `"contact_classified"` to `_TACTICAL_TYPES` and `"active_ping"` to `_INFORMATIONAL_TYPES`:

```python
_TACTICAL_TYPES: frozenset[str] = frozenset({
    "group_quality_warning",
    "geometry_degradation",
    "battery_rotation",
    "contact_classified",
})

_INFORMATIONAL_TYPES: frozenset[str] = frozenset({
    "progress_report",
    "question",
    "state_changed",
    "repair_applied",
    "active_ping",
})
```

**Step 3 — Run and confirm it passes:**

```
PYTHONPATH=. .venv/bin/python -m pytest tests/simulation/test_active_sonar.py tests/simulation/test_kinematics.py tests/simulation -q
PYTHONPATH=. .venv/bin/ruff check src tests
PYTHONPATH=. .venv/bin/mypy src
```

**Step 4 — Commit.**

```
git add -A
git commit -m "feat: active sonar model and decoy entities in the simulation"
```

---

## Task T4 — LLM trajectory segmentation for relay tracking (R3)

**Files:**
- `src/underwater_tracking/domain/agent_models.py`
- `src/underwater_tracking/planning/segmentation.py` (NEW)
- `src/underwater_tracking/agent/nodes/strategy.py`
- `src/underwater_tracking/agent/nodes/verify.py`
- `src/underwater_tracking/agent/nodes/optimize.py`
- `src/underwater_tracking/agent/nodes/commit.py`
- `src/underwater_tracking/agent/prompts.py`
- `tests/agent/test_segmentation.py` (NEW)
- `tests/integration/test_llm_real_api.py`

**Step 1 — Failing tests.** Create `tests/agent/test_segmentation.py`:

```python
# tests/agent/test_segmentation.py
"""Trajectory segmentation for relay tracking (spec 6.7/14 amendment, R3).

``default_segment_plan`` splits one predicted track into equal time slices
across the available groups; the optimizer seeds each group's waypoint
lattice from its segment intercept; the verify scan rejects malformed
segments and exempts the ``group_id`` key (never any other segment field).
All tests are offline and deterministic.
"""

import math

from underwater_tracking.agent.nodes.commit import validate_plan
from underwater_tracking.agent.nodes.optimize import (
    PlanningConfig,
    optimize_candidates,
)
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.nodes.verify import validate_strategy
from underwater_tracking.domain.agent_models import (
    PredictedTrackRef,
    Segment,
    SegmentPlan,
    StrategyProposal,
    StrategySet,
)
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.planning.segmentation import (
    default_segment_plan,
    initial_intercept,
)

TARGETS = ("T1",)
EVIDENCE = ("B:T1:900",)

PREDICTION = PredictedTrackRef(
    prediction_id="S1:track:T1:3",
    target_id="T1",
    sim_time_s=900,
    horizon_s=600.0,
    sample_step_s=30.0,
    times_s=(930.0, 960.0, 990.0, 1020.0, 1050.0),
    points_xy=(
        (140.0, 230.0),
        (150.0, 240.0),
        (160.0, 250.0),
        (170.0, 260.0),
        (180.0, 270.0),
    ),
    corridor_radius_m=(40.0, 42.0, 44.0, 46.0, 48.0),
)


def _situation() -> SituationSnapshot:
    uuvs = tuple(
        UUVState(
            uuv_id=f"uuv_{index:02d}",
            position_xy=(2000.0 * math.cos(2.0 * math.pi * index / 6),
                         2000.0 * math.sin(2.0 * math.pi * index / 6)),
            heading_rad=0.0,
            speed_mps=4.0,
            energy_fraction=1.0,
            status=UUVStatus.AVAILABLE,
        )
        for index in range(6)
    )
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=uuvs,
        group_reports=(
            GroupReport(
                group_id="G-T1",
                target_id="T1",
                sim_time_s=900,
                member_ids=("uuv_00", "uuv_01", "uuv_02"),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=900,
                    mean=(130.0, 220.0, 1.0, 0.5),
                    covariance=(
                        (400.0, 0.0, 0.0, 0.0),
                        (0.0, 400.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                    model_probabilities={"cv": 0.7, "ct": 0.3},
                    source_observation_ids=("B:T1:900", "B:T1:870"),
                    fim_min_eigenvalue=0.005,
                    fim_condition=12.0,
                ),
                quality=GroupQuality(
                    instant=0.8,
                    window_mean=0.75,
                    ewma=0.76,
                    components={"cov": 0.7},
                    hard_guard_reasons=(),
                ),
                plan_revision=1,
            ),
        ),
        pending_events=(),
    )


def _proposal(*, segment_plan: SegmentPlan | None = None) -> StrategyProposal:
    return StrategyProposal(
        concept="balanced",
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7},
        reinforcement_policy={"T1": "release_when_stable"},
        releasable_soft_constraints=("energy_reserve_0.1",),
        evidence_ids=EVIDENCE,
        rationale="relay tracking along the predicted track",
        segment_plan=segment_plan,
    )


def test_default_segment_plan_splits_track_uniformly():
    plan = default_segment_plan(PREDICTION, ("G-T1", "G-OTHER"))
    assert [segment.group_id for segment in plan.segments] == ["G-OTHER", "G-T1"]
    assert [segment.index for segment in plan.segments] == [0, 1]
    assert plan.segments[0].end_s == plan.segments[1].start_s
    assert plan.segments[0].start_s == 900
    assert plan.segments[1].end_s == 1500


def test_default_segment_plan_handles_empty_inputs():
    assert default_segment_plan(PREDICTION, ()).segments == ()


def test_initial_intercept_picks_earliest_segment_for_group():
    plan = SegmentPlan(
        segments=(
            Segment(index=0, start_s=900, end_s=1200, group_id="G-OTHER",
                    intercept_xy=(150.0, 240.0)),
            Segment(index=1, start_s=1200, end_s=1500, group_id="G-T1",
                    intercept_xy=(180.0, 270.0)),
        )
    )
    assert initial_intercept(plan, "T1") == (180.0, 270.0)
    assert initial_intercept(None, "T1") is None


def test_validate_strategy_accepts_valid_segments():
    plan = SegmentPlan(
        segments=(
            Segment(index=0, start_s=900, end_s=1200, group_id="G-T1",
                    intercept_xy=(150.0, 240.0)),
        )
    )
    report = validate_strategy(
        _proposal(segment_plan=plan).model_dump(),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert report.valid is True


def test_validate_strategy_flags_bad_segments():
    bad = SegmentPlan(
        segments=(
            Segment(index=1, start_s=900, end_s=900, group_id="G-T1",
                    intercept_xy=(float("nan"), 240.0)),
        )
    )
    report = validate_strategy(
        _proposal(segment_plan=bad).model_dump(),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert {issue.code for issue in report.issues} == {
        "segment_index_gap",
        "segment_time_invalid",
        "non_finite",
    }


def test_marker_scan_exempts_group_id_inside_segments():
    plan = SegmentPlan(
        segments=(
            Segment(index=0, start_s=900, end_s=1200, group_id="G-T1",
                    intercept_xy=(150.0, 240.0)),
        )
    )
    report = validate_strategy(
        _proposal(segment_plan=plan).model_dump(),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert "member_or_waypoint" not in {issue.code for issue in report.issues}


def test_optimize_applies_proposal_segment_plan():
    proposal = _proposal(
        segment_plan=SegmentPlan(
            segments=(
                Segment(index=0, start_s=900, end_s=1500, group_id="G-T1",
                        intercept_xy=(150.0, 240.0)),
            )
        )
    )
    snapshot = build_planning_snapshot(_situation(), active_plan=None, applied_directives=())
    evaluations = optimize_candidates(
        snapshot,
        StrategySet(proposals=(proposal,)),
        config=PlanningConfig(),
    )
    plan = evaluations[0].plan
    assert plan.segment_plan is not None
    assert plan.segment_plan.segments[0].group_id == "G-T1"
    assert plan.member_ids_by_target["T1"]
    assert validate_plan(snapshot, plan, PlanningConfig()) == ()


def test_optimize_falls_back_to_default_segment_plan():
    proposal = _proposal(segment_plan=None)
    snapshot = build_planning_snapshot(_situation(), active_plan=None, applied_directives=())
    evaluations = optimize_candidates(
        snapshot,
        StrategySet(proposals=(proposal,)),
        config=PlanningConfig(),
        predictions={"T1": PREDICTION},
    )
    plan = evaluations[0].plan
    assert plan.segment_plan is not None
    assert plan.segment_plan.segments[0].group_id == "G-T1"
```

Run and confirm it fails (no `Segment`/`SegmentPlan`, no `segment_plan` on `StrategyProposal`):

```
PYTHONPATH=. .venv/bin/python -m pytest tests/agent/test_segmentation.py -q
```

**Step 2 — Implement.**

`src/underwater_tracking/domain/agent_models.py` — add before `StrategyProposal`:

```python
class Segment(StrictModel):
    """One relay-tracking time slice of a predicted track (spec 6.7 amendment, R3).

    ``intercept_xy`` is the track point where the assigned group
    initializes its standoff; ``start_s``/``end_s`` are absolute
    simulation times inside the prediction horizon.
    """

    index: int = Field(ge=0)
    start_s: int = Field(ge=0)
    end_s: int = Field(ge=0)
    group_id: str
    intercept_xy: tuple[float, float]


class SegmentPlan(StrictModel):
    """Ordered track segments across the tracking groups (R3)."""

    segments: tuple[Segment, ...] = ()
```

`StrategyProposal` gains trailing field `segment_plan: SegmentPlan | None = None`. `TrackingPlan` gains trailing field `segment_plan: SegmentPlan | None = None`.

Create `src/underwater_tracking/planning/segmentation.py`:

```python
# src/underwater_tracking/planning/segmentation.py
"""Deterministic trajectory segmentation (spec 6.7 amendment, R3).

``SegmentPlan`` partitions a target's predicted track into time slices,
each assigned to one group with an intercept point where that group
initializes its waypoint standoff. LLM proposals may carry a
``segment_plan``; when they do not, ``default_segment_plan`` splits the
predicted track into equal time slices across the available groups — a
small pure function, no LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

from underwater_tracking.domain.agent_models import (
    PredictedTrackRef,
    Segment,
    SegmentPlan,
)


def default_segment_plan(
    prediction: PredictedTrackRef,
    group_ids: Sequence[str],
) -> SegmentPlan:
    """Uniform time split of one predicted track across ``group_ids``.

    Times are absolute simulation times within the prediction horizon;
    each segment's intercept is the track point nearest its midpoint time
    (linear interpolation between bracketing samples). Empty inputs yield
    an empty ``SegmentPlan``. Deterministic: sorted, no randomness.
    """
    ordered = tuple(sorted(set(group_ids)))
    points = prediction.points_xy
    if not ordered or not points:
        return SegmentPlan(segments=())
    times = prediction.times_s
    t0 = prediction.sim_time_s
    horizon_s = float(prediction.horizon_s)
    step = horizon_s / len(ordered)
    segments: list[Segment] = []
    for index, group_id in enumerate(ordered):
        start_s = t0 + index * step
        end_s = t0 + (index + 1) * step
        intercept = _point_at(times, points, 0.5 * (start_s + end_s))
        segments.append(
            Segment(
                index=index,
                start_s=int(round(start_s)),
                end_s=int(round(end_s)),
                group_id=group_id,
                intercept_xy=intercept,
            )
        )
    return SegmentPlan(segments=tuple(segments))


def initial_intercept(
    segment_plan: SegmentPlan | None, target: str
) -> tuple[float, float] | None:
    """The intercept of the earliest segment assigned to this group (R3).

    The segment's intercept point becomes the group's initial waypoint
    target: the waypoint lattice is recentered there (see
    ``_plan_waypoints``), so the group's committed standoff converges on
    the predicted intercept instead of the current belief mean.
    """
    if segment_plan is None:
        return None
    for segment in sorted(segment_plan.segments, key=lambda s: s.index):
        if segment.group_id == f"G-{target}":
            return segment.intercept_xy
    return None


def _point_at(
    times: tuple[float, ...],
    points: tuple[tuple[float, float], ...],
    target_s: float,
) -> tuple[float, float]:
    """The track point nearest ``target_s``, interpolated between samples."""
    if not times:
        return points[len(points) // 2]
    if target_s <= times[0]:
        return points[0]
    if target_s >= times[-1]:
        return points[-1]
    for index in range(len(times) - 1):
        if times[index] <= target_s <= times[index + 1]:
            span = max(times[index + 1] - times[index], 1e-9)
            weight = (target_s - times[index]) / span
            return (
                points[index][0] + weight * (points[index + 1][0] - points[index][0]),
                points[index][1] + weight * (points[index + 1][1] - points[index][1]),
            )
    return points[-1]
```

`src/underwater_tracking/agent/nodes/strategy.py`: imports gain `from collections.abc import Mapping` and `PredictedTrackRef`; `build_payload` gains, between `"trigger_events"` and `"evidence_ids"`:

```python
            "predicted_tracks": _predicted_track_summary(
                state.get("predictions", {})
            ),
```

and append the module-level helper:

```python
def _predicted_track_summary(
    predictions: Mapping[str, PredictedTrackRef],
) -> list[dict[str, object]]:
    """Downsampled predicted-track summary for the strategy payload (R3).

    At most 24 samples per target keep the payload bounded; the corridor
    array is downsampled with the same stride.
    """
    summaries: list[dict[str, object]] = []
    for target_id, prediction in sorted(predictions.items()):
        points = prediction.points_xy
        stride = max(1, (len(points) + 23) // 24)
        summaries.append(
            {
                "target_id": target_id,
                "horizon_s": prediction.horizon_s,
                "sample_step_s": prediction.sample_step_s,
                "points_xy": [list(point) for point in points[::stride]],
                "corridor_radius_m": list(prediction.corridor_radius_m[::stride]),
                "fallback_used": prediction.fallback_used,
            }
        )
    return summaries
```

`src/underwater_tracking/agent/nodes/verify.py`:
- `_SCANNED_STRUCTURAL_FIELDS` gains `"segment_plan"`; add the exemption set; the docstring of `_find_forbidden_marker` notes the exemption:

```python
_SCANNED_STRUCTURAL_FIELDS = (
    "concept",
    "target_priorities",
    "required_quality",
    "reinforcement_policy",
    "releasable_soft_constraints",
    "segment_plan",
)

# ``group_id`` inside a segment plan is an identifier the LLM writes for
# the group that takes the segment; exempting that key (never any other
# segment field) keeps legitimately relay-named groups from tripping the
# member/waypoint scan.
_SEGMENT_MARKER_EXEMPT_KEYS = frozenset({"group_id"})
```

- `_find_forbidden_marker` and `_scan_value` become:

```python
def _find_forbidden_marker(dump: dict[str, object]) -> tuple[str, str] | None:
    """First (path, value) in a structural field naming members/waypoints.

    Only ``_SCANNED_STRUCTURAL_FIELDS`` are scanned — the concept,
    priorities, quality, reinforcement policies, soft constraints, and the
    segment plan — where final members or waypoints would have to appear
    if smuggled (spec 6.8). Citation fields like ``evidence_ids``
    legitimately embed producing UUV ids (e.g. ``B:T1:uuv_00:900``) and
    the free-text ``rationale`` may discuss members; both are exempt.
    """
    for field in _SCANNED_STRUCTURAL_FIELDS:
        value = dump.get(field)
        if value is not None:
            skip_keys = (
                _SEGMENT_MARKER_EXEMPT_KEYS if field == "segment_plan" else frozenset()
            )
            found = _scan_value(value, path=field, skip_keys=skip_keys)
            if found is not None:
                return found
    return None


def _scan_value(
    value: object, path: str, skip_keys: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    """First (path, value) under ``value`` whose key or string names a marker."""
    if isinstance(value, str):
        if any(marker in value.lower() for marker in _FORBIDDEN_MARKERS):
            return (path, value)
        return None
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            if key in skip_keys:
                continue
            if any(marker in str(key).lower() for marker in _FORBIDDEN_MARKERS):
                return (f"{path}.{key}", str(key))
            found = _scan_value(child, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _scan_value(child, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return None
```

- `_semantic_issues`: after the marker check, add:

```python
    segments = proposal.segment_plan
    if segments is not None:
        for index, segment in enumerate(segments.segments):
            if segment.index != index:
                issues.append(
                    _issue("segment_index_gap", f"segment_plan.segments[{index}]",
                           "segment indices must be contiguous from 0",
                           segment.index, index)
                )
            if segment.end_s <= segment.start_s:
                issues.append(
                    _issue("segment_time_invalid", f"segment_plan.segments[{index}]",
                           "segment end must follow its start",
                           segment.end_s, f"> {segment.start_s}")
                )
            if not (
                math.isfinite(segment.intercept_xy[0])
                and math.isfinite(segment.intercept_xy[1])
            ):
                issues.append(
                    _issue("non_finite", f"segment_plan.segments[{index}].intercept_xy",
                           "segment intercept must be finite",
                           segment.intercept_xy, "finite floats")
                )
```

`src/underwater_tracking/agent/nodes/optimize.py`:
- Imports gain `from underwater_tracking.domain.agent_models import PredictedTrackRef` and `from underwater_tracking.planning.segmentation import default_segment_plan, initial_intercept`.
- `optimize_candidates` gains keyword param `predictions: Mapping[str, PredictedTrackRef] | None = None`; the `_build_evaluation` call gains `predictions=predictions`; `_emergency_evaluation` gains the param and passes it through.
- `_build_evaluation` gains `predictions: Mapping[str, PredictedTrackRef] | None = None`; the waypoint loop becomes:

```python
    segment_plan = proposal.segment_plan
    for target in sorted(members_by_target):
        members = members_by_target[target]
        if not members:
            continue
        if segment_plan is None and predictions is not None:
            prediction = predictions.get(target)
            if prediction is not None:
                segment_plan = default_segment_plan(prediction, (f"G-{target}",))
        waypoints.update(
            _plan_waypoints(
                snapshot,
                target,
                members,
                config,
                previous_by_member,
                intercept=initial_intercept(segment_plan, target),
            )
        )
```

and the `TrackingPlan(...)` construction gains `segment_plan=segment_plan,`.
- `_plan_waypoints` gains keyword `intercept: tuple[float, float] | None = None`; the sigma-points argument becomes:

```python
    sigma_points = np.asarray(
        _belief_sigma_points(_belief(snapshot, target)), dtype=float
    )
    if intercept is not None:
        sigma_points = sigma_points + (
            np.asarray(intercept) - sigma_points.mean(axis=0)
        )
    result = plan_group_waypoints(
        positions,
        sigma_points,
        previous,
        ...
```

- `OptimizeNode.__call__` passes `predictions=state.get("predictions")` into `optimize_candidates`.

`src/underwater_tracking/agent/nodes/commit.py`:
- `validate_plan` gains `_check_segments(snapshot, plan, config, issues)` after `_check_waypoints`.
- Add:

```python
def _check_segments(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    config: PlanningConfig,
    issues: list[ValidationIssue],
) -> None:
    """Segment-plan sanity: contiguous indices, ordered times, bounded intercepts.

    No horizon cap is enforced: a segment may end past ``valid_until_s``,
    so relay plans remain committable while the prediction horizon exceeds
    the plan window.
    """
    segment_plan = plan.segment_plan
    if segment_plan is None:
        return
    xmin, xmax, ymin, ymax = config.bounds
    for index, segment in enumerate(segment_plan.segments):
        if segment.index != index:
            issues.append(
                ValidationIssue(
                    code="segment_index_gap",
                    field=f"segment_plan.segments[{index}]",
                    message="segment indices must be contiguous from 0",
                )
            )
        if segment.end_s <= segment.start_s:
            issues.append(
                ValidationIssue(
                    code="segment_time_invalid",
                    field=f"segment_plan.segments[{index}]",
                    message="segment end must follow its start",
                )
            )
        x, y = segment.intercept_xy
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            issues.append(
                ValidationIssue(
                    code="segment_out_of_bounds",
                    field=f"segment_plan.segments[{index}]",
                    message="segment intercept outside the scenario box",
                )
            )
        if segment.start_s < plan.valid_from_s:
            issues.append(
                ValidationIssue(
                    code="segment_past",
                    field=f"segment_plan.segments[{index}]",
                    message="segment starts before the plan window",
                )
            )
```

- `build_commands`'s `PlanCommand(...)` construction gains `sensor_mode="passive"` (committed plans stay passive; the active-sonar protocol is driven by verification commands, which the loop applies after plan commands so the protocol wins).

`src/underwater_tracking/agent/prompts.py`: `STRATEGY_PROMPT_VERSION = "strategy-v2"`, and `STRATEGY_SYSTEM_PROMPT` becomes:

```python
STRATEGY_SYSTEM_PROMPT = (
    "You are the carrier strategy officer. You convert validated intent "
    "hypotheses and trigger events into candidate strategy proposals.\n"
    "Allowed evidence: the target intent summaries, trigger events, "
    "evidence ids, and predicted_tracks summary in the payload. No other "
    "source may be used.\n"
    "Output schema purpose: produce exactly one StrategyProposal for the "
    "requested concept — target_priorities, required_quality, "
    "reinforcement_policy, releasable_soft_constraints, evidence_ids, "
    "rationale, and an optional segment_plan — with the concept from the "
    "fixed set (quality_first, balanced, resource_saving, hold_current).\n"
    "Ground-reality rule: target ground reality is never provided; base "
    "priorities and quality targets only on belief-derived intent and "
    "confidence.\n"
    "Member and waypoint prohibition: never output final group members, "
    "rotations, or waypoints; StrategyProposal carries none, and numeric "
    "assignment is solved deterministically. When the payload carries a "
    "predicted_tracks summary you MAY segment the target tracks for relay "
    "tracking: each segment names one group (its id like G-target_id), its "
    "start/end simulation times inside the prediction horizon, and the "
    "intercept point where that group initializes its standoff. Segments "
    "must be contiguous from index 0; never invent groups or targets."
)
```

**Step 3 — Live segmentation test.** Append to `tests/integration/test_llm_real_api.py` (module skipif already uses `has_live_api_key()` from T1; imports gain `StrategyProposal`, `StrategyGenerationNode`, `PredictedTrackRef`, `SegmentPlan` as needed):

```python
@pytest.mark.real_llm
def test_strategy_payload_with_predicted_tracks_yields_valid_segments(
    live_llm: HTTPStructuredLLM,
) -> None:
    """1 request: the curated payload with predicted tracks answers validly.

    Whatever the provider returns, the proposal must validate under the
    same rules; if it carries a segment_plan, the segments are contiguous
    from 0, lie inside the predicted horizon, and name the allocation set's
    group only.
    """
    node = StrategyGenerationNode(live_llm, model_id="LongCat-2.0")
    prediction = PredictedTrackRef(
        prediction_id="S1:track:T1:3",
        target_id="T1",
        sim_time_s=900,
        horizon_s=600.0,
        sample_step_s=30.0,
        times_s=(930.0, 960.0, 990.0, 1020.0, 1050.0),
        points_xy=(
            (140.0, 230.0),
            (150.0, 240.0),
            (160.0, 250.0),
            (170.0, 260.0),
            (180.0, 270.0),
        ),
        corridor_radius_m=(40.0, 42.0, 44.0, 46.0, 48.0),
    )
    payload = node.build_payload(
        {
            "scenario_id": "S1",
            "coalesced_events": (),
            "route": EventLevel.STRATEGIC,
            "intent_hypotheses": {
                "T1": IntentHypothesis(
                    label="transit",
                    confidence=0.8,
                    evidence_ids=("B:T1:870", "B:T1:900"),
                    model_id="LongCat-2.0",
                    prompt_version=INTENT_PROMPT_VERSION,
                )
            },
            "predictions": {"T1": prediction},
        },
        "balanced",
    )
    assert any(track["target_id"] == "T1" for track in payload["predicted_tracks"])
    proposal = live_llm.invoke_structured(
        "strategy", payload, StrategyProposal, prompt_version=STRATEGY_PROMPT_VERSION
    )
    assert isinstance(proposal, StrategyProposal)
    report = validate_strategy(
        proposal,
        target_ids=("T1",),
        evidence_ids=("B:T1:870", "B:T1:900"),
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert report.valid is True
    if proposal.segment_plan is not None:
        for index, segment in enumerate(proposal.segment_plan.segments):
            assert segment.index == index
            assert segment.end_s > segment.start_s
            assert 900 <= segment.start_s <= 1500
            assert segment.group_id == "G-T1"
```

Imports needed in that module: `from underwater_tracking.agent.nodes.strategy import StrategyGenerationNode`, `from underwater_tracking.agent.nodes.verify import validate_strategy`, `from underwater_tracking.agent.prompts import INTENT_PROMPT_VERSION, STRATEGY_PROMPT_VERSION`, `from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef, StrategyProposal`, `from underwater_tracking.domain.models import EventLevel`.

**Step 4 — Run and confirm it passes:**

```
PYTHONPATH=. .venv/bin/python -m pytest tests/agent/test_segmentation.py tests/agent/test_verify_graph.py tests/agent/test_llm_port.py -q
PYTHONPATH=. .venv/bin/ruff check src tests
PYTHONPATH=. .venv/bin/mypy src
```

**Step 5 — Commit.**

```
git add -A
git commit -m "feat: llm trajectory segmentation for relay tracking"
```

---

## Task T5 — Human assignment mode and active verification protocol (R4, R5)

## Task T5 Step 1 — authoritative test surface (supersedes any earlier draft of this step)

**If an earlier draft of these four test files appears earlier in this document, replace it with this version.** The files below are the binding API contract for Task T5: the executor writes exactly these files, runs them (they must fail red on import), then implements Step 2 against them. The API surface they fix is:

- `underwater_tracking.planning.reservations.ReservationRegistry` — `reserve(uuv_ids, target_id)`, `release(uuv_ids)`, `reserved_uuvs() -> frozenset[str]`, `reserved_for(target_id) -> frozenset[str]`, `is_reserved(uuv_id) -> bool`.
- `ExpertDirective` gains `directive_type: Literal["constraint", "assignment"] = "constraint"`, `assignment_target_id: str | None = None`, `assignment_uuv_ids: tuple[str, ...] = ()`.
- `underwater_tracking.agent.nodes.directives.assign_target_uuvs` — the typed assignment shortcut.
- `underwater_tracking.agent.runtime.CarrierRuntime.preview_assignment` and `.reservations()`.
- `underwater_tracking.domain.agent_models.VerificationCommand` with `sensor_mode` in `{"ping", "return_to_passive", "dispatch", "drop"}`.
- `underwater_tracking.agent.nodes.active_verification.ActiveVerificationNode(reservations, situation_provider, *, in_position_gate_m=1200.0)` with a situation provider `Callable[[str], SituationSnapshot]` and protocol constants `_STATE_VERIFYING`, `_STATE_CLASSIFIED_SUBMARINE`, `_STATE_IN_POSITION`.
- `underwater_tracking.planning.allocation.AllocationInput` gains `reserved_uuv_ids: AbstractSet[str]` (default `frozenset()`) and `AllocationInput.synthetic(..., reserved_uuv_ids=frozenset())`.

### File: `tests/planning/test_reservations.py` (new)

```python
# tests/planning/test_reservations.py
"""Reservation registry for human-assigned UUVs (spec 17.2, R4).

UUVs the operator has explicitly assigned to a target are reserved: the
allocator must never assign them elsewhere and the verification protocol
must never pick them as pingers. One registry is owned by the
CarrierRuntime and shared with the engine through
``SimulationEngine.set_reservations``; every consumer reads only the
immutable ``reserved_uuvs()`` projection.
"""

import pytest

from underwater_tracking.planning.reservations import ReservationRegistry


def test_reserve_release_round_trip() -> None:
    registry = ReservationRegistry()
    registry.reserve(("uuv_01", "uuv_02"), "T2")
    assert registry.reserved_uuvs() == frozenset({"uuv_01", "uuv_02"})
    assert registry.reserved_for("T2") == frozenset({"uuv_01", "uuv_02"})
    assert registry.is_reserved("uuv_01") is True
    assert registry.is_reserved("uuv_03") is False
    registry.release(("uuv_01",))
    assert registry.reserved_uuvs() == frozenset({"uuv_02"})
    assert registry.is_reserved("uuv_01") is False
    registry.release(("uuv_02",))
    assert registry.reserved_uuvs() == frozenset()
    assert registry.reserved_for("T2") == frozenset()


def test_reserving_for_another_target_is_rejected() -> None:
    registry = ReservationRegistry()
    registry.reserve(("uuv_01",), "T2")
    with pytest.raises(ValueError):
        registry.reserve(("uuv_01",), "T3")


def test_reserving_the_same_target_again_is_idempotent() -> None:
    registry = ReservationRegistry()
    registry.reserve(("uuv_01",), "T2")
    registry.reserve(("uuv_01",), "T2")
    assert registry.reserved_for("T2") == frozenset({"uuv_01"})
```

### File: `tests/agent/test_assignment_directives.py` (new)

```python
# tests/agent/test_assignment_directives.py
"""Human assignment directives (spec 17.2, R4).

``assign_target_uuvs`` builds an assignment directive that reserves UUVs
for one target; validation rejects unknown ids and empty assignments and
conflicts against other applied assignments; applying the preview through
the runtime reserves the UUVs immediately, so the allocator and the
verification protocol both exclude them. No LLM is involved anywhere in
this module (the typed shortcut never parses).
"""

from pathlib import Path

import pytest

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.nodes.directives import (
    assign_target_uuvs,
    validate_directive,
)
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.simulation.clock import SimulationClock


def _uuv(uuv_id: str, x: float, y: float) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(x, y),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=1.0,
        status=UUVStatus.AVAILABLE,
        group_id=None,
    )


def _report(target_id: str, members: tuple[str, ...]) -> GroupReport:
    return GroupReport(
        group_id=f"G-{target_id}",
        target_id=target_id,
        sim_time_s=900,
        member_ids=members,
        belief=TargetBelief(
            target_id=target_id,
            sim_time_s=900,
            mean=(130.0, 220.0, 1.0, 0.5),
            covariance=(
                (400.0, 0.0, 0.0, 0.0),
                (0.0, 400.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 0.7, "ct": 0.3},
            source_observation_ids=(f"B:{target_id}:900",),
            fim_min_eigenvalue=0.005,
            fim_condition=12.0,
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.75,
            ewma=0.76,
            components={"cov": 0.7},
            hard_guard_reasons=(),
        ),
        plan_revision=1,
    )


def _situation() -> SituationSnapshot:
    """Two tracked targets plus four healthy UUVs."""
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=900,
        uuvs=(
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 500.0, 500.0),
            _uuv("uuv_04", 700.0, 700.0),
        ),
        group_reports=(
            _report("T1", ("uuv_01", "uuv_02")),
            _report("T2", ("uuv_03",)),
        ),
        pending_events=(),
    )


def test_assignment_shortcut_resolves_to_preview() -> None:
    directive = assign_target_uuvs(
        directive_id="S1:assign:T1:uuv_03,uuv_04",
        uuv_ids=("uuv_03", "uuv_04"),
        target_id="T1",
        situation=_situation(),
    )
    assert directive.directive_type == "assignment"
    assert directive.assignment_target_id == "T1"
    assert directive.assignment_uuv_ids == ("uuv_03", "uuv_04")
    assert directive.status == "preview"
    assert directive.conflicts == ()


def test_assignment_rejects_unknown_ids_and_empty_assignments() -> None:
    situation = _situation()
    unknown_target = assign_target_uuvs(
        directive_id="S1:assign:T9:uuv_03",
        uuv_ids=("uuv_03",),
        target_id="T9",
        situation=situation,
    )
    assert unknown_target.status == "needs_clarification"
    assert any("unknown_target" in issue for issue in unknown_target.conflicts)
    unknown_uuv = assign_target_uuvs(
        directive_id="S1:assign:T1:uuv_99",
        uuv_ids=("uuv_99",),
        target_id="T1",
        situation=situation,
    )
    assert unknown_uuv.status == "needs_clarification"
    assert any("unknown_uuv" in issue for issue in unknown_uuv.conflicts)
    empty = assign_target_uuvs(
        directive_id="S1:assign:T1:",
        uuv_ids=(),
        target_id="T1",
        situation=situation,
    )
    assert empty.status == "needs_clarification"
    assert any("empty_assignment" in issue for issue in empty.conflicts)


def test_assignment_conflicts_with_an_applied_assignment() -> None:
    situation = _situation()
    applied = validate_directive(
        assign_target_uuvs(
            directive_id="S1:assign:T1:uuv_03",
            uuv_ids=("uuv_03",),
            target_id="T1",
            situation=situation,
        ),
        situation=situation,
    )
    assert applied.status == "preview"
    conflicting = assign_target_uuvs(
        directive_id="S1:assign:T2:uuv_03",
        uuv_ids=("uuv_03",),
        target_id="T2",
        situation=situation,
        applied_directives=(applied,),
    )
    assert conflicting.status == "needs_clarification"
    assert any("uuv_03" in issue for issue in conflicting.conflicts)
    # Re-assigning the same target is idempotent, not a conflict.
    same_target = assign_target_uuvs(
        directive_id="S1:assign:T1:uuv_03,uuv_04",
        uuv_ids=("uuv_03", "uuv_04"),
        target_id="T1",
        situation=situation,
        applied_directives=(applied,),
    )
    assert same_target.status == "preview"


class _NeverLLM:
    """Stands in for the structured LLM port: the assignment flow never calls it."""

    def invoke_structured(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the assignment flow must not call the LLM")


def _make_runtime(tmp_path: Path, situation: SituationSnapshot) -> CarrierRuntime:
    database_path = tmp_path / "assign.db"
    dependencies = CarrierDependencies(
        plans=PlanRepository(database_path),
        events=EventRepository(database_path),
        ledger=DecisionLedger(database_path),
        llm=_NeverLLM(),  # type: ignore[arg-type]
        predictor=lambda situation, target_id: None,  # type: ignore[arg-type]
        situation_provider=lambda ref: situation,
        clock=SimulationClock(step_s=30),
    )
    return CarrierRuntime(
        dependencies, scenario_id="S1", database_path=database_path
    )


def test_runtime_apply_assignment_reserves_the_uuvs(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, _situation())
    try:
        preview = runtime.preview_assignment(
            uuv_ids=("uuv_03", "uuv_04"), target_id="T1"
        )
        assert preview.status == "preview"
        applied = runtime.apply_directive(preview.directive_id)
        assert applied.status == "applied"
        assert runtime.reservations().reserved_uuvs() == frozenset(
            {"uuv_03", "uuv_04"}
        )
        assert runtime.reservations().reserved_for("T1") == frozenset(
            {"uuv_03", "uuv_04"}
        )
    finally:
        runtime.close()
```

### File: `tests/agent/test_active_verification.py` (new)

```python
# tests/agent/test_active_verification.py
"""Deterministic active-sonar verification protocol (spec 17.3, R5).

The node is pure graph logic; the engine's ping simulation is out of
scope (the active-sonar task). These tests drive the state machine with
explicit events and assert the emitted verification commands and the
per-contact state transitions — nearest-first pinger selection, reserved
and unavailable UUV exclusion, submarine dispatch, decoy drop, and the
geometric in-position gate.
"""

from underwater_tracking.agent.nodes.active_verification import (
    ActiveVerificationNode,
    _STATE_CLASSIFIED_SUBMARINE,
    _STATE_IN_POSITION,
    _STATE_VERIFYING,
)
from underwater_tracking.domain.models import (
    EventLevel,
    GroupQuality,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.planning.reservations import ReservationRegistry


def _uuv(
    uuv_id: str, x: float, y: float, *, status: UUVStatus = UUVStatus.AVAILABLE
) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(x, y),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=1.0,
        status=status,
        group_id=None,
    )


def _report(target_id: str, member_ids: tuple[str, ...]) -> GroupReport:
    return GroupReport(
        group_id=f"G-{target_id}",
        target_id=target_id,
        sim_time_s=1200,
        member_ids=member_ids,
        belief=TargetBelief(
            target_id=target_id,
            sim_time_s=1200,
            mean=(130.0, 220.0, 1.0, 0.5),
            covariance=(
                (400.0, 0.0, 0.0, 0.0),
                (0.0, 400.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            model_probabilities={"cv": 0.7, "ct": 0.3},
            source_observation_ids=(f"B:{target_id}:1200",),
            fim_min_eigenvalue=0.005,
            fim_condition=12.0,
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.75,
            ewma=0.76,
            components={"cov": 0.7},
            hard_guard_reasons=(),
        ),
        plan_revision=1,
    )


def _situation(
    uuvs: tuple[UUVState, ...], reports: tuple[GroupReport, ...] = ()
) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=1200,
        uuvs=uuvs,
        group_reports=reports,
        pending_events=(),
    )


def _event(event_type: str, target_id: str, *, payload: dict) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"S1:{event_type}:{target_id}:1200",
        scenario_id="S1",
        sim_time_s=1200,
        event_type=event_type,
        entity_id=target_id,
        level=EventLevel.INFORMATIONAL,
        payload=payload,
    )


def _active_ping(target_id: str, x: float, y: float) -> RuntimeEvent:
    return _event("active_ping", target_id, payload={"position_xy": (x, y)})


def _classified(target_id: str, outcome: str) -> RuntimeEvent:
    return _event("contact_classified", target_id, payload={"outcome": outcome})


def _run(
    node: ActiveVerificationNode, *events: RuntimeEvent
) -> dict[str, object]:
    return node(
        {
            "snapshot_ref": "S1:live",
            "coalesced_events": events,
        }
    )


def test_contact_pings_the_nearest_available_uuv() -> None:
    situation = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 500.0, 500.0),
            _uuv("uuv_03", 1200.0, 1200.0),
        )
    )
    node = ActiveVerificationNode(None, lambda ref: situation)
    result = _run(node, _active_ping("C1", 120.0, 110.0))
    assert result["verification_states"] == {"C1": _STATE_VERIFYING}
    assert result["verification_pingers"] == {"C1": "uuv_01"}
    commands = result["verification_commands"]
    assert len(commands) == 1
    assert commands[0].sensor_mode == "ping"
    assert commands[0].target_id == "C1"
    assert commands[0].uuv_ids == ("uuv_01",)


def test_reserved_uuvs_are_never_picked_as_pingers() -> None:
    reservations = ReservationRegistry()
    reservations.reserve(("uuv_01",), "T1")
    situation = _situation(
        (_uuv("uuv_01", 100.0, 100.0), _uuv("uuv_02", 500.0, 500.0))
    )
    node = ActiveVerificationNode(reservations, lambda ref: situation)
    result = _run(node, _active_ping("C1", 110.0, 110.0))
    assert result["verification_commands"][0].uuv_ids == ("uuv_02",)


def test_failed_and_busy_uuvs_are_skipped() -> None:
    situation = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0, status=UUVStatus.FAILED),
            _uuv("uuv_02", 110.0, 110.0),
        )
    )
    node = ActiveVerificationNode(None, lambda ref: situation)
    first = _run(node, _active_ping("C0", 105.0, 105.0))
    assert first["verification_pingers"] == {"C0": "uuv_02"}
    # uuv_02 is busy pinging C0, so C1 must wait (no command, no state).
    second = _run(node, _active_ping("C1", 105.0, 105.0))
    assert second["verification_commands"] == ()
    assert "C1" not in second["verification_states"]


def test_repeated_ping_events_are_idempotent() -> None:
    situation = _situation((_uuv("uuv_01", 100.0, 100.0),))
    node = ActiveVerificationNode(None, lambda ref: situation)
    first = _run(node, _active_ping("C1", 110.0, 110.0))
    assert first["verification_commands"] != ()
    second = _run(node, _active_ping("C1", 110.0, 110.0))
    assert second["verification_commands"] == ()


def test_submarine_classification_dispatches_and_closes_the_gate() -> None:
    holder: dict[str, SituationSnapshot] = {}
    holder["situation"] = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 2000.0, 2000.0),
        )
    )
    node = ActiveVerificationNode(None, lambda ref: holder["situation"])
    first = _run(node, _active_ping("T2", 250.0, 250.0))
    assert first["verification_commands"][0].sensor_mode == "ping"
    assert first["verification_pingers"] == {"T2": "uuv_02"}
    second = _run(node, _classified("T2", "submarine"))
    assert second["verification_states"] == {"T2": _STATE_CLASSIFIED_SUBMARINE}
    dispatch = [
        command
        for command in second["verification_commands"]
        if command.sensor_mode == "dispatch"
    ]
    assert len(dispatch) == 1
    assert dispatch[0].target_id == "T2"
    # The dispatched group forms but uuv_03 is still 2582 m from the belief
    # mean at (130, 220), beyond the 1200 m gate.
    holder["situation"] = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 2000.0, 2000.0),
        ),
        reports=(_report("T2", ("uuv_01", "uuv_03")),),
    )
    not_yet = _run(node)
    assert not_yet["verification_states"] == {"T2": _STATE_CLASSIFIED_SUBMARINE}
    assert not_yet["verification_commands"] == ()
    # uuv_03 closes the gate (604 m to the mean at (130, 220)); both members
    # and the original pinger uuv_02 return to passive.
    holder["situation"] = _situation(
        (
            _uuv("uuv_01", 100.0, 100.0),
            _uuv("uuv_02", 300.0, 300.0),
            _uuv("uuv_03", 600.0, 600.0),
        ),
        reports=(_report("T2", ("uuv_01", "uuv_03")),),
    )
    closed = _run(node)
    assert closed["verification_states"] == {"T2": _STATE_IN_POSITION}
    passive = [
        command
        for command in closed["verification_commands"]
        if command.sensor_mode == "return_to_passive"
    ]
    assert len(passive) == 1
    assert set(passive[0].uuv_ids) == {"uuv_01", "uuv_02", "uuv_03"}
    assert closed["verification_pingers"] == {}


def test_decoy_classification_drops_and_releases_the_pinger() -> None:
    situation = _situation((_uuv("uuv_01", 100.0, 100.0),))
    node = ActiveVerificationNode(None, lambda ref: situation)
    _run(node, _active_ping("C3", 110.0, 110.0))
    result = _run(node, _classified("C3", "decoy"))
    modes = {command.sensor_mode for command in result["verification_commands"]}
    assert modes == {"drop", "return_to_passive"}
    assert "C3" not in result["verification_states"]
    assert result["verification_pingers"] == {}
```

### File: `tests/planning/test_allocation.py` — append these two tests

Append at the end of `tests/planning/test_allocation.py`:

```python
def test_reserved_uuvs_are_never_assigned():
    """Human-assigned UUVs (spec 17.2) are excluded from every solution."""
    problem = AllocationInput.synthetic(
        uuv_count=6,
        target_count=2,
        reserved_uuv_ids=frozenset({"uuv_1", "uuv_4"}),
    )
    solution = allocate_groups(problem)
    assert solution.solver_status == "milp"
    for members in solution.members_by_target.values():
        assert not (set(members) & {"uuv_1", "uuv_4"})


def test_fully_reserved_problem_cannot_form_a_group():
    problem = AllocationInput.synthetic(
        uuv_count=2,
        target_count=1,
        reserved_uuv_ids=frozenset({"uuv_0", "uuv_1"}),
    )
    solution = allocate_groups(problem)
    assert solution.members_by_target.get("target_0", ()) == ()
```

Then run-and-fail:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/planning/test_reservations.py tests/agent/test_assignment_directives.py tests/agent/test_active_verification.py tests/planning/test_allocation.py -q
```

Expected: FAIL — `ImportError` for `underwater_tracking.planning.reservations` and `underwater_tracking.agent.nodes.active_verification`, `AttributeError`/`TypeError` for the new `ExpertDirective`/`VerificationCommand`/`AllocationInput` fields.

## Task T5 Step 2 — implementation (complete edit list)

**The edit list below is the complete, authoritative Step 2 for Task T5. If an earlier partial Step 2 appears earlier in this document, ignore it and apply only the edits below, in order.**

### 2.1 New file: `src/underwater_tracking/planning/reservations.py`

```python
# src/underwater_tracking/planning/reservations.py
"""Human-assignment reservation registry (spec 17.2, R4).

UUVs the operator has explicitly assigned to a target are reserved: the
allocator never assigns them elsewhere (``AllocationInput.reserved_uuv_ids``)
and the verification protocol never picks them as pingers. One registry is
owned by the CarrierRuntime and shared with the engine through
``SimulationEngine.set_reservations``; every consumer reads only the
immutable ``reserved_uuvs()`` projection, so the registry itself stays a
plain in-memory map.
"""

from __future__ import annotations

from collections.abc import Iterable


class ReservationRegistry:
    """In-memory map of uuv_id -> target_id with reverse lookup.

    A UUV is reserved for at most one target at a time; re-reserving the
    same UUV for a different target raises ValueError. The directive
    conflict validator rejects this earlier; the registry is the final
    guard.
    """

    def __init__(self) -> None:
        self._by_uuv: dict[str, str] = {}
        self._by_target: dict[str, set[str]] = {}

    def reserve(self, uuv_ids: Iterable[str], target_id: str) -> None:
        """Reserve every named UUV for ``target_id`` (idempotent per target)."""
        for uuv_id in uuv_ids:
            current = self._by_uuv.get(uuv_id)
            if current is not None and current != target_id:
                raise ValueError(
                    f"uuv {uuv_id!r} is already reserved for {current!r}"
                )
            self._by_uuv[uuv_id] = target_id
            self._by_target.setdefault(target_id, set()).add(uuv_id)

    def release(self, uuv_ids: Iterable[str]) -> None:
        """Release every named UUV (unknown ids are ignored)."""
        for uuv_id in uuv_ids:
            target_id = self._by_uuv.pop(uuv_id, None)
            if target_id is None:
                continue
            reserved = self._by_target.get(target_id)
            if reserved is not None:
                reserved.discard(uuv_id)
                if not reserved:
                    del self._by_target[target_id]

    def reserved_uuvs(self) -> frozenset[str]:
        """The frozen projection every consumer reads (never mutated)."""
        return frozenset(self._by_uuv)

    def reserved_for(self, target_id: str) -> frozenset[str]:
        """Every UUV currently reserved for ``target_id``."""
        return frozenset(self._by_target.get(target_id, ()))

    def is_reserved(self, uuv_id: str) -> bool:
        return uuv_id in self._by_uuv
```

### 2.2 Edit: `src/underwater_tracking/planning/allocation.py`

Edit A — add the field. Replace:

```python
    uuv_available: Mapping[str, bool] = field(default_factory=dict)
```

with:

```python
    uuv_available: Mapping[str, bool] = field(default_factory=dict)
    reserved_uuv_ids: AbstractSet[str] = frozenset()
```

Edit B — validate the new field in `__post_init__`. Insert immediately after the existing `uuv_available` validation loop (the `for uuv in self.uuv_available: ...` block):

```python
        for uuv in self.reserved_uuv_ids:
            if uuv not in uuvs:
                raise ValueError(f"reserved_uuv_ids mentions unknown uuv {uuv!r}")
```

Edit C — exclude reserved UUVs from every availability gate. Replace the whole function:

```python
def _available(problem: AllocationInput, uuv: str) -> bool:
    return problem.uuv_available.get(uuv, True)
```

with:

```python
def _available(problem: AllocationInput, uuv: str) -> bool:
    if uuv in problem.reserved_uuv_ids:
        return False
    return problem.uuv_available.get(uuv, True)
```

Edit D — the `synthetic` builder gains the field. Replace the whole classmethod `synthetic` (its signature and the `cls(...)` call) with:

```python
    @classmethod
    def synthetic(
        cls,
        uuv_count: int = 6,
        target_count: int = 2,
        feasible_pair_quality: float = 0.8,
        reserved_uuv_ids: AbstractSet[str] = frozenset(),
    ) -> AllocationInput:
        """Build a deterministic problem where every pair is feasible.

        ``feasible_pair_quality`` becomes the quality of every target;
        all UUVs are available with full energy, all economic costs are
        zero, and there is no prior assignment. ``reserved_uuv_ids`` are
        excluded from assignment.
        """
        target_ids = tuple(f"target_{i}" for i in range(target_count))
        return cls(
            uuv_ids=tuple(f"uuv_{i}" for i in range(uuv_count)),
            target_ids=target_ids,
            quality_by_target={target: feasible_pair_quality for target in target_ids},
            reserved_uuv_ids=reserved_uuv_ids,
        )
```

### 2.3 Edit: `src/underwater_tracking/domain/agent_models.py`

Edit A — `PlanCommand` gains `sensor_mode`. Replace the whole class body of `PlanCommand` (from `class PlanCommand(StrictModel):` through the `actions` field) with:

```python
class PlanCommand(StrictModel):
    """Versioned per-group execution command (spec 5.2)."""

    command_id: str
    plan_id: str
    plan_revision: int = Field(ge=1)
    scenario_id: str
    group_id: str
    target_id: str
    sim_time_s: int = Field(ge=0)
    member_ids: tuple[str, ...] = ()
    waypoints_by_member: dict[str, tuple[Waypoint, ...]] = Field(default_factory=dict)
    actions: dict[str, str] = Field(default_factory=dict)
    sensor_mode: Literal["active", "passive"] = "passive"
```

Edit B — insert the `VerificationCommand` class immediately after the `PlanCommand` class (before `class ValidationIssue`):

```python
class VerificationCommand(StrictModel):
    """Engine-facing active-sonar verification protocol command (spec 17.3).

    ``sensor_mode`` drives the engine's ping simulation: ``ping`` turns
    active sonar on for ``uuv_ids``, ``return_to_passive`` turns it off,
    ``dispatch`` promotes a verified submarine contact into the tracking
    loop, and ``drop`` discards a classified decoy. Commands are emitted
    by the deterministic verification node and applied by the agent loop
    after plan commands, so the protocol's sensor-mode writes win.
    """

    command_id: str
    target_id: str
    sensor_mode: Literal["ping", "return_to_passive", "dispatch", "drop"]
    uuv_ids: tuple[str, ...] = ()
    sim_time_s: int = 0
```

Edit C — `ExpertDirective` gains the assignment fields. Replace the whole class body of `ExpertDirective` (from `class ExpertDirective(StrictModel):` through the `status` field) with:

```python
class ExpertDirective(StrictModel):
    directive_id: str
    raw_text: str
    target_scope: tuple[str, ...]
    locked_members: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    # The RUF012 tags on the two defaults below are intentional: pydantic
    # deep-copies mutable defaults per instance, so the pattern is safe.
    target_priorities: dict[str, float] = {}  # noqa: RUF012
    minimum_quality: dict[str, float] = {}  # noqa: RUF012
    disabled_uuv_ids: tuple[str, ...] = ()
    directive_type: Literal["constraint", "assignment"] = "constraint"
    assignment_target_id: str | None = None
    assignment_uuv_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    conflicts: tuple[str, ...] = ()
    status: Literal["preview", "applied", "rejected", "needs_clarification"] = "preview"
```

### 2.4 Edit: `src/underwater_tracking/agent/nodes/event_monitor.py`

Replace the whole `_INFORMATIONAL_TYPES` frozenset literal with:

```python
_INFORMATIONAL_TYPES: frozenset[str] = frozenset({
    "progress_report",
    "question",
    "state_changed",
    "repair_applied",
    "active_ping",
    "contact_classified",
})
```

The two new event types are informational: the verification protocol is deterministic and never routes the LLM strategic chain.

### 2.5 Edit: `src/underwater_tracking/agent/nodes/directives.py`

Edit A — `_has_any_constraint` counts assignment directives. Replace the whole function:

```python
def _has_any_constraint(directive: ExpertDirective) -> bool:
    return bool(
        directive.directive_type == "assignment"
        or directive.target_scope
        or directive.locked_members
        or directive.target_priorities
        or directive.minimum_quality
        or directive.disabled_uuv_ids
    )
```

Edit B — `_id_and_resource_issues` gains the assignment branch. Replace the whole function with:

```python
def _id_and_resource_issues(
    directive: ExpertDirective, situation: SituationSnapshot
) -> list[str]:
    """Unknown IDs and out-of-bounds resources as deterministic issue strings."""
    issues: list[str] = []
    known_targets = {report.target_id for report in situation.group_reports}
    known_uuvs = {uuv.uuv_id for uuv in situation.uuvs}
    for target_id in sorted(set(directive.target_scope) - known_targets):
        issues.append(f"unknown_target {target_id!r}: no group report for it")
    for target_id, members in sorted(directive.locked_members.items()):
        if target_id not in known_targets:
            issues.append(f"unknown_target {target_id!r}: locked members group missing")
        for member_id in sorted(set(members) - known_uuvs):
            issues.append(f"unknown_member {member_id!r}: no resource state for it")
    for uuv_id in sorted(set(directive.disabled_uuv_ids) - known_uuvs):
        issues.append(f"unknown_uuv {uuv_id!r}: no resource state for it")
    for target_id, priority in sorted(directive.target_priorities.items()):
        if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
            issues.append(
                f"invalid_priority {target_id!r}: {priority!r} outside [0, 1]"
            )
    for target_id, quality in sorted(directive.minimum_quality.items()):
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            issues.append(
                f"invalid_quality {target_id!r}: {quality!r} outside [0, 1]"
            )
    if directive.directive_type == "assignment":
        if directive.assignment_target_id is None:
            issues.append("ambiguous_scope: assignment names no target")
        elif directive.assignment_target_id not in known_targets:
            issues.append(
                f"unknown_target {directive.assignment_target_id!r}: no group report for it"
            )
        if not directive.assignment_uuv_ids:
            issues.append("empty_assignment: at least one UUV must be assigned")
        for uuv_id in sorted(set(directive.assignment_uuv_ids) - known_uuvs):
            issues.append(f"unknown_uuv {uuv_id!r}: no resource state for it")
    locked = {
        member
        for members in directive.locked_members.values()
        for member in members
    }
    for uuv_id in sorted(set(directive.disabled_uuv_ids) & locked):
        issues.append(f"internal_conflict: uuv {uuv_id!r} is both locked and disabled")
    return issues
```

Edit C — `_conflict_issues` gains the assignment-conflict branch. Replace the whole function with:

```python
def _conflict_issues(
    directive: ExpertDirective, applied_directives: Sequence[ExpertDirective]
) -> list[str]:
    """Hard-constraint conflicts against the applied directives."""
    issues: list[str] = []
    for other in applied_directives:
        for target_id, members in sorted(directive.locked_members.items()):
            other_members = other.locked_members.get(target_id)
            if other_members is not None and set(other_members) != set(members):
                issues.append(
                    f"conflicts with applied {other.directive_id}: locked members"
                    f" of {target_id!r} differ"
                )
        for target_id, priority in sorted(directive.target_priorities.items()):
            if (
                target_id in other.target_priorities
                and other.target_priorities[target_id] != priority
            ):
                issues.append(
                    f"conflicts with applied {other.directive_id}: priority of"
                    f" {target_id!r} differs"
                )
        for target_id, quality in sorted(directive.minimum_quality.items()):
            if (
                target_id in other.minimum_quality
                and other.minimum_quality[target_id] != quality
            ):
                issues.append(
                    f"conflicts with applied {other.directive_id}: minimum quality"
                    f" of {target_id!r} differs"
                )
        locked_elsewhere = {
            member
            for members in other.locked_members.values()
            for member in members
        }
        for uuv_id in sorted(set(directive.disabled_uuv_ids) & locked_elsewhere):
            issues.append(
                f"conflicts with applied {other.directive_id}: uuv {uuv_id!r}"
                " is locked by it"
            )
    if directive.directive_type == "assignment":
        for other in applied_directives:
            if other.directive_type != "assignment":
                continue
            if other.assignment_target_id == directive.assignment_target_id:
                continue  # re-assigning the same target is idempotent
            for uuv_id in sorted(
                set(directive.assignment_uuv_ids) & set(other.assignment_uuv_ids)
            ):
                issues.append(
                    f"conflicts with applied {other.directive_id}: uuv {uuv_id!r} is"
                    f" assigned to {other.assignment_target_id!r}"
                )
        occupied = {
            member
            for other in applied_directives
            for members in other.locked_members.values()
            for member in members
        } | {
            uuv_id
            for other in applied_directives
            for uuv_id in other.disabled_uuv_ids
        }
        for uuv_id in sorted(set(directive.assignment_uuv_ids) & occupied):
            issues.append(
                f"conflicts with applied directives: uuv {uuv_id!r} is locked or disabled"
            )
    return issues
```

Edit D — new typed shortcut `assign_target_uuvs`, inserted after `disable_uuv`:

```python
def assign_target_uuvs(
    *,
    directive_id: str,
    uuv_ids: Sequence[str],
    target_id: str,
    situation: SituationSnapshot,
    confidence: float = 1.0,
    applied_directives: Sequence[ExpertDirective] = (),
) -> ExpertDirective:
    """Typed shortcut: reserve ``uuv_ids`` for one target (spec 17.2).

    The assigned UUVs are excluded from the LLM-allocatable set and from
    the verification protocol's pinger pool for as long as the directive
    stays applied.
    """
    return validate_directive(
        ExpertDirective(
            directive_id=directive_id,
            raw_text=f"assignment: {', '.join(uuv_ids)} -> {target_id}",
            target_scope=(target_id,),
            directive_type="assignment",
            assignment_target_id=target_id,
            assignment_uuv_ids=tuple(uuv_ids),
            confidence=confidence,
            status="preview",
        ),
        situation=situation,
        applied_directives=applied_directives,
    )
```

This completes the directives.py edit that was interrupted earlier in this document.

### 2.6 Edit: `src/underwater_tracking/agent/runtime.py`

Edit A — imports. In the `from dataclasses import ...` area, add `replace` (there is no dataclasses import yet; add `from dataclasses import replace` after the `collections.abc` import). In the directives import block, add `assign_target_uuvs`. Change `from collections.abc import Mapping` to `from collections.abc import Mapping, Sequence`. Add `from underwater_tracking.planning.reservations import ReservationRegistry` after the `underwater_tracking.persistence.checkpoints` import.

Edit B — `__init__` owns the registry. Replace the beginning of `__init__`:

```python
        self._dependencies = dependencies
        self._scenario_id = scenario_id
```

with:

```python
        reservations = (
            dependencies.reservations
            if dependencies.reservations is not None
            else ReservationRegistry()
        )
        dependencies = replace(dependencies, reservations=reservations)
        self._dependencies = dependencies
        self._reservations = reservations
        self._scenario_id = scenario_id
```

(`CarrierDependencies` is a frozen dataclass, so the registry is injected with `dataclasses.replace` before the graph is built; the graph's verification node and the runtime share the same object.)

Edit C — new accessor, inserted after `active_plan`:

```python
    def reservations(self) -> ReservationRegistry:
        """The scenario's human-assignment reservation registry (spec 17.2)."""
        return self._reservations
```

Edit D — new typed preview, inserted after `preview_directive`:

```python
    def preview_assignment(
        self, *, uuv_ids: Sequence[str], target_id: str
    ) -> ExpertDirective:
        """Typed assignment preview: one directive reserving ``uuv_ids``.

        Unlike ``preview_directive`` there is no LLM parse: the assignment
        is a deterministic typed operation, so the preview is built by the
        typed shortcut, validated against the live situation, persisted,
        and returned for the expert's explicit confirmation.
        """
        scenario_id = self._scenario_id
        situation = self._dependencies.situation_provider(
            live_situation_ref(scenario_id)
        )
        applied = self._dependencies.ledger.list_directives(
            scenario_id, status="applied"
        )
        directive = assign_target_uuvs(
            directive_id=(
                f"{scenario_id}:assign:{target_id}:{','.join(sorted(uuv_ids))}"
            ),
            uuv_ids=uuv_ids,
            target_id=target_id,
            situation=situation,
            applied_directives=applied,
        )
        self._dependencies.ledger.save_directive(directive, scenario_id)
        return directive
```

Edit E — `apply_directive` reserves on assignment. Replace the success-path block:

```python
        applied = ExpertDirective.model_validate(
            {**preview.model_dump(mode="json"), "status": "applied"}
        )
        self._dependencies.ledger.save_directive(applied, scenario_id)
        self.submit_event(
```

with:

```python
        applied = ExpertDirective.model_validate(
            {**preview.model_dump(mode="json"), "status": "applied"}
        )
        self._dependencies.ledger.save_directive(applied, scenario_id)
        if applied.directive_type == "assignment":
            assigned_target = applied.assignment_target_id
            assert assigned_target is not None, "a clean assignment names its target"
            self._reservations.reserve(applied.assignment_uuv_ids, assigned_target)
        self.submit_event(
```

The reservation takes effect immediately (the allocator and the engine read the registry on the next cycle); the applied directive joins the ledger so the next snapshot's `applied_directives` excludes the UUVs from the LLM-allocatable set as well.

## Task T5 Step 2 (continued) — items 2.7 through 2.13

### 2.7 Edit: `src/underwater_tracking/agent/state.py`

Edit A — the agent_models import block gains `VerificationCommand`. Replace:

```python
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PredictedTrackRef,
    StrategySet,
)
```

with:

```python
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PredictedTrackRef,
    StrategySet,
    VerificationCommand,
)
```

Edit B — append the verification channels after the last channel. Replace the final line:

```python
    output_messages: tuple[str, ...]
```

with:

```python
    output_messages: tuple[str, ...]
    # Active-sonar verification protocol (spec 17.3): per-contact protocol
    # state, the UUV id pinging each contact, and the engine commands.
    verification_states: dict[str, str]
    verification_pingers: dict[str, str]
    verification_commands: tuple[VerificationCommand, ...]
```

### 2.8 Edit: `src/underwater_tracking/agent/prompts.py`

Edit A — bump the directive prompt version. Replace:

```python
DIRECTIVE_PROMPT_VERSION = "directive-v1"
```

with:

```python
DIRECTIVE_PROMPT_VERSION = "directive-v2"
```

Edit B — the directive system prompt teaches the assignment directive type (spec 17.2). Replace the whole `DIRECTIVE_SYSTEM_PROMPT` block (from `DIRECTIVE_SYSTEM_PROMPT = (` through the closing `)`) with:

```python
DIRECTIVE_SYSTEM_PROMPT = (
    "You are the carrier directive parser. You translate an expert's "
    "free-text instruction into a structured ExpertDirective preview.\n"
    "Allowed evidence: the expert instruction text and the scenario "
    "identifiers in the payload. No other source may be used.\n"
    "Output schema purpose: produce an ExpertDirective with target_scope, "
    "locked_members, target_priorities, minimum_quality, disabled_uuv_ids, "
    "directive_type, assignment_target_id, assignment_uuv_ids, confidence, "
    "conflicts, and status; ambiguous or low-confidence instructions must "
    "be previewed as needs_clarification and never applied. An instruction "
    "that reserves specific UUVs for one target is an assignment directive "
    "(directive_type \"assignment\" with assignment_uuv_ids and "
    "assignment_target_id); all other directives are constraint "
    "directives.\n"
    "Ground-reality rule: hidden ground reality is never an input; only the "
    "expert's stated constraints may enter the directive.\n"
    "Member and waypoint prohibition: never invent waypoints or complete "
    "assignments; locked_members may only repeat members the expert named."
)
```

The replacement keeps the prompt free of the contiguous substring "truth".

### 2.9 New file: `src/underwater_tracking/agent/nodes/active_verification.py`

```python
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
        """Nearest AVAILABLE non-reserved UUV not already pinging."""
        position = event.payload.get("position_xy")
        if not isinstance(position, Sequence):
            return None
        try:
            x = float(position[0])
            y = float(position[1])
        except (TypeError, IndexError, ValueError):
            return None
        reserved = (
            self._reservations.reserved_uuvs()
            if self._reservations is not None
            else frozenset()
        )
        busy = set(self._pingers.values())
        candidates = [
            uuv
            for uuv in situation.uuvs
            if uuv.status == UUVStatus.AVAILABLE
            and uuv.uuv_id not in reserved
            and uuv.uuv_id not in busy
        ]
        if not candidates:
            return None
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
```

### 2.10 Edit: `src/underwater_tracking/agent/graphs/central.py`

Edit A — two imports. Insert `from underwater_tracking.agent.nodes.active_verification import ActiveVerificationNode` immediately before the line `from underwater_tracking.agent.nodes.commit import CommitNode, validate_plan`, and `from underwater_tracking.planning.reservations import ReservationRegistry` immediately after the line `from underwater_tracking.persistence.plans import PlanRepository`.

Edit B — the dependencies gain the shared registry. Replace the final field line of `CarrierDependencies`:

```python
    model_id: str = "underwater-assistant-model"
```

with:

```python
    model_id: str = "underwater-assistant-model"
    reservations: ReservationRegistry | None = None
```

Edit C — new router after `_route_question_branch` (after its closing return line `return _route_events(state)` at the end of that function; place the new function between `_route_question_branch` and `_route_after_prediction`):

```python
def _route_after_verification(state: CentralState) -> Literal["informational", "error"]:
    """After active-sonar verification: defer branch errors, then record.

    The verification node (spec 17.3) is deterministic and never routes
    the LLM chain; a missing situation reference defers to
    ``handle_error`` while clean cycles continue to record and report.
    """
    if state.get("node_error") is not None:
        return "error"
    return "informational"
```

Edit D — node registration. Replace:

```python
    builder.add_node("question_branch", QuestionBranchNode(dependencies.ledger))
```

with:

```python
    builder.add_node("question_branch", QuestionBranchNode(dependencies.ledger))
    builder.add_node(
        "active_verification",
        ActiveVerificationNode(
            dependencies.reservations, dependencies.situation_provider
        ),
    )
```

Edit E — the informational tier passes through verification. Replace the question_branch conditional edges block:

```python
    builder.add_conditional_edges(
        "question_branch",
        _route_question_branch,
        {
            "strategic": "intent_analysis",
            "tactical": "trajectory_prediction",
            "informational": "record_decision",
            "error": "handle_error",
        },
    )
```

with:

```python
    builder.add_conditional_edges(
        "question_branch",
        _route_question_branch,
        {
            "strategic": "intent_analysis",
            "tactical": "trajectory_prediction",
            "informational": "active_verification",
            "error": "handle_error",
        },
    )
    builder.add_conditional_edges(
        "active_verification",
        _route_after_verification,
        {
            "informational": "record_decision",
            "error": "handle_error",
        },
    )
```

### 2.11 Edit: `src/underwater_tracking/agent/nodes/optimize.py`

Edit A — `_build_problem` derives the reserved set and folds it into `unavailable`. Replace:

```python
    unavailable = {
        uuv_id
        for uuv_id in uuvs
        if status_by_id[uuv_id] in (UUVStatus.FAILED, UUVStatus.RETURNING)
        or uuv_id in disabled
    }
```

with:

```python
    reserved = {
        uuv_id
        for directive in snapshot.applied_directives
        if directive.directive_type == "assignment"
        for uuv_id in directive.assignment_uuv_ids
    }
    unavailable = {
        uuv_id
        for uuv_id in uuvs
        if status_by_id[uuv_id] in (UUVStatus.FAILED, UUVStatus.RETURNING)
        or uuv_id in disabled
        or uuv_id in reserved
    }
```

Edit B — the `AllocationInput` construction carries the reserved ids. Replace:

```python
        uuv_available=uuv_available,
        prior_members=prior_members,
```

with:

```python
        uuv_available=uuv_available,
        reserved_uuv_ids=frozenset(reserved),
        prior_members=prior_members,
```

Edit C — `_previous_plan_infeasible` treats human-reserved members as unusable. Replace:

```python
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in directive.disabled_uuv_ids
    }
    return any(
        status_by_id.get(member) in (UUVStatus.FAILED, UUVStatus.RETURNING)
        or member in disabled
        for members in active.member_ids_by_target.values()
        for member in members
    )
```

with:

```python
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in directive.disabled_uuv_ids
    }
    reserved = {
        uuv_id
        for directive in snapshot.applied_directives
        if directive.directive_type == "assignment"
        for uuv_id in directive.assignment_uuv_ids
    }
    return any(
        status_by_id.get(member) in (UUVStatus.FAILED, UUVStatus.RETURNING)
        or member in disabled
        or member in reserved
        for members in active.member_ids_by_target.values()
        for member in members
    )
```

### 2.12 Edit: `src/underwater_tracking/cli.py`

Edit A — imports. Insert `from underwater_tracking.domain.agent_models import VerificationCommand` immediately before the existing line `from underwater_tracking.domain.models import SituationSnapshot`.

Edit B — `on_situation` syncs the registry and applies the protocol commands. Replace the whole `on_situation` method (from `def on_situation(self, situation: SituationSnapshot) -> None:` through its final `self._apply_new_commands()` line) with:

```python
    def on_situation(self, situation: SituationSnapshot) -> None:
        """Engine hook: run one carrier cycle over the latest situation."""
        runtime = self._runtime
        assert runtime is not None
        engine = self._engine
        assert engine is not None
        self.situation = situation
        engine.set_reservations(runtime.reservations())
        if not self._initialization_submitted and self._initialization_ready(situation):
            self._initialization_submitted = True
            runtime.submit_event(
                event_type="initialization",
                entity_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
            )
        try:
            result = runtime.tick()
        except Exception:  # noqa: BLE001 - the group loop must keep running
            self.carrier_error_count += 1
            return
        if result.get("commit_status") == "committed":
            self._apply_new_commands()
        self._apply_verification_commands(result)
```

Edit C — new method after `_apply_new_commands` (after its final `engine.apply_plan_command(command)` line):

```python
    def _apply_verification_commands(self, result: dict[str, Any]) -> None:
        """Apply the deterministic verification protocol commands to the engine.

        Runs after the plan-command gate: the protocol's sensor-mode writes
        win over any plan command from the same cycle.
        """
        engine = self._engine
        assert engine is not None
        for command in result.get("verification_commands") or ():
            assert isinstance(command, VerificationCommand)
            engine.apply_verification_command(command)
```

The engine API (`set_reservations`, `apply_verification_command`) is the T3 contract: `set_reservations(registry)` is duck-typed through `registry.reserved_uuvs()`.

### 2.13 Edit: `tests/integration/test_agent_loop.py`

Edit A — extend the agent_models import. Replace:

```python
from underwater_tracking.domain.agent_models import TrackingPlan
```

with:

```python
from underwater_tracking.domain.agent_models import TrackingPlan, VerificationCommand
```

Edit B — the test harness `AgentLoop.on_situation` syncs the registry and applies protocol commands. Replace the whole `on_situation` method (from `def on_situation(self, situation: SituationSnapshot) -> None:` through the closing `self.commits.append((active, situation))` block) with:

```python
    def on_situation(self, situation: SituationSnapshot) -> None:
        """Engine hook: run one carrier cycle over the latest situation."""
        runtime = self._runtime
        assert runtime is not None
        engine = self._engine
        assert engine is not None
        self.situation = situation
        engine.set_reservations(runtime.reservations())
        if not self._initialization_submitted and self._initialization_ready(situation):
            self._initialization_submitted = True
            runtime.submit_event(
                event_type="initialization",
                entity_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
            )
        plan_before = self._last_plan_id
        try:
            result = runtime.tick()
        except Exception as exc:  # noqa: BLE001 - the group loop must keep running
            self.cycle_errors.append((situation.sim_time_s, repr(exc)))
            return
        self.cycle_results.append(result)
        self._apply_new_commands()
        self._apply_verification_commands(result)
        # A real commit broadcasts a new plan id; the ``commit_status``
        # channel is checkpointed and persists on informational cycles, so
        # only a broadcast change counts as a commit.
        if self._last_plan_id != plan_before:
            active = self.plans.get_active(self.scenario_id)
            if active is not None:
                self.commits.append((active, situation))
```

Edit C — new method after `_apply_new_commands` (after its final `engine.apply_plan_command(command)` line):

```python
    def _apply_verification_commands(self, result: dict[str, Any]) -> None:
        """Apply the deterministic verification protocol commands to the engine."""
        engine = self._engine
        assert engine is not None
        for command in result.get("verification_commands") or ():
            assert isinstance(command, VerificationCommand)
            engine.apply_verification_command(command)
```

## Task T5 Step 3 — verify green

Run the T5 test surface:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/planning/test_reservations.py tests/agent/test_assignment_directives.py tests/agent/test_active_verification.py tests/planning/test_allocation.py -q
```

Expected: PASS, all green.

Then the full offline suite (live-LLM modules self-skip when the API key is unset):

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Expected: PASS. The pre-existing `tests/agent/test_directives.py` and `tests/agent/test_central_graph.py` remain green (the directives changes are additive branches guarded by `directive_type == "assignment"`, and the informational route only gains a deterministic pass-through node).

Exit criteria checklist:

- [ ] The four test files pass; `test_reserved_uuvs_are_never_assigned` proves reserved UUVs appear in no membership set.
- [ ] `test_runtime_apply_assignment_reserves_the_uuvs` proves preview -> apply -> reserved registry state, with the LLM never invoked.
- [ ] `test_submarine_classification_dispatches_and_closes_the_gate` proves ping -> dispatch -> 1200 m gate -> collective return to passive.
- [ ] The full offline suite passes.

## Task T5 Step 4 — commit

```bash
git add src/underwater_tracking tests
git commit -m "feat: human assignment mode and active verification protocol"
```

Exit criteria:

- [ ] `git status` clean; the commit message is exactly `feat: human assignment mode and active verification protocol`.
- [ ] The commit contains the new modules `planning/reservations.py` and `agent/nodes/active_verification.py` and no unintended files.

## Task T6 — human control mode and sonar overlays in the UI

**Prerequisite (binding):** the UI plan's Tasks 1-3 and 5-9 must be complete before this task starts. Concretely: `src/underwater_tracking/domain/ui_models.py` and `src/underwater_tracking/api/frame_builder.py` (with `build_operational_frame(snapshot, plan, ledger_tail, events, metrics)`) exist, and `src/underwater_tracking/ui/src/types/frames.ts` (the authoritative contract mirroring `ui_models`) exists together with `components/layout/CommandShell.tsx` and `components/drawer/BottomDrawer.tsx`. If any of these is missing, STOP and complete the UI plan first, then return to this task. This task extends the UI contract with the R4/R5/R3 state (spec 17.2, 17.3, 17.1) and adds the assignment panel and the sonar/segment overlays.

### T6 Step 1 — Python contract extensions

Write the three test additions below by appending them to `tests/api/test_frame_contracts.py` (reusing its existing `_full_frame` helper), then run-and-fail:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/api/test_frame_contracts.py -q
```

Expected: FAIL — `ValidationError` on the new keywords (`UUVView` rejects `sensor_mode`/`reserved`, `TargetEstimateView` rejects `classification`/`last_ping_s`, `PlanView` rejects `segment_plan`).

```python
def test_uuv_view_carries_sensor_mode_and_reservation_state():
    """R4/R5 (spec 17.2/17.3): the frame renders active-sonar and
    human-assignment state without leaking truth."""
    frame = _full_frame()
    assert frame.uuvs[0].sensor_mode == "passive"
    assert frame.uuvs[0].reserved is False
    active = UUVView(
        uuv_id="UUV-2",
        status="tracking",
        position=Point2D(x=500.0, y=600.0),
        heading_rad=0.2,
        speed_mps=2.0,
        energy_fraction=0.6,
        sensor_mode="active",
        reserved=True,
    )
    assert active.sensor_mode == "active"
    assert active.reserved is True


def test_target_estimate_carries_classification_and_ping_recency():
    estimate = TargetEstimateView(
        target_id="T1",
        mean=Point2D(x=310.0, y=390.0),
        covariance_ellipse=CovarianceEllipse(
            semimajor_m=25.0, semiminor_m=8.0, rotation_rad=0.3
        ),
        intent=IntentView(label="transit", confidence=0.85),
        quality=EstimateQualityView(
            quality_score=0.9,
            estimated_rmse_m=12.5,
            fim_min_eigenvalue=0.4,
            fim_condition=3.0,
        ),
        classification="submarine",
        last_ping_s=15,
    )
    assert estimate.classification == "submarine"
    assert estimate.last_ping_s == 15
    assert _full_frame().target_estimates[0].classification == "unknown"
    assert _full_frame().target_estimates[0].last_ping_s is None


def test_plan_view_carries_the_segmented_relay_plan():
    plan = PlanView(
        plan_id="plan-7",
        version=4,
        status="active",
        segment_plan=("relay:G-T1:0-300", "relay:G-T2:300-600"),
    )
    assert plan.segment_plan == ("relay:G-T1:0-300", "relay:G-T2:300-600")
    assert _full_frame().plans[0].segment_plan == ()
```

Implement — three additive edits to `src/underwater_tracking/domain/ui_models.py`:

Edit A — in `UUVView`, replace:

```python
    current_waypoint: Point2D | None = None
    breadcrumb: tuple[Point2D, ...] = ()
```

with:

```python
    current_waypoint: Point2D | None = None
    breadcrumb: tuple[Point2D, ...] = ()
    sensor_mode: Literal["active", "passive"] = "passive"
    reserved: bool = False
```

Edit B — in `TargetEstimateView`, replace:

```python
    prediction: PredictionCorridorView | None = None
    quality: EstimateQualityView
```

with:

```python
    prediction: PredictionCorridorView | None = None
    quality: EstimateQualityView
    classification: Literal["submarine", "decoy", "unknown"] = "unknown"
    last_ping_s: int | None = None
```

Edit C — in `PlanView`, replace:

```python
    valid_from_s: int = Field(default=0, ge=0)
    valid_until_s: int | None = None
```

with:

```python
    valid_from_s: int = Field(default=0, ge=0)
    valid_until_s: int | None = None
    segment_plan: tuple[str, ...] = ()
```

Then the conditional `segment_plan` source-field check. Run:

```bash
grep -n "segment_plan" src/underwater_tracking/domain/agent_models.py
```

If the two `segment_plan` fields are already present (the T3 segmentation task adds them to `StrategyProposal` and `TrackingPlan`), skip this paragraph. Otherwise append `segment_plan: tuple[str, ...] = ()` as the final field of `StrategyProposal` and of `TrackingPlan` in `src/underwater_tracking/domain/agent_models.py` (after the last field of each class body).

Run-and-pass:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/api/test_frame_contracts.py -q
```

Exit criteria:

- [ ] All three new contract tests pass; the pre-existing five frame-contract tests stay green.
- [ ] The frame schema still contains no truth fields (`test_operational_frame_schema_contains_no_truth_fields` passes).

### T6 Step 2 — frame adapter mapping (additive, anchored on `build_operational_frame`)

**Supersession note (amends T3):** the engine's `active_ping` events must carry their pinger list in the payload: `{"position_xy": (x, y), "uuv_ids": (pinger_id,)}`. If the T3 `_process_pings` emits only `position_xy`, add the `"uuv_ids"` key populated from the verification command that initiated the ping (the command carries `uuv_ids`).

Edit A — in `src/underwater_tracking/api/frame_builder.py`, insert immediately before the `return OperationalFrame(...)` statement of `build_operational_frame`:

```python
    # R4/R5 (spec 17.2/17.3): human-assignment reservation state comes from
    # the applied directives; active-sonar pinger state from the protocol's
    # active_ping events (the engine stamps each ping with its pinger).
    reserved_uuvs = {
        uuv_id
        for directive in snapshot.applied_directives
        if directive.directive_type == "assignment"
        for uuv_id in directive.assignment_uuv_ids
    }
    active_pingers = {
        uuv_id
        for event in events
        if event.event_type == "active_ping"
        for uuv_id in event.payload.get("uuv_ids", ())
    }
    latest_ping_by_target: dict[str, int] = {}
    for event in events:
        if event.event_type == "active_ping" and event.entity_id:
            latest_ping_by_target[event.entity_id] = event.sim_time_s
    classification_by_target: dict[str, str] = {}
    for event in events:
        if event.event_type == "contact_classified" and event.entity_id:
            classification_by_target[event.entity_id] = str(
                event.payload.get("outcome", "unknown")
            )
```

Edit B — add the mapping keywords to the three view constructors (each strict model rejects unknown fields, so the anchor is the existing constructor call by field name):

- in the `UUVView(...)` constructor add `sensor_mode="active" if uuv.uuv_id in active_pingers else "passive",` and `reserved=uuv.uuv_id in reserved_uuvs,`;
- in the `TargetEstimateView(...)` constructor add `classification=classification_by_target.get(target_id, "unknown"),` and `last_ping_s=latest_ping_by_target.get(target_id),`;
- in the `PlanView(...)` constructor add `segment_plan=plan.segment_plan,`.

Exit criteria:

- [ ] `grep -n "sensor_mode\|reserved=\|classification=\|last_ping_s\|segment_plan" src/underwater_tracking/api/frame_builder.py` shows the mapping lines.
- [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/api -q` passes.

### T6 Step 3 — TypeScript contract and the three new components

Pre-flight naming check (the components below assume the ui_models-mirrored names `uuv_id`, `target_estimates`, `plans`, `target_id`, `segment_plan`):

```bash
grep -n "uuv_id\|target_estimates\|segment_plan\|last_ping_s" src/underwater_tracking/ui/src/types/frames.ts
```

If `src/underwater_tracking/ui/src/types/frames.ts` does not exist, STOP (the UI plan Tasks 5-6 are incomplete). If the grep shows different names, adjust the property accesses in the components and tests below to the actual names; the Python contract (Step 1) is authoritative.

Edit A — append the additive fields to `frames.ts`. In the `UUVView` interface add:

```ts
  /** Active-sonar protocol state (spec 17.3): "active" while pinging. */
  sensor_mode: "active" | "passive";
  /** True when the operator reserved this UUV for a target (spec 17.2). */
  reserved: boolean;
```

In the `TargetEstimateView` interface add:

```ts
  /** Contact classification once verified (spec 17.3). */
  classification: "submarine" | "decoy" | "unknown";
  /** Simulation time of the latest active ping, when verified. */
  last_ping_s: number | null;
```

In the `PlanView` interface add:

```ts
  /** Segmented relay plan: one label per segment (spec 17.1, R3). */
  segment_plan: string[];
```

Edit B — create `src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx`:

```tsx
import { useState } from "react";
import type { TargetEstimateView, UUVView } from "../../types/frames";

export interface AssignmentPanelProps {
  targets: TargetEstimateView[];
  uuvs: UUVView[];
  onAssign: (uuvIds: string[], targetId: string) => void;
}

/**
 * Human-in-loop assignment mode (spec 17.2, R4): the expert picks UUVs for
 * a target; the carrier reserves them, so the allocator and the
 * verification protocol both exclude them. Reserved UUVs are hidden from
 * the picker (already committed to another target).
 */
export default function AssignmentPanel({
  targets,
  uuvs,
  onAssign,
}: AssignmentPanelProps) {
  const [targetId, setTargetId] = useState<string>(targets[0]?.target_id ?? "");
  const [selected, setSelected] = useState<string[]>([]);
  const available = uuvs.filter((uuv) => !uuv.reserved);
  const toggle = (uuvId: string) =>
    setSelected((prev) =>
      prev.includes(uuvId) ? prev.filter((id) => id !== uuvId) : [...prev, uuvId],
    );
  return (
    <section className="assignment-panel" aria-label="人为指派模式">
      <div className="section-heading">
        <span>人为指派</span>
        <small>已指派 {uuvs.filter((uuv) => uuv.reserved).length} 艇</small>
      </div>
      <label className="field">
        <span>跟踪目标</span>
        <select
          value={targetId}
          onChange={(event) => setTargetId(event.target.value)}
          aria-label="选择跟踪目标"
        >
          {targets.map((target) => (
            <option key={target.target_id} value={target.target_id}>
              {target.target_id}
            </option>
          ))}
        </select>
      </label>
      <div className="uuv-list compact" role="group" aria-label="可选 UUV">
        {available.map((uuv) => (
          <label key={uuv.uuv_id} className="uuv-check">
            <input
              type="checkbox"
              checked={selected.includes(uuv.uuv_id)}
              onChange={() => toggle(uuv.uuv_id)}
            />
            <span>{uuv.uuv_id}</span>
            <small>{uuv.sensor_mode === "active" ? "主动声纳" : "被动声纳"}</small>
          </label>
        ))}
        {available.length === 0 && <p>无可用 UUV</p>}
      </div>
      <button
        className="primary-btn"
        disabled={selected.length === 0 || !targetId}
        onClick={() => onAssign([...selected].sort(), targetId)}
      >
        指派跟踪
      </button>
    </section>
  );
}
```

Edit C — create `src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TargetEstimateView, UUVView } from "../../types/frames";
import AssignmentPanel from "./AssignmentPanel";

const targets: TargetEstimateView[] = [
  {
    target_id: "T1",
    group_id: "G1",
    mean: { x: 310, y: 390 },
    intent: "transit",
    quality: 0.9,
    prediction_corridor: null,
    classification: "unknown",
    last_ping_s: null,
  },
];

const uuvs: UUVView[] = [
  {
    uuv_id: "UUV-1",
    status: "tracking",
    position: { x: 100, y: 200 },
    heading_rad: 0.5,
    speed_mps: 2.0,
    energy_fraction: 0.8,
    group_id: "G1",
    sensor_mode: "passive",
    reserved: false,
  },
  {
    uuv_id: "UUV-2",
    status: "tracking",
    position: { x: 300, y: 400 },
    heading_rad: 0.2,
    speed_mps: 2.0,
    energy_fraction: 0.6,
    group_id: null,
    sensor_mode: "active",
    reserved: true,
  },
];

describe("AssignmentPanel", () => {
  it("lists only non-reserved UUVs and reports the picked assignment", () => {
    const onAssign = vi.fn();
    render(<AssignmentPanel targets={targets} uuvs={uuvs} onAssign={onAssign} />);
    expect(screen.getByRole("checkbox", { name: /UUV-1/ })).toBeInTheDocument();
    // UUV-2 is operator-reserved: excluded from the LLM-allocatable set and
    // therefore hidden from the assignment picker.
    expect(screen.queryByRole("checkbox", { name: /UUV-2/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /UUV-1/ }));
    fireEvent.click(screen.getByRole("button", { name: "指派跟踪" }));
    expect(onAssign).toHaveBeenCalledWith(["UUV-1"], "T1");
  });

  it("disables the assign button until a UUV is selected", () => {
    const onAssign = vi.fn();
    render(<AssignmentPanel targets={targets} uuvs={uuvs} onAssign={onAssign} />);
    const button = screen.getByRole("button", { name: "指派跟踪" });
    expect(button).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: /UUV-1/ }));
    expect(button).toBeEnabled();
  });

  it("counts reserved UUVs as 已指派", () => {
    render(<AssignmentPanel targets={targets} uuvs={uuvs} onAssign={vi.fn()} />);
    expect(screen.getByText("已指派 1 艇")).toBeInTheDocument();
  });
});
```

Edit D — create `src/underwater_tracking/ui/src/components/map/SonarBadges.tsx`:

```tsx
import type { UUVView } from "../../types/frames";

export interface SonarBadgesProps {
  uuvs: UUVView[];
}

/**
 * Compact active-sonar status strip over the map (spec 17.3, R5): every
 * pinging UUV and every operator-reserved UUV is badged; an all-passive
 * fleet shows a single passive marker.
 */
export default function SonarBadges({ uuvs }: SonarBadgesProps) {
  const active = uuvs.filter((uuv) => uuv.sensor_mode === "active");
  const reserved = uuvs.filter((uuv) => uuv.reserved);
  return (
    <div className="sonar-badges" aria-label="主动声纳状态">
      {active.map((uuv) => (
        <span key={uuv.uuv_id} className="badge badge-active">
          {uuv.uuv_id} 主动
        </span>
      ))}
      {reserved.map((uuv) => (
        <span key={uuv.uuv_id} className="badge badge-reserved">
          {uuv.uuv_id} 指派
        </span>
      ))}
      {active.length === 0 && reserved.length === 0 && (
        <span className="badge badge-passive">全部被动</span>
      )}
    </div>
  );
}
```

Edit E — create `src/underwater_tracking/ui/src/components/map/SonarBadges.test.tsx` (reusing the `uuvs` fixture above, same shape):

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { UUVView } from "../../types/frames";
import SonarBadges from "./SonarBadges";

const uuvs: UUVView[] = [
  {
    uuv_id: "UUV-1",
    status: "tracking",
    position: { x: 100, y: 200 },
    heading_rad: 0.5,
    speed_mps: 2.0,
    energy_fraction: 0.8,
    group_id: "G1",
    sensor_mode: "passive",
    reserved: false,
  },
  {
    uuv_id: "UUV-2",
    status: "tracking",
    position: { x: 300, y: 400 },
    heading_rad: 0.2,
    speed_mps: 2.0,
    energy_fraction: 0.6,
    group_id: null,
    sensor_mode: "active",
    reserved: true,
  },
];

describe("SonarBadges", () => {
  it("badges pinging and reserved UUVs", () => {
    render(<SonarBadges uuvs={uuvs} />);
    expect(screen.getByText("UUV-2 主动")).toBeInTheDocument();
    expect(screen.getByText("UUV-2 指派")).toBeInTheDocument();
    expect(screen.queryByText("全部被动")).not.toBeInTheDocument();
  });

  it("shows a single passive marker for a fully passive fleet", () => {
    const passive = uuvs.map((uuv) => ({
      ...uuv,
      sensor_mode: "passive" as const,
      reserved: false,
    }));
    render(<SonarBadges uuvs={passive} />);
    expect(screen.getByText("全部被动")).toBeInTheDocument();
  });
});
```

Edit F — create `src/underwater_tracking/ui/src/components/map/SegmentOverlay.tsx`:

```tsx
import type { PlanView } from "../../types/frames";

export interface SegmentOverlayProps {
  plans: PlanView[];
}

/**
 * Segmented relay plan timeline (spec 17.1, R3): the LLM segments the
 * predicted track; groups relay the target segment by segment. Renders the
 * newest plan's segment list as an ordered relay timeline.
 */
export default function SegmentOverlay({ plans }: SegmentOverlayProps) {
  const segments = plans[0]?.segment_plan ?? [];
  return (
    <div className="segment-overlay" aria-label="分段接力方案">
      <div className="section-heading">
        <span>分段接力</span>
        <small>{segments.length} 段</small>
      </div>
      {segments.length === 0 ? (
        <p>尚未生成分段接力方案</p>
      ) : (
        <ol className="segment-list">
          {segments.map((segment, index) => (
            <li key={`${segment}-${index}`}>{segment}</li>
          ))}
        </ol>
      )}
    </div>
  );
}
```

Edit G — create `src/underwater_tracking/ui/src/components/map/SegmentOverlay.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PlanView } from "../../types/frames";
import SegmentOverlay from "./SegmentOverlay";

const plan: PlanView = {
  plan_id: "plan-7",
  version: 4,
  status: "active",
  segment_plan: ["relay:G-T1:0-300", "relay:G-T2:300-600"],
};

describe("SegmentOverlay", () => {
  it("renders the segmented relay plan in order", () => {
    render(<SegmentOverlay plans={[plan]} />);
    expect(screen.getByText("2 段")).toBeInTheDocument();
    const items = screen.getAllByRole("listitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "relay:G-T1:0-300",
      "relay:G-T2:300-600",
    ]);
  });

  it("shows an empty state before a segmented plan exists", () => {
    render(<SegmentOverlay plans={[{ ...plan, segment_plan: [] }]} />);
    expect(screen.getByText("尚未生成分段接力方案")).toBeInTheDocument();
  });
});
```

Run-and-fail (each of the four test files fails on import before its component exists):

```bash
npm --prefix src/underwater_tracking/ui test -- AssignmentPanel.test.tsx
npm --prefix src/underwater_tracking/ui test -- SonarBadges.test.tsx
npm --prefix src/underwater_tracking/ui test -- SegmentOverlay.test.tsx
```

Then run-and-pass:

```bash
npm --prefix src/underwater_tracking/ui test -- AssignmentPanel.test.tsx
npm --prefix src/underwater_tracking/ui test -- SonarBadges.test.tsx
npm --prefix src/underwater_tracking/ui test -- SegmentOverlay.test.tsx
```

Exit criteria:

- [ ] All three component test files pass; the assignment panel excludes reserved UUVs and reports `(["UUV-1"], "T1")`.
- [ ] `npm --prefix src/underwater_tracking/ui run build` exits 0.

### T6 Step 4 — wiring

Edit A — `src/underwater_tracking/ui/src/components/layout/CommandShell.tsx` (created by UI Task 5): add an "指派" mode button next to the existing mode switch. When the assignment mode is active, render above the drawer:

```tsx
      {assignmentMode && (
        <AssignmentPanel
          targets={frame?.target_estimates ?? []}
          uuvs={frame?.uuvs ?? []}
          onAssign={(uuvIds, targetId) => {
            void assignTargets(uuvIds, targetId);
            setAssignmentMode(false);
          }}
        />
      )}
```

with the imports `import AssignmentPanel from "../assistant/AssignmentPanel";` and `import { assignTargets } from "../../services/assistantApi";` (the service is created by UI Task 9; its `assignTargets` POSTs the assignment to the directive queue, which the carrier previews and applies through `preview_assignment`/`apply_directive`). If `CommandShell.tsx` does not exist, STOP (UI Tasks 5-9 incomplete).

Edit B — `src/underwater_tracking/ui/src/components/drawer/BottomDrawer.tsx` (created by UI Task 8): in the plan tab, after the plan header block, render `<SegmentOverlay plans={frame.plans} />` (import from `../../components/map/SegmentOverlay`). If the anchor is absent, STOP.

Edit C — `src/underwater_tracking/ui/src/App.tsx` (rewritten by UI Task 7 to consume the authoritative frame): inside the map container, after the `<TacticalMap ... />` element, render `<SonarBadges uuvs={frame?.uuvs ?? []} />` (import from `./components/map/SonarBadges`). If the anchor is absent, STOP.

### T6 Step 5 — full verification and commit

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
npm --prefix src/underwater_tracking/ui test
npm --prefix src/underwater_tracking/ui run build
```

Expected: all green. Commit:

```bash
git add src/underwater_tracking
git commit -m "feat: human control mode and sonar overlays in the ui"
```

Exit criteria:

- [ ] The full Python suite and the UI suite pass; the production build exits 0.
- [ ] The commit message is exactly `feat: human control mode and sonar overlays in the ui`.

## Plan exit criteria (whole plan)

- [ ] Tasks T1-T6 are committed in order with the exact commit messages; `git status` clean.
- [ ] The full offline suite is green: `PYTHONPATH=. .venv/bin/python -m pytest tests -q`.
- [ ] The live LongCat tests are green with the API key inserted by the executor (they self-skip without it): `UNDERWATER_TRACKING_API_KEY=<key supplied in the task dispatch> PYTHONPATH=. .venv/bin/python -m pytest tests/integration/test_llm_real_api.py tests/agent/test_verify_graph.py tests/agent/test_directives.py tests/integration/test_agent_loop.py -q`. The key never appears in this document, is never printed, logged, or asserted by value; it enters git history only through the shipped `configs/llm.yaml` per the user's explicit requirement (risk accepted and noted in spec 22).
- [ ] `ruff check src tests` and `mypy` (strict on `src`) are clean.
- [ ] R1-R5 are all implemented: config explicitness (T1), physics realism (T2), trajectory segmentation and multi-group relay (T3), human assignment mode with reserved-UUV exclusion and bearing-pursuit execution (T4/T5), and the active-sonar protocol state machine (T5) with UI overlays (T6).

## Self-review

- R1 config explicitness: YAML carries the API key and every LLM knob; the environment variable overrides when set (`api_key_env` read at call time in `llm.py`); the key enters git history by explicit user requirement.
- R2 physics: UUV max 6.0 -> 4.0 m/s, submarine cruise 8.0 / sprint 14.0 m/s, scaled turn rates (T2).
- R3 segmentation: `StrategyProposal.segment_plan` with a predicted-track summary in the strategy payload, multi-group relay, deterministic uniform time-split fallback (T3).
- R4 assignment: `ReservationRegistry`; reserved UUVs excluded from `AllocationInput` (`_available` gate, `_build_problem`, `_previous_plan_infeasible`), from the verification pinger pool, and from the LLM-allocatable set; the typed `assign_target_uuvs` shortcut and `preview_assignment` never invoke the LLM; the engine runs the bearing-pursuit controller for assigned UUVs (T4).
- R5 protocol: idle -> verifying (nearest-first, one available non-reserved UUV) -> classified (submarine dispatch via the existing allocation channel; decoy drop + return to passive) -> in_position (1200 m gate) -> all passive; `PlanCommand` gains `sensor_mode`; verification commands apply after plan commands so the protocol's sensor-mode writes win.
- Placeholder scan: no TBD/TODO/`<...>` placeholders; every test file and edit is exact text.
- The api_key string does not appear anywhere in this document.

## Return envelope

Plan path: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/.claude/worktrees/underwater-tracking-sdd/docs/superpowers/plans/2026-08-15-underwater-tracking-tactical-realism-plan.md`

One-line summary: T1-T6 implement R1-R5 (config explicitness, physics realism, LLM trajectory segmentation, human assignment mode with reserved-UUV exclusion, and the deterministic active-sonar verification protocol) as six self-contained tasks with verbatim tests and edits, plus the R4/R5/R3 UI contract and overlays.

Interface facts NOT verifiable from the codebase (the controller fills these during pre-flight):

1. `planning/validator.py` internals — imported by the plan but never read; its exported surface is not asserted by any T1-T6 test.
2. `prediction/bspline.py` internals — the T3 trajectory-segmentation edit targets it, but its current text was never read; the executor must apply T3's edits by anchor.
3. `tests/agent/test_central_graph.py` — only grep-scanned (route/LLM-call assertions); not read in full.
4. `tracking/imm.py` `DEFAULT_PROCESS_NOISE` — value not verified; T2's physics constants do not depend on it.
5. The LongCat API key value — deliberately absent here; the task dispatch must supply it verbatim to the executor.
6. `tests/api/test_frame_pipeline.py` and `tests/api/test_truth_isolation.py` — confirmed to NOT exist yet (UI plan Tasks 3 and 10 are pending).
7. `groups/state.py` lines 120+ — never read; the T3/T4 engine edits reference the group manager's public methods only.
8. `persistence/ledger.py` `save_directive`/`list_directives` signatures — inferred from call sites only.
9. `tests/integration/test_agent_loop.py` — verified through line 249 (the `on_situation` edit anchor at 177-203 is exact); the tail is unread.
10. T6 anchors: `frame_builder.py`'s exact constructor calls and `frames.ts`'s exact interface shapes are created by the UI plan's Tasks 3/6/8, which are not yet executed; T6 includes pre-flight existence checks and name greps, and its frame adapter keywords are anchored on the strict ui_models field names.
