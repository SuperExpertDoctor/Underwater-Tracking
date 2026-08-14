# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan:** Underwater Tracking Agent Master Roadmap

**Goal:** Deliver the approved, reproducible multi-UUV bearing-only tracking assistant through four independently testable implementation plans.

**Architecture:** Build the deterministic simulation and tracking loop first, add the persistent LangGraph intelligence layer second, migrate and restyle the reference visualization third, and add baselines plus Monte Carlo acceptance last. Each stage leaves a runnable artifact and a green test suite; later stages depend only on public schemas from earlier stages.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy, SciPy, LangGraph 1.x, FastAPI, SQLite, React 18, Vite, Canvas, Pytest, Hypothesis, Playwright.

---

## Approved source specification

Implement only requirements in [`docs/superpowers/specs/2026-08-14-underwater-tracking-agent-design.md`](../specs/2026-08-14-underwater-tracking-agent-design.md). If implementation pressure reveals a design change, update and re-approve the specification before changing behavior.

## Plan suite and dependency order

1. [`2026-08-14-underwater-tracking-foundation-plan.md`](2026-08-14-underwater-tracking-foundation-plan.md)
   - Produces a deterministic, headless, 2-D simulation with bearing observations, IMM-UIF beliefs, B-spline prediction, quality signals, elastic group allocation, FIM waypoint control, and JSONL output.
   - Exit command: `python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42`.

2. [`2026-08-14-underwater-tracking-agent-plan.md`](2026-08-14-underwater-tracking-agent-plan.md)
   - Adds the carrier LangGraph, provider-neutral LLM port, structured Verify loop, event routing, history compression, DecisionLedger, non-blocking directives, questions, and deterministic degradation.
   - Exit command: `python -m underwater_tracking.cli agent-run --config configs/scenario/default.yaml --steps 360 --seed 42 --llm mock`.

3. [`2026-08-14-underwater-tracking-ui-plan.md`](2026-08-14-underwater-tracking-ui-plan.md)
   - Migrates the front/back visualization architecture from `E:\项目\创新院\Maritime-Surveillance`, replaces the maritime business model, and applies the deep-ocean command style.
   - Exit commands: `python -m underwater_tracking.cli serve --config configs/scenario/default.yaml --seed 42` and `npm --prefix src/underwater_tracking/ui run test:e2e`.

4. [`2026-08-14-underwater-tracking-evaluation-plan.md`](2026-08-14-underwater-tracking-evaluation-plan.md)
   - Adds B0-2, B0-3, B1, B2 and Full policies, scenario matrices, paired Monte Carlo execution, metrics, acceptance freezing, reports, and truth-isolation gates.
   - Exit command: `python -m underwater_tracking.evaluation.cli accept --experiment-dir outputs/experiments/formal`.

Do not execute the plans in parallel: the Agent plan consumes Foundation schemas, UI consumes Agent runtime APIs, and Evaluation consumes all previous public interfaces.

## Specification coverage

| Approved specification area | Owning plan and tasks |
|---|---|
| Overall architecture, data contracts, multi-rate loop | Foundation Tasks 1–2 and 10–11; Agent Tasks 1 and 8 |
| Bearing-only initialization and IMM-UIF | Foundation Tasks 4–5 |
| B-spline prediction and trajectory features | Foundation Task 6; Agent Task 5 |
| FIM, group quality, economic allocation, robust waypoints | Foundation Tasks 7–10 |
| LangGraph nodes, event routing, Verify/Retry, persistence | Agent Tasks 2–8 |
| History compression and context control | Agent Task 9 |
| Expert annotation, questioning, counterfactual explanation | Agent Tasks 10–11; UI Task 9 |
| DecisionLedger, failure degradation, recovery | Agent Tasks 2, 8, and 12 |
| Reference-project reuse, command UI, API, replay | UI Tasks 1–10 |
| Unit, property, graph, integration, and end-to-end tests | Foundation Task 12; Agent Task 12; UI Task 10 |
| Baselines, ablations, paired statistics, metrics | Evaluation Tasks 2–7 |
| Acceptance thresholds, reproducibility, truth isolation | Evaluation Tasks 1, 8–10 |

## Locked repository structure

```text
configs/
  acceptance.yaml
  agent.yaml
  llm.yaml
  tracking.yaml
  scenario/
    default.yaml
    small_6uuv_1target.yaml
    pressure_20uuv_6target.yaml
  experiments/pilot.yaml
  experiments/formal.yaml
docs/superpowers/
  specs/
  plans/
src/
  underwater_tracking/
    agent/
      graphs/
      nodes/
    config/
    domain/
    evaluation/
    groups/
    persistence/
    planning/
    prediction/
    simulation/
    tracking/
    ui/
    cli.py
tests/
  api/
  agent/
  domain/
  evaluation/
  groups/
  integration/
  planning/
  prediction/
  simulation/
  tracking/
  ui/
  e2e/
```

## Cross-plan interface rules

- `domain` contains Pydantic schemas only and imports no runtime services.
- `simulation` can publish observations and truth to separate ports; it cannot import `agent` or `vis`.
- `tracking`, `prediction`, and `planning` are deterministic for a fixed input and seed.
- `groups` can import deterministic algorithms but never the LLM client.
- `agent` consumes `SituationSnapshot` and returns validated strategy/plan commands; it cannot import `EvaluationStore`.
- `api/frame_builder.py` adapts public runtime state to UI frames; the operational frame never includes truth.
- `evaluation` may read both operational outputs and hidden truth but cannot feed truth back into the runtime.

## Integration checkpoints

### Checkpoint A: deterministic headless loop

- [ ] Foundation unit, property, and integration tests pass.
- [ ] A fixed seed produces byte-stable normalized observations and deterministic plan assignments.
- [ ] Default 12-UUV scenario runs without an LLM or web server.

### Checkpoint B: intelligent carrier loop

- [ ] Mock LLM drives initialization, strategic replanning, Verify repair, directives, and evidence-backed questions.
- [ ] Malformed model output cannot reach plan commit.
- [ ] Provider failure leaves the high-frequency group loop running.

### Checkpoint C: operator console

- [ ] Operational map, sidebar, bottom drawer, replay, directives, questions, and evaluation truth toggle work.
- [ ] The new project contains its own migrated UI files and has no runtime dependency on `Maritime-Surveillance`.
- [ ] Playwright covers the approved human-in-the-loop flows.

### Checkpoint D: formal evaluation

- [ ] Pilot freezes only noise-, unit-, and sensor-dependent RMSE, FIM, and quality thresholds into `configs/acceptance.yaml`; economic and safety gates remain unchanged.
- [ ] Formal experiments use at least 30 paired seeds per scenario.
- [ ] Acceptance report evaluates every hard gate from the specification.

## Global verification commands

Run after every plan and before declaring the entire project complete:

```powershell
python -m pytest -q
python -m ruff check src tests
python -m mypy src/underwater_tracking
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
python -m underwater_tracking.evaluation.cli batch --experiment configs/experiments/pilot.yaml --output-root outputs/experiments/pilot
git status --short
```

Expected: all commands exit `0`; the final `git status --short` prints nothing.
