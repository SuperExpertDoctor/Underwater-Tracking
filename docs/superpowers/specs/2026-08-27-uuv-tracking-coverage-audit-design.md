# Multi-UUV Tracking and Coverage Audit Design

**Date:** 2026-08-27
**Status:** Approved in conversation; implementation and runtime verification have not started
**Workspace:** `D:\Air\反Q\Underwater-Tracking`
**Baseline:** `origin/master` at `63b13f60f7de639bed4751260c83236c67e9e54c`
**Working branch:** `review/uuv-tracking-coverage-20260827`

## 1. Objective

Audit the repository-native multi-UUV behavior in two areas and, only where evidence confirms a defect, make the smallest test-backed correction:

1. cooperative tracking of a moving underwater target, including whether the implemented estimator/planner/controller chain produces physically applied UUV control and a measurable tracking effect; and
2. cooperative coverage search, with particular attention to the implemented serpentine scan, including whether planned lanes are valid, assigned coherently, executed by UUVs, and sufficient for the coverage claim made by the repository.

The final handoff must include reproducible evidence, a Chinese audit report, machine-readable metrics, and headless visual evidence. A short video is preferred; representative screenshots are required as a fallback. The branch will be committed and pushed without merging into `master`.

This is a focused, same-day audit. A fixed-seed short simulation can demonstrate one reproducible control path and expose defects, but it cannot support a statistical robustness or performance-superiority claim.

## 2. Authority, isolation, and non-goals

All persistent work is confined to this repository. In particular:

- do not modify `D:\Air\反Q\tracking\multi_AUV_pursuit_ros_ws`;
- do not modify `D:\Air\反Q\tracking\stft_bot_ros_ws`;
- do not change system, Conda, Python, Node, ROS, shell, PATH, service, port, firewall, or shared-directory configuration;
- do not control real hardware or start a network-dependent LLM workflow;
- do not install or build the full web UI, and do not add Node as a prerequisite;
- do not run long-duration or multi-seed campaign experiments in this acceptance cycle;
- do not merge into `master`; and
- do not delete files or directories.

The repository's native UUV count and scenario scale are the acceptance scale. The audit will not force the scenario to three UUVs merely to match another project.

## 3. Evidence discipline

Every conclusion in the final report will be labelled using one of these evidence levels:

- **Runtime evidence:** current command output, test result, metric, log, or rendered trajectory produced from this branch.
- **Source evidence:** behavior or structure directly present in current code, configuration, or interfaces.
- **Document statement:** a claim made by repository documentation, not independently proven by its presence.
- **Working inference:** a reasoned interpretation that still requires a test or run.

Files, comments, old reports, and old media do not prove current runtime behavior. Existing `test1.webm`, `test1-preview.png`, historical pass counts, and prior audit narratives may provide context, but they will not be reused as current acceptance evidence.

### 3.1 Verified baseline facts

The following facts were established before this design was written:

- the repository was cloned from the requested GitHub origin into the isolated workspace;
- the working branch was created from the current `origin/master` baseline shown above;
- the worktree was clean immediately after branch creation;
- the project declares Python `>=3.11,<3.13`;
- the default `python` on this machine is Python 3.9.13 and is therefore not a valid project runtime;
- an existing Python 3.12.13 interpreter is available, but its existing environment will not be modified;
- Node and FFmpeg are not available on PATH;
- VS Code Git Graph is installed as `mhutchie.git-graph@1.30.0`; and
- Git is available, while GitHub CLI is not.

### 3.2 Current source facts relevant to the audit

Current code and configuration show that:

- `configs/scenario/uuv_only_single_target.yaml` declares a deterministic UUV-only scenario with seed 42, 12 UUVs, one target, a 5 s physics step, and a 30 s observation interval;
- bearing-only estimation, IMM/UIF tracking, group planning, mission-state updates, UUV waypoint commands, and evaluation hooks exist in the source tree;
- `serpentine_coverage_waypoints` and `serpentine_coverage_waypoints_by_uuv` generate scan-line routes inside a polygon;
- the mission optimizer assigns generated routes to `RegionMissionState.scan_waypoints_by_uuv`;
- the simulation engine consumes those routes during active scan and fallback behavior, writes waypoints to UUV entities, and preserves the unconsumed suffix of a route;
- once a usable belief exists, a rolling FIM-oriented group plan may supersede the static serpentine route; and
- current unit tests cover basic rectangular serpentine geometry, distinct per-UUV lanes, deployment-side orientation, vertex-boundary handling, and several IMM/UIF numerical properties.

These source facts prove that the path exists in code. They do not yet prove that the full path executes successfully, that the vehicles obey it, or that the resulting tracking and coverage effects are acceptable.

### 3.3 Applicable design history

The initial 2026-08-14 agent design described large-area search and coverage as a non-goal. The later 2026-08-19 and 2026-08-21 UUV-only designs explicitly introduced multi-UUV serpentine search before target entry. For this UUV-only audit, the later and more specific documents govern the intended coverage behavior.

Historical plans that propose evaluation packages, seed counts, thresholds, or future interfaces are design evidence only unless the current code and runtime independently confirm them. No proposed numerical threshold becomes a hard acceptance gate solely because it appears in an older plan.

## 4. Audit questions

### 4.1 Cooperative tracking

The audit must answer:

1. Is the configured scenario actually connected to the bearing observation, estimator, mission planner, UUV waypoint, and physics-update chain?
2. Are truth data and estimated/public data separated so that target truth cannot leak into control decisions?
3. Do the IMM/UIF update and cooperative planner remain numerically valid under the exercised fixed-seed trajectory?
4. Do two or more UUVs receive distinct, bounded commands that are physically applied rather than merely logged or stored?
5. Do actual UUV positions respond to those commands within configured speed, turn, boundary, and separation constraints?
6. Does the target-estimation error exhibit a measurable, reportable result over the short run, and what limitations prevent stronger claims?
7. Are apparent failures estimator failures, geometry/observability failures, planner failures, controller failures, or execution failures?

### 4.2 Cooperative coverage

The audit must answer:

1. Does serpentine generation keep segments inside the region and alternate their direction consistently?
2. Does multi-UUV assignment avoid duplicate lanes, provide deterministic allocation, and orient routes sensibly from deployment?
3. Are route waypoints retained and consumed as the UUV moves, rather than reset on every observation cycle?
4. When a rolling tracking plan supersedes coverage, is the transition explicit and consistent with mission state?
5. Do actual UUV trajectories visit the planned lanes while respecting map, speed, turn, and separation constraints?
6. Is there a declared sensor footprint or swath width from which area coverage can be computed honestly?
7. If no footprint exists, can only lane visitation and route completion be claimed, rather than percent area coverage?
8. Does the current choice of one generated lane per participating UUV leave meaningful gaps, and is that a bug under the repository's stated model or merely an undocumented assumption?

The last question is deliberately open until geometry, configuration, and runtime evidence are available. The implementation will not invent a sonar swath width to make the result appear complete.

## 5. End-to-end chains to inspect

### 5.1 Tracking chain

```text
scenario/configuration
  -> deterministic target and UUV state propagation
  -> bearing observation generation and gating
  -> IMM/UIF belief update and publication
  -> cooperative group/FIM or fallback planning
  -> per-UUV waypoint command
  -> UUV kinematic execution
  -> truth-isolated evaluation metrics and rendering
```

The audit will trace concrete configuration names, time units, coordinates, angle wrapping, sensor ranges, update intervals, random seed, group membership, planner bounds, minimum separation, command persistence, and the physical state update.

### 5.2 Coverage chain

```text
region polygon and deployment state
  -> scan-line intersections
  -> alternating serpentine lanes
  -> deterministic per-UUV assignment
  -> RegionMissionState route storage
  -> active-scan/fallback selection
  -> persistent waypoint execution
  -> lane-visitation and safety evaluation
  -> trajectory rendering
```

The audit will distinguish three different objects that must not be conflated:

- the ideal geometric scan pattern;
- the route actually assigned to each UUV; and
- the trajectory physically executed by each UUV.

## 6. Verification strategy

Verification proceeds from low cost to higher cost and stops if a lower layer exposes a blocking defect.

### 6.1 Static and configuration audit

Inspect the complete call chains above, including relevant configuration, models, mission-state transitions, observation logic, estimators, planners, waypoint handling, physics, evaluation outputs, and existing tests.

Static acceptance includes:

- syntax and import consistency;
- YAML/configuration compatibility;
- deterministic seed propagation;
- coordinate, unit, and time-step consistency;
- angle normalization and covariance validity paths;
- map, kinematic, separation, and sensor constraints;
- truth/public-data boundaries; and
- output paths that remain repository-local.

### 6.2 Existing tests

Run the smallest relevant test groups first, followed by the repository's current test suite if dependencies and same-day runtime permit. Historical test counts are not acceptance evidence. The exact current command, Python version, exit status, duration, and failures will be recorded.

### 6.3 Test-driven fixes

For each confirmed defect:

1. reduce it to a deterministic failing test or invariant check;
2. confirm the failure on the unmodified implementation;
3. apply the smallest focused fix;
4. rerun the new test and neighboring tests; and
5. rerun the affected end-to-end evidence path.

No speculative refactor is authorized. If behavior is reasonable but under-documented, the report should explain it rather than change it.

### 6.4 Fixed-seed short simulation

Use a shortened, deterministic form of the repository-native UUV-only scenario. Only duration/output settings needed to make a same-day headless run practical may be overridden; algorithm and safety parameters must remain the repository's values unless a confirmed bug requires a documented fix.

At minimum, capture:

- initial deployment/search geometry;
- UUV route assignments;
- actual UUV and target trajectories;
- transition from search to tracking if it occurs in the shortened run;
- estimate and uncertainty over time;
- command-to-motion correspondence;
- safety and kinematic invariant results; and
- deterministic reproduction of key outputs when rerun with the same seed.

If target acquisition does not occur within a defensible short run, do not manipulate truth or silently move the target. Report that limitation and use separately identified deterministic component scenarios for coverage execution and tracking control, while preserving the distinction from the native end-to-end result.

## 7. Metrics and acceptance interpretation

Metrics will use repository definitions where implemented. New metric code may observe truth only through a separate evaluation channel and must never feed truth into control.

### 7.1 Tracking metrics

Where the current interfaces support them, record:

- planar position error over time, including RMSE, median, p95, and maximum;
- track availability under an existing repository definition, if one exists;
- valid observation/update counts and rejected measurements;
- covariance validity and uncertainty trend;
- group membership and distinct-command evidence;
- minimum pairwise UUV separation;
- map-boundary, speed, turn-rate, and step-limit violations; and
- deterministic equality or bounded numerical equivalence for the repeated fixed-seed run.

Position truth is evaluation-only. A falling error curve is useful evidence but is not, by itself, proof of general tracking robustness.

### 7.2 Coverage metrics

Where the current model supports them, record:

- geometric validity of planned scan segments;
- number of planned, assigned, started, and completed lanes;
- planned-lane visitation by actual trajectories;
- repeated or duplicate lane assignments;
- per-UUV path length and load balance;
- route progress/completion;
- minimum pairwise separation;
- map-boundary, speed, turn-rate, and step-limit violations; and
- area coverage and overlap only if a source-backed sensor footprint exists.

If the model has no explicit footprint, the final report must say that percent area coverage is **not available** and limit the claim to waypoint/lane visitation. Missing model information is an evidence gap, not a zero and not permission to fabricate a parameter.

### 7.3 Hard gates

Hard pass/fail gates may come only from:

- current configured physical or safety limits;
- mathematical invariants such as finite values, positive-semidefinite covariance within numerical tolerance, valid polygon intersections, and deterministic assignment; or
- an existing, currently implemented acceptance definition whose call path is verified.

Any descriptive statistic without a source-backed threshold will be reported as a measurement, not converted into an invented pass/fail criterion.

## 8. Headless visual evidence

Visuals must be derived from the same recorded trajectories used for metrics. The renderer must not rerun a different stochastic scenario or depict ideal paths as if they were executed paths.

The durable evidence directory is:

`docs/verification/2026-08-27-uuv-tracking-coverage/`

Planned committed artifacts are:

- `README.md` — Chinese audit report, commands, environment, conclusions, limitations, and reproduction notes;
- `metrics.json` — machine-readable scenario metadata, metrics, invariants, and status;
- `tracking-control.mp4` or another browser-viewable video format available without a system dependency;
- `coverage-search.mp4` or another browser-viewable video format available without a system dependency;
- `tracking-keyframe.png` — required fallback/summary frame;
- `coverage-keyframe.png` — required fallback/summary frame; and
- optionally small CSV/JSON trajectory records if needed to reproduce the plots.

The exact video container will be selected from formats that the isolated Python environment can generate. If reliable video encoding is unavailable, a documented animated image plus the required PNG keyframes satisfies the visual fallback; the report must state the limitation.

Render overlays should clearly distinguish:

- target truth, shown only as evaluation ground truth;
- target estimate and uncertainty;
- planned routes;
- actual UUV trajectories;
- current commands/waypoints;
- region boundary and sensor/constraint geometry when source-backed; and
- elapsed simulation time and key metric values.

The repository-local virtual environment, raw frame sequences, caches, temporary logs, and other bulky intermediates will not be committed.

## 9. Local environment design

Runtime work will use a repository-local `.venv` created from the already available Python 3.12.13 interpreter. Dependencies will be installed only into that virtual environment using the repository's declared metadata plus the minimum headless rendering/test dependencies approved in the implementation plan.

This design does not authorize changes to the existing Conda environment; it is used only as the source of the Python executable. No activation, package installation, upgrade, or configuration write will target Conda itself. No global PATH or file association will be changed.

Before environment creation and dependency installation, the exact commands, files, download/network effects, estimated disk usage, verification, and non-destructive rollback procedure must be disclosed and separately approved under the project governance rules.

## 10. Defect decision rules

A code change is justified only when at least one of the following is demonstrated:

- current behavior violates an explicit current requirement;
- current behavior violates a configured physical/safety constraint;
- current behavior breaks a mathematical invariant;
- a documented call chain is unreachable or fails at runtime;
- planned commands are not applied to physical UUV state;
- deterministic inputs produce unintended nondeterministic results; or
- the same trajectory and evaluation definitions expose a reproducible mismatch between plan and execution.

Ambiguity, an absent coverage-footprint model, or a weak research claim is not automatically a software defect. Those findings may require documentation or a future modeling decision rather than an implementation change.

## 11. Error handling and stop conditions

Stop implementation and report the evidence if:

- a required step would modify an existing Conda environment, system configuration, PATH, Node installation, ROS workspace, service, port, or unrelated project;
- a run requires a real external LLM, real robot, secret, paid API, or uncontrolled network service;
- the worktree contains unowned or unexplained changes that overlap the task;
- verification would require deleting or overwriting user material;
- push requires an interactive credential or authentication action that Codex cannot safely complete;
- a documentation/source conflict materially changes the intended behavior; or
- the remaining runtime cannot produce honest same-day evidence without lowering the agreed acceptance depth.

For every failed command, preserve the command, exit status, relevant output, and diagnostic conclusion. Do not hide failures, replace a failing test with a weaker one, or convert an unverified result into a success claim.

## 12. Git and delivery structure

All work remains on `review/uuv-tracking-coverage-20260827`. Expected commit boundaries are:

1. design specification;
2. tests and minimal code fixes, if evidence requires them; and
3. final report, metrics, and visual evidence.

The final branch will be pushed to `origin` only after fresh verification. It will not be merged into `master`. Push success and the remote branch tip will be checked explicitly.

## 13. Completion criteria

This task is complete only when the final report distinguishes verified facts from source facts, document claims, and remaining inference, and records:

- the inspected tracking and coverage call chains;
- current test commands and results;
- each confirmed defect, its failing evidence, fix, and regression result;
- a fixed-seed short tracking/control result;
- a fixed-seed short coverage-execution result;
- safety, kinematic, determinism, and geometry checks;
- machine-readable metrics;
- video or explicit fallback visuals from the evaluated trajectories;
- all modified and created files;
- environment and reproduction details;
- unverified items and research limitations;
- the commit IDs; and
- successful push of the review branch without a merge.

Passing tests alone does not prove control effectiveness. A single successful simulation does not prove statistical robustness. Conversely, absence of a defensible area-coverage footprint does not erase valid evidence that UUVs executed distinct serpentine lanes; it limits the strength and wording of the conclusion.

## 14. Primary repository references

- `README.md`
- `pyproject.toml`
- `configs/scenario/uuv_only_single_target.yaml`
- `configs/environment_uuv_only.yaml`
- `configs/tracking.yaml`
- `src/underwater_tracking/planning/coverage.py`
- `src/underwater_tracking/planning/search_control.py`
- `src/underwater_tracking/planning/mission_optimizer.py`
- `src/underwater_tracking/simulation/engine.py`
- `src/underwater_tracking/runtime/mission_controller.py`
- `src/underwater_tracking/api/evaluation.py`
- `src/underwater_tracking/verification/physics_invariants.py`
- `tests/planning/test_coverage_paths.py`
- `tests/planning/test_search_control.py`
- `tests/tracking/test_imm_uif.py`
- `docs/superpowers/specs/2026-08-14-underwater-tracking-agent-design.md`
- `docs/superpowers/specs/2026-08-19-uuv-only-carrier-region-mission-design.md`
- `docs/superpowers/specs/2026-08-21-uuv-initialization-local-perception-design.md`
- `docs/superpowers/specs/2026-08-23-system-goal-acceptance-criteria.md`
- `docs/superpowers/specs/2026-08-26-dedicated-uuv-tracking-mode-design.md`
- `docs/superpowers/audits/2026-08-19-uuv-only-carrier-region-mission-verification.md`
- `docs/audit-hyperparameters.md`
