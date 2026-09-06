# UUV Tracking Control C-01–C-06 Remediation Design

**Date:** 2026-09-06  
**Status:** Approved by the user for implementation on an isolated branch  
**Branch:** `fix/uuv-tracking-control-c01-c06-20260906`

## 1. Goal and scope

This change closes only the **UUV tracking control** findings C-01 through C-06 in
`docs/three-uuv-tracking-team-remediation-and-acceptance.md`. The intended behavior is
defined by `docs/three-uuv-tracking-backend-state-machine-overview.md`.

The implementation must make the repository-native, no-network, seed-42 production
path demonstrate all of the following:

- a newer execution snapshot can be admitted before the current snapshot expires;
- an injected non-executable candidate does not poison later valid candidates;
- public fused observations, not truth or a synthetic probability, drive the two-cycle
  transition from `ACTIVE_SCAN` to `PASSIVE_TRACK`;
- active coverage is accumulated from physical, source-backed sonar transmissions;
- regional handoff has no owner gap and requires all three successor observations from
  the current observation cycle;
- regional replacement and dedicated-mode restoration keep incoming and outgoing
  groups distinct and preserve ownership until takeover is valid;
- one `OperationalFrame` exposes enough state and evidence to audit every decision.

This work does not change the frontend, LLM policy, other users' environments, the
legacy ROS workspaces, or real hardware behavior.

## 2. Verified starting facts

The current source contains the core three-UUV lifecycle and synthetic state-machine
tests, but the documented 600-second live run remains in `ACTIVE_SCAN` with no owner.
The existing acceptance test injects `entry_probability=0.95`, so it does not prove the
production observation-to-probability path.

Static inspection also establishes these contract conflicts:

1. Rolling snapshots allocate the same baseline UUV identifiers to replacement groups,
   while snapshot validation forbids one member from belonging to two groups.
2. An owner group can be projected as `EXITING` before its successor is deployed,
   observed, and made owner.
3. published coverage can be derived from planned routes or visibility rather than
   executed sonar emissions.
4. the `active_ping` event is coupled to a target echo, so an empty region can be scanned
   without source-backed transmission evidence.
5. the execution frame omits the confirmation, scan, handoff, and blocking evidence
   required by C-06.

These facts identify the first tests to write. Runtime effects remain hypotheses until
the baseline and regression tests reproduce them on this branch.

## 3. Chosen approach

Use a contract-first, incremental remediation rather than a symptom patch or a broad
state-machine rewrite.

The change will preserve the existing public architecture:

```text
configured policy and snapshot
  -> execution coordinator admission
  -> mission controller lifecycle
  -> simulation deployment / sensing / motion
  -> public fused estimate and region probability
  -> mission controller transition / handoff
  -> frame builder and verification audit
```

Each layer will gain only the state needed to make its own decision authoritative and
observable. Existing model names and event ordering are retained when compatible.

## 4. Execution snapshot admission and resource identity (C-01)

Snapshot admission remains a compare-and-set transaction. Validation covers scenario,
base execution revision, freshness, geometry, resources, and evidence before any
controller state is mutated. A rejection restores the complete previous controller
state and records a structured reason; terminal metadata belonging to an older candidate
cannot prevent a higher valid revision from committing.

Replacement groups must never reuse member identifiers that are still owned by visible
outgoing groups. When a rolling revision changes region geometry, the controller creates
a distinct incoming group with a new deployment revision and three unique member IDs.
The engine's existing dynamic-UUV materialization path then creates the corresponding
physical entities. No member is transferred or released until the lifecycle transition
that owns that action commits.

Progress from unchanged regions is merged forward by stable region/group identity.
Revisions, deployment revisions, event times, coverage, confirmation counts, ownership,
and resource mileage are monotonic unless the state machine explicitly starts a new
round or replacement generation.

## 5. Public-estimate entry transition (C-02)

The production observation cycle is the only input to entry confirmation:

```text
active/passive sensor observation IDs
  -> selected group observation set
  -> public fused target report
  -> probability mass inside each task region
  -> confirmation counter
  -> atomic group mode and owner transition
```

Group fusion will select the strongest valid current-cycle observation set rather than
choosing by lexical group ID. Region probabilities are finite values derived from the
public belief and covariance. Missing, non-finite, below-threshold, or simultaneous-exit
inputs reset confirmation to zero with a structured reset reason and source evidence IDs.

At `2/2` confirmations, all three group members switch to passive in one controller
transition. The first effective passive group becomes the unique owner in the same
transition. Evaluation truth is never accepted as an operational input.

## 6. Physical scan progress and coverage (C-03)

An active sonar **transmission** and a target **echo** are separate facts. A deployed,
healthy active group emits its configured physical ping on schedule even when no target
is in range. Echo generation and target observation still depend on physical geometry.

For each region and scan round, runtime scan state records:

- current route index and normalized route progress for each assigned member;
- scan round number;
- source-backed ping IDs and ping count;
- the de-duplicated union of ping footprints clipped to the region polygon;
- active coverage fraction;
- completion status and its threshold.

Coverage is monotonic within a round. Planned waypoints never directly add coverage.
Overlapping ping footprints are counted once. A completed route loops into the next scan
round; route completion alone does not set coverage to 100 percent. Replacement starts a
new group generation without leaking the outgoing group's source evidence.

## 7. Handoff, replacement, and dedicated restoration (C-04/C-05)

Regional handoff is adjacent only. A successor is eligible when all three members are
deployed, healthy, passive, and each has a valid observation from the same current
observation cycle. Until then, the old group remains owner and the execution frame
publishes the exact blocking reason and per-member state.

The successful transition is atomic and ordered:

1. set the new unique owner;
2. mark the old owner `EXITING`;
3. keep the old group and resources visible until boundary-exit evidence arrives;
4. mark it `DISAPPEARED` and release its resources exactly once.

Each regional slot can hold one outgoing/incoming pair. A newer geometry revision
supersedes an older pending replacement without rolling back accepted progress. Four
simultaneous replacements therefore publish at most eight groups and 24 unique visible
UUVs.

Dedicated mode exposes only its current three-member passive owner after other groups
finish boundary exit. When any owner member reaches 7,000 m remaining mileage, the
latest four regional groups are created. Regional mode is restored only after the new
owner group is deployed and has three current-cycle passive observations; ownership is
transferred before the dedicated group exits.

## 8. Authoritative OperationalFrame (C-06)

The execution projection will expose, using bounded typed fields:

- execution and geometry revisions plus event time;
- per-region probability, confirmation current/required, and reset reason;
- per-region route progress, scan round, active coverage, ping count, and completion;
- group lifecycle, sensor mode, three members, and deployment revision;
- current owner, pending successor, replacement pairing, and latest revision;
- per-successor-member deployment/health/passive-observation state;
- structured blocking reason and source evidence IDs.

The frame builder reads controller/engine runtime state; it does not reconstruct state
from UI history and does not inspect evaluation truth. JSONL, WebSocket, REST, replay,
and in-process readers continue to serialize the same canonical model.

## 9. Error handling and safety invariants

- Candidate validation and apply are all-or-nothing.
- Every owner-bearing frame has exactly one owner; zero-owner frames are permitted only
  before the first valid passive acquisition.
- Event IDs and resource release operations are idempotent.
- Non-finite probability or geometry data fail closed with an observable reason.
- No truth field, evaluation frame, or configured target start position is used by a
  controller decision.
- No existing evidence directory is overwritten or deleted.

## 10. Verification strategy

Development is test-driven. Each contract first receives a focused failing test, then a
minimal implementation and a focused green run. Verification proceeds through:

1. snapshot admission/recovery and resource-identity unit tests;
2. public fusion, confirmation, reset, and atomic passive-transition tests;
3. physical ping, coverage monotonicity, overlap de-duplication, and scan-round tests;
4. 0/3 through 3/3 handoff and owner-gap tests;
5. four-pair replacement and dedicated-restoration tests;
6. operational-frame schema, canonical transport, and truth-isolation tests;
7. related integration/acceptance tests and the complete test suite;
8. two identical seed-42, 120-step repository-native production audits.

Completion requires fresh passing evidence for every C-01–C-06 item. If the runtime
evidence contradicts this design or requires files outside the approved scope,
implementation stops for renewed user review.

