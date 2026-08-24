# Event-Driven Adaptive Tracking Closure Design

## Goal

Deliver a demonstrable, event-driven adversarial tracking loop in which
operational events automatically produce a physically valid plan revision,
memory preserves the situation and decision evidence, and the command center
shows the resulting execution outcome.

## Scope

The implementation covers four bounded areas:

1. A handoff with stale or insufficient effective observations must degrade the
   predecessor region and must not authorize an invalid handoff.
2. A planning-relevant event must create a rolling planning epoch. A plan that
   passes semantic and physical revalidation is committed automatically; a
   failing plan leaves the current active plan in place, records the reason,
   degrades affected regions, and remains eligible for a later retry.
3. The local multilingual embedding model must be checked at startup. Memory
   work must reach terminal completed or degraded status, and completed work
   must create retrievable short-term or long-term memory rather than remain
   indefinitely pending.
4. The command center must render current operational frames, scene assets,
   prediction and handoff evidence, replay, and memory state on desktop and
   mobile layouts. Its browser tests must use a valid current frame fixture.

## Event-To-Execution Flow

1. The simulation or runtime emits a durable event with an event ID, scenario,
   simulation time, target or platform identity, and evidence references.
2. The runtime deduplicates the event and queues one planning epoch for the
   affected scenario. The epoch stores its base physics revision and source
   event IDs.
3. The planner generates an executable mission plan using the current target
   belief, active resource states, carrier routes, UUV availability, and
   resource-rotation rules.
4. Semantic mission revalidation checks the candidate against the latest
   physics revision. A valid candidate becomes the next active plan without an
   operator confirmation step.
5. A rejected candidate does not replace the active plan. The runtime emits a
   durable rejection event, marks each affected predecessor region degraded
   with the validation reason, and leaves the event eligible for retry when a
   newer observation or resource event arrives.
6. The operational frame publishes the trigger IDs, active plan version,
   revalidation status, degraded reasons, and execution events. The UI renders
   these values as a single evidence chain.

## Handoff Safety

`HANDOFF_PENDING` is a provisional sensing state. If the current cycle lacks
the successor's complete, healthy, deployed, and effective observations, the
mission controller must call the existing handoff-block path. That path moves
the predecessor to `DEGRADED`, records `handoff_blocked`, and prevents recovery
or handoff completion from treating stale evidence as current evidence. A
subsequent current evidence set may enter a newly committed plan, but cannot
retroactively complete the rejected handoff.

## Memory Behavior

The configured `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
model is local-only. Startup checks that it can load and encode one fixed
probe. If it is unavailable, the API exposes a truthful degraded reason before
accepting memory work.

When available, each event-derived or operator-derived work item has a bounded
retry budget. Terminal completion writes a stream event and persists the
resulting short-term context and long-term episodic or semantic material. A
terminal failure writes a degraded stream event with the cause; it does not
remain in `processing` or `pending` after its retry budget is exhausted.

## Command Center Behavior

The application obtains a validated operational snapshot before rendering
frame-dependent map content. Scene assets load from the public asset URLs and
fall back to vector rendering if one image is unavailable. The operational,
prediction-and-handoff, replay, and memory panels use the same authoritative
frame and expose degraded state rather than substituting fabricated data.

The event ledger panel groups each automatic adjustment by trigger event,
planning epoch, validation report, committed or retained plan version, and
observable execution outcome. Operators observe this chain; their confirmation
is not required for event-triggered plan submission.

## Acceptance Criteria

1. The stale-observation handoff test ends with the predecessor region in
   `DEGRADED`, emits `handoff_blocked`, and never emits `handoff_completed`.
2. An eight-minute, 96-step run records at least one adversary decision, one
   event-triggered plan revision after the initial plan, and a valid
   revalidation report for every committed plan.
3. The run produces terminal memory work states and at least one persisted
   short-term or long-term memory item with an evidence reference.
4. Physics invariant tests and UUV physical-execution integration tests pass.
5. Command-center Playwright tests pass for normal assets, single missing asset
   fallback, prediction evidence on desktop/mobile, UUV detail selection, and
   replay.
6. The working tree remains clean apart from the intentional implementation
   changes and test artifacts ignored by Git.
