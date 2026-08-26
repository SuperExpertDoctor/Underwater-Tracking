# Dedicated UUV Tracking Mode Design

## Goal

Support two mutually exclusive tracking modes for each target submarine:

- `regional`: the default mode. Prediction-derived task regions and their UUV
  task groups perform a bounded, handoff-based tracking flow.
- `dedicated`: an operator-approved group follows one target continuously,
  independent of regional boundaries, until its endurance reserve requires a
  return. The system then resumes regional planning automatically.

The natural-language command "continue tracking" is the canonical request to
enter dedicated mode. A command to resume regional handoff is the canonical
early-release request.

## Authority And State

The LLM parses intent only. It returns a structured tracking-mode request and
one target scope; it must not select or invent UUV identifiers. The directive
preview resolves the selected target's current active tracking group from the
live situation, stores those resolved members in the persisted directive, and
shows them to the operator before application.

Application revalidates the resolved members against the current situation.
On success, their membership is frozen for the lifetime of the dedicated
directive. A regional replan cannot replace or add members to that group.

An applied dedicated directive synchronizes its frozen members through the
existing reservation path into `SimulationEngine` and
`MissionController.set_dedicated_group`. `MissionController` remains the sole
owner of execution modes and resource transitions.

## Entry And Execution

When dedicated mode becomes active for a target:

1. The frozen group enters `DEDICATED_TRACK`, uses passive tracking, and is
   routed continuously to the latest estimated target position.
2. Other deployed UUVs assigned to that target's regional tasks enter
   `RETURN_REQUIRED`. The simulation sends them to their task boundary and
   removes them from waterborne execution using the existing boundary-exit
   lifecycle.
3. Existing regional tasks may remain as audit state, but their UUVs cannot
   continue tracking the dedicated target or be reallocated into the dedicated
   group.

The mode is persistent. Unrelated directives do not release it. The only
normal exits are an explicit regional-resume directive, a member failure, or
the endurance reserve condition.

## Exit And Recovery

Before an assigned member exhausts its range, the existing resource reserve
rule changes it to `RETURN_TO_REGION`. Once the dedicated group is released,
the runtime emits a dedicated-release event and queues a strategic replan.
The next verified executable plan restores `regional` tracking with fresh task
regions and UUV task groups from currently available resources. UUVs that
previously crossed the boundary remain unavailable until their normal refuel
cooldown ends.

## Safety And Error Handling

- A dedicated request with zero, multiple, unknown, or unavailable resolved
  members resolves to `needs_clarification` and cannot be applied.
- The LLM may never bypass preview, confirmation, resource validation, or
  executable-plan validation.
- A failed LLM request leaves the current execution mode unchanged.
- Dedicated membership and mode changes are represented in directive and
  runtime events so replay and operational frames remain auditable.

## Verification

Automated coverage must prove that:

- the directive schema and parser payload expose the mode request;
- applying the preview freezes the current tracking group and causes the
  engine to enter dedicated tracking;
- non-dedicated regional UUVs exit at the boundary;
- a dedicated group ignores ordinary regional handoffs while following the
  live target estimate;
- endurance reserve releases the group and queues a regional replan;
- malformed, ambiguous, and unavailable-mode requests leave the existing
  regional workflow unchanged.
