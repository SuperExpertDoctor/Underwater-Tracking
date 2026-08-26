# Live Motion Fallback Design

## Goal

Starting `python main.py` must immediately publish advancing operational
frames and deploy moving UUVs. A slow, unavailable, or invalid LLM planning
result must not leave every UUV onboard with `plan_version == 0`.

## Design

The live UUV-only entry point will install a deterministic, geometry-derived
operational baseline at startup. It uses the current public prediction and
the existing mission-controller and engine command paths; it does not create
a second motion model. The baseline dispatches a bounded set of UUVs into
non-overlapping search/tracking regions, so the first physical steps contain
real UUV motion and sensor activity.

LLM planning remains asynchronous. A committed LLM plan replaces the baseline
through the existing plan-application path. A rejected regional result is
recorded as planning degradation and leaves the baseline active, rather than
withholding all deployment commands.

Before a regional result is committed, its geometry is normalized against the
public target prediction: a missing centerline is supplied from that
prediction, and region bounds are made non-overlapping. Results that still
cannot satisfy the existing safety and resource constraints remain rejected.

## Error Handling

- Physics and frame publication continue while an LLM epoch is running.
- Invalid LLM regional output does not erase or pause the active baseline.
- The planning health endpoint retains the last validation error for operator
  visibility.
- The baseline is restricted to UUV-only live runs; finite/headless commands
  retain their existing deterministic behavior.

## Verification

Tests will prove that a live run with a failed regional planning result still
has increasing frames, at least one deployed UUV, and a changing UUV position.
Tests will also prove that an LLM plan can replace the baseline and that
normalized regional geometry still satisfies existing validation.
