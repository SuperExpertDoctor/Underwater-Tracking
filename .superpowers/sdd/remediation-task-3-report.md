# Remediation Task 3 report

Base: `92dcfbc`

## Scope

- Updated `CanvasMap.tsx` and `geometry.ts`, with focused CanvasMap and geometry tests only.
- Did not change backend code, React panels, or E2E tests.
- Left the pre-existing `remediation-task-2-report.md` modification and untracked `src/underwater_tracking.egg-info/` untouched.

## Implementation

- Documented and applied a `+pi/2` carrier-image orientation offset: the local carrier PNG is bow-up, while the vector fallback treats heading 0 as right/east. Both now point right at heading 0 and up at heading `pi/2`.
- Kept status cues visible when the UUV PNG loads: state color ring (active/failed), reserved violet ring, and selected light outline are drawn around the rendered sprite.
- Made `recoverySegment` accept and use the current view, including zoom and pan.
- Replaced the fixed 18px circular UUV click target with a rotated hit rectangle calculated from the same clamped sprite dimensions used by rendering, plus a 6px tolerance.

## TDD evidence

1. Added focused heading, loaded-image cue, view-aware recovery, and rotated sprite-edge hit tests.
2. Observed the initial focused run fail because the new APIs were absent and recovery geometry ignored the supplied view.
3. Implemented the minimum rendering and geometry changes, then ran the focused tests green.

## Verification

- `npm test -- src/components/map/geometry.test.ts src/components/CanvasMap.test.ts` — 2 files, 9 tests passed.
- `npm test` — 13 files, 34 tests passed.
- `npm run build` — TypeScript check and Vite production build passed.
- Ruff is not applicable to this TypeScript-only task; E2E remains intentionally deferred to remediation Task 4.
