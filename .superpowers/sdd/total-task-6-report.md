# Task 6 Report

Implemented command-center map focus and display scale controls.

## Review Follow-up

- Wired regional hit testing into the canvas click path. After UUV, USV, and target markers miss, screen coordinates are transformed through the current camera before selecting a detailed regional cell. The selected region is shown in a persistent map status block; UUV selection keeps its existing priority and callback behavior.
- Added a 1 km minimum two-axis camera span for the single-target-mean fallback when no prediction centerline or visible regional cells exist. This retains nearby operating space without pulling the hidden detection range into the default bounds.
- Added regression coverage for the rendered canvas coordinate conversion and region selection, UUV marker selection, and the lone-target fallback bounds.

- Added local `ViewConfig` defaults for prediction-corridor focus, 15% padding, 16 grid divisions, marker pixel limits, radar scale, playback rate, and hidden detection range.
- Camera bounds now prioritize target means, corridor geometry, and visible regional cells. Detection circles affect bounds only when explicitly visible or in `full_area` mode.
- Regional cells retain polygon detail for hit testing; low zoom aggregates labels and high zoom uses stable `R01`-style labels. UUV, USV, and target markers render persistent rings and role/status cues.
- View-only configuration is removed by `toPlanningPayload`; App state remains local and API request shapes are unchanged.

Verification run from `src/underwater_tracking/ui`:

```text
npm test -- --run src/components/CanvasMap.test.ts src/types/regionalTasks.test.ts
Test Files  2 passed (2)
Tests  12 passed (12)

npm run build
tsc --noEmit && vite build
exited with code 2 at the existing App.tsx and regionalTasks type errors
```

The build failure was already present in parent commit `345249a`; it is not caused by the Task6 map change and remains for Task11/baseline repair.

Risk: camera focus currently resets only through the map focus control after operator pan/zoom, preserving manual inspection during streaming updates. The view configuration is intentionally client-local and is not persisted between reloads.
