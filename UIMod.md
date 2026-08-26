# UI Modification Record

## Region-focus interaction

### Change

- Double-clicking a predicted task region (`R01`, `R02`, etc.) doubles the current local map zoom (up to 8×) and centres that region in the visible map area. A single click only selects the region.
- The existing **fit current focus** map-tool button restores the normal 1× fitted view.

### Scope

- Modified `src/underwater_tracking/ui/src/components/CanvasMap.tsx` and its unit test `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`.
- No backend API, WebSocket payload, simulation state, planning data, or persistence format was changed.

### Behaviour and implementation notes

- The feature calculates the centre from the selected region's existing polygon geometry and updates the Canvas viewport (`zoom` and `pan`) locally.
- All map layers use the same viewport transform: region overlays, predictions, UUVs, carriers, targets, trails, labels, and canvas hit-testing remain aligned after focusing.
- Clicking a UUV or target keeps its existing selection behaviour and does not trigger region focus.

### Verification

- Added unit-level assertions for 2× current-zoom multiplication, the 8× cap, and region centring in `CanvasMap.test.ts`.
- Ran the related CanvasMap test file and the production frontend build successfully.
