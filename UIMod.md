# UI Modification Record

This document records front-end-only changes made to the command-centre UI. None of the items below change simulation logic, planning behaviour, backend APIs, WebSocket payloads, or persisted data formats.

## 1. Theme switch and visual system

### User-facing behaviour

- A compact theme button is located in the existing top-right action group, beside the connection state.
- The application starts in **dark mode**. In dark mode the button shows a sun icon and switches to light mode; in light mode it shows a moon icon and switches back to dark mode.
- The selected theme is stored under the browser `localStorage` key `underwater-tracking-theme` and is restored on the next visit.
- The light theme uses white primary panels, pale-gray separators, and dark text. It preserves the same layout, components, content, controls, and map layers as dark mode.

### Dark-mode consistency work

- Preserved the dark command-centre visual language as the default: deep charcoal surfaces, restrained teal accents, and high-contrast text.
- Removed the full-width red separator that appeared for failed/awaiting-retry planning runs. Failure remains visible through the red status text and existing error details.
- Added dark-theme styles for all expanded and interactive surfaces so they do not fall back to white, cream, or browser-default controls:
  - bottom drawer tabs, timeline, plan cards, ledgers, metrics, and Memory Steam;
  - current-situation metrics, selected UUV rows, UUV detail controls, and selects;
  - target-submarine brain, adversary-decision details, decision history, and status badges;
  - prediction/relay, assignment graph/list, assistant proposal/evidence, memory tabs/items, empty states, and refresh buttons.

### Files

- `src/underwater_tracking/ui/src/App.tsx`
- `src/underwater_tracking/ui/src/App.css`
- `src/underwater_tracking/ui/src/components/SidebarPanels.css`

## 2. Predicted-region focus

### User-facing behaviour

- Single-clicking a predicted task region keeps the original region-selection behaviour.
- Double-clicking a region (`R01`, `R02`, etc.) doubles the current map zoom and centres that region in the visible map area.
- Zoom is capped at 8×. The existing **适配当前焦点** control restores the standard 1× fitted view.

### Implementation notes

- Focus is calculated from the existing region polygon geometry and updates only the Canvas viewport (`zoom` and `pan`).
- All existing map layers share the same transform, including region overlays, UUVs, carriers, targets, prediction labels, trails, and Canvas hit-testing.
- Selecting a UUV or a target does not trigger region focus.

### Files

- `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`

## Verification

- `npm --prefix src/underwater_tracking/ui test -- --run src/components/CanvasMap.test.ts src/components/RightSidebar.test.tsx`
  - Result: **2 test files, 25 tests passed**.
- `npm --prefix src/underwater_tracking/ui run build`
  - Result: TypeScript check and Vite production build passed.
- Local 60-step demo visual checks covered dark/light switching, persistence after refresh, the bottom drawer, selected UUVs, expanded prediction/relay, and expanded assistant/memory panels.
