export type FocusMode = "prediction_corridor" | "full_area";

export interface ViewConfig {
  focusMode: FocusMode;
  radarScale: number;
  predictionPadding: number;
  gridDivisions: number;
  targetMarkerPixels: number;
  uuvMarkerPixels: number;
  playbackRate: number;
  showDetectionRange: boolean;
}

export const DEFAULT_VIEW_CONFIG: ViewConfig = {
  focusMode: "prediction_corridor",
  radarScale: 1,
  predictionPadding: 0.15,
  gridDivisions: 24,
  targetMarkerPixels: 28,
  uuvMarkerPixels: 30,
  playbackRate: 1,
  showDetectionRange: true,
};

const VIEW_CONFIG_KEYS = [
  "focusMode",
  "radarScale",
  "predictionPadding",
  "gridDivisions",
  "targetMarkerPixels",
  "uuvMarkerPixels",
  "playbackRate",
  "showDetectionRange",
] as const satisfies readonly (keyof ViewConfig)[];

export function toPlanningPayload<T extends Record<string, unknown>>(payload: T): Omit<T, keyof ViewConfig> {
  const displayKeys = new Set<string>(VIEW_CONFIG_KEYS);
  return Object.fromEntries(Object.entries(payload).filter(([key]) => !displayKeys.has(key))) as Omit<T, keyof ViewConfig>;
}
