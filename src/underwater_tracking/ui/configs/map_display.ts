/**
 * Frontend-only geometry for the map overlays.
 *
 * These values control what the operator sees and are deliberately separate
 * from the sensor/target values in OperationalFrame. They must not be used
 * as inputs to simulation, tracking, or planning.
 *
 * Distances are metres. Edit this file and restart/rebuild the UI to change
 * the displayed overlays.
 */
export const MAP_DISPLAY_CONFIG = {
  /** Shared angular width of the UUV sonar fan. */
  uuvSensorSpanRad: Math.PI / 2,
  /** Minimum screen-space spacing between displayed IMM sample markers. */
  predictionSampleSpacingPx: 24,
  /** Maximum displayed semimajor axis of a target uncertainty ellipse. */
  estimateEllipseMaxSemimajorM: 1_500,
  /** Maximum displayed long-axis/short-axis ratio for target uncertainty. */
  estimateEllipseMaxAspectRatio: 4,
  /** Minimum displayed short semi-axis in screen pixels for readability. */
  estimateEllipseMinSemiminorPx: 6,
} as const;
