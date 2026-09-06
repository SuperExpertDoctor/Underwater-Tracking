import { describe, expect, it } from "vitest";
import { MOCK_FRAMES } from "../mock/mockFrames";
import {
  entryEvidenceForRegion,
  estimatePresentationStatus,
  handoffProgress,
  scanCoveragePercent,
  scanPingCount,
  scanRouteProgressPercent,
  scanTelemetryForRegion,
  targetGroups,
} from "./operationalStatus";

describe("operational evidence presentation", () => {
  it("does not infer scan completion from a completed route", () => {
    const frame = MOCK_FRAMES[0];
    const region = frame.execution!.regions[0];
    const telemetry = scanTelemetryForRegion(region);
    expect(scanRouteProgressPercent(telemetry)).toBe(100);
    expect(scanCoveragePercent(telemetry)).toBe(25);
    expect(scanPingCount(telemetry)).toBe(3);
    expect(telemetry?.scan_completed).toBe(false);
    expect(scanPingCount(scanTelemetryForRegion(frame.execution!.regions[1]))).toBe(0);
    const completed = scanTelemetryForRegion(MOCK_FRAMES[6].execution!.regions[0]);
    expect(scanCoveragePercent(completed)).toBe(90);
    expect(scanPingCount(completed)).toBe(18);
    expect(completed?.scan_completed).toBe(true);
  });

  it("keeps entry confirmations and handoff progress source-backed", () => {
    const entry = entryEvidenceForRegion(MOCK_FRAMES[3].execution, "T1:task:01");
    expect(entry).toMatchObject({ confirmation_count: 2, required_cycles: 2 });
    expect(handoffProgress(MOCK_FRAMES[3].execution)).toMatchObject({ valid: 2, required: 3, missing: ["uuv-06"] });
    expect(handoffProgress(MOCK_FRAMES[4].execution)).toMatchObject({ valid: 3, required: 3, missing: [] });
    const groups = targetGroups(MOCK_FRAMES[4], "T1");
    expect(groups.some((group) => group.lifecycle === "exiting")).toBe(true);
    expect(groups.some((group) => group.ownership_status === "owner")).toBe(true);
  });

  it("takes valid_until over a contradictory prediction health label", () => {
    const expired = MOCK_FRAMES.at(-1)!;
    const target = expired.target_estimates[0];
    expect(target.prediction?.health.status).toBe("valid");
    expect(estimatePresentationStatus(expired, target)).toBe("expired");
    expect(estimatePresentationStatus({ sim_time_s: 100 }, { ...target, estimate_freshness: null })).toBe("live");
    expect(estimatePresentationStatus(
      { sim_time_s: 100 },
      { ...target, prediction: null, estimate_freshness: null },
    )).toBe("unknown");
  });
});
