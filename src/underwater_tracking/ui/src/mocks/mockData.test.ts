import { describe, expect, it } from "vitest";
import { createMockFrame, MOCK_FRAME_COUNT } from "./mockData";

describe("versioned operational mock", () => {
  it("starts with onboard UUVs and an estimator-only low-confidence contact", () => {
    const frame = createMockFrame(0);
    expect(frame.carrier?.position).toEqual({ x: -3000, y: -3000 });
    expect(frame.carrier?.onboard_uuv_ids).toHaveLength(12);
    expect(frame.uuvs.every((uuv) => uuv.deployment_state === "onboard")).toBe(
      true,
    );
    expect(frame.target_estimates[0]?.quality.quality_score).toBeLessThan(0.5);
    expect(frame.plans).toHaveLength(0);
  });

  it("models a fixed-patrol carrier and overlapping A/B bearing formations", () => {
    expect(createMockFrame(240).carrier?.position).toEqual({
      x: 3000,
      y: -3000,
    });
    const overlap = createMockFrame(174); // 870 s: immediately after v2 is committed
    expect(overlap.plan_version).toBe(2);
    expect(
      overlap.uuvs.filter((uuv) => uuv.status === "tracking"),
    ).toHaveLength(8);
    expect(overlap.groups).toHaveLength(2);
    expect(overlap.bearing_rays.length).toBeGreaterThanOrEqual(4);
    expect(overlap.regional_plans?.target.regions).toHaveLength(5);
    expect(
      overlap.regional_plans?.target.regions.some(
        (region) => region.display_name === "region_3",
      ),
    ).toBe(false);
    expect(
      overlap.regional_plans?.target.regions.some(
        (region) => region.display_name === "region_3a",
      ),
    ).toBe(true);
    expect(overlap.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ event_type: "prediction_revision" }),
      ]),
    );
    expect(overlap.llm_thinking).toContain("协方差不确定性");
  });

  it("fans revised v2 cells out from their v1 predecessors instead of teleporting them", () => {
    const before = createMockFrame(173); // 865 s
    const committed = createMockFrame(174); // 870 s
    const region = (frame: ReturnType<typeof createMockFrame>, name: string) =>
      frame.regional_plans?.target.regions.find(
        (candidate) => candidate.display_name === name,
      )!;
    const center = (geometry: { x: number; y: number }[]) => ({
      x: (geometry[0]!.x + geometry[2]!.x) / 2,
      y: (geometry[0]!.y + geometry[2]!.y) / 2,
    });
    const distance = (
      left: { x: number; y: number },
      right: { x: number; y: number },
    ) => Math.hypot(left.x - right.x, left.y - right.y);
    expect(
      distance(
        center(region(before, "region_2").geometry),
        center(region(committed, "region_2_revised").geometry),
      ),
    ).toBeLessThan(100);
    expect(
      distance(
        center(region(before, "region_3").geometry),
        center(region(committed, "region_3a").geometry),
      ),
    ).toBeLessThan(100);
  });

  it("propagates the maneuver into uncertainty, UUV return, and relay allocation", () => {
    const degraded = createMockFrame(168); // 840 s: innovation spike before active verification
    expect(
      degraded.metrics.find((metric) => metric.metric_id === "tracking_quality")
        ?.value,
    ).toBeLessThan(0.65);
    expect(
      createMockFrame(180).target_estimates[0]?.covariance_ellipse.semimajor_m,
    ).toBeGreaterThan(2000);

    const handedOff = createMockFrame(180);
    expect(handedOff.carrier?.returning_uuv_ids).toEqual([
      "uuv_00",
      "uuv_01",
      "uuv_02",
      "uuv_03",
    ]);
    expect(
      handedOff.uuvs
        .filter((uuv) => uuv.status === "tracking")
        .map((uuv) => uuv.uuv_id),
    ).toEqual(["uuv_04", "uuv_05", "uuv_06", "uuv_07"]);
    expect(
      handedOff.regional_plans?.target.regions.find(
        (region) => region.display_name === "region_3b",
      )?.assigned_usv_ids,
    ).toEqual(["usv_03"]);
  });

  it("fades completed regions from the current map and models a recoverable relay outage", () => {
    const recentHandoff = createMockFrame(198); // 990 s, A completed fewer than 180 s ago
    expect(
      recentHandoff.regional_plans?.target.regions.some(
        (region) => region.display_name === "region_1",
      ),
    ).toBe(true);
    const expiredHandoff = createMockFrame(217); // 1085 s, A is historical only
    expect(
      expiredHandoff.regional_plans?.target.regions.some(
        (region) => region.display_name === "region_1",
      ),
    ).toBe(false);

    const outage = createMockFrame(336);
    expect(
      (outage.usvs ?? []).find((usv) => usv.usv_id === "usv_03")?.connected,
    ).toBe(false);
    expect(
      outage.regional_plans?.target.regions.find(
        (region) => region.display_name === "region_3b",
      )?.status,
    ).toBe("degraded");
    const recovered = createMockFrame(348);
    expect(
      (recovered.usvs ?? []).find((usv) => usv.usv_id === "usv_03")?.connected,
    ).toBe(true);
  });

  it("enters a continuous completed-plan coast instead of reverting at 2400 seconds", () => {
    const before = createMockFrame(479); // 2395 s
    const complete = createMockFrame(480); // 2400 s
    expect(complete.plans[0]?.status).toBe("completed");
    expect(
      complete.regional_plans?.target.regions.some(
        (region) => region.display_name === "region_4",
      ),
    ).toBe(true);
    const beforePoint =
      before.target_estimates[0]?.prediction?.centerline_xy[1];
    const completePoint =
      complete.target_estimates[0]?.prediction?.centerline_xy[1];
    expect(
      Math.hypot(
        (beforePoint?.x ?? 0) - (completePoint?.x ?? 0),
        (beforePoint?.y ?? 0) - (completePoint?.y ?? 0),
      ),
    ).toBeLessThan(100);
    expect(createMockFrame(516).regional_plans).toEqual({}); // 2580 s: completed-plan fade is over
  });

  it("starts a new coastal plan after recovery instead of extending v2", () => {
    const preparation = createMockFrame(540); // 2700 s: v3 committed, A is redeploying
    expect(preparation.plan_version).toBe(3);
    expect(preparation.ledger.map((row) => row.final_plan_version)).toEqual([
      1, 2, 3,
    ]);
    expect(
      preparation.plan_timeline?.find((row) => row.plan?.version === 3)?.plan
        ?.status,
    ).toBe("active");

    const coastalTrack = createMockFrame(570); // 2850 s: first v3 segment is active
    expect(coastalTrack.regional_plans?.target.regions).toHaveLength(3);
    expect(
      coastalTrack.regional_plans?.target.regions.find(
        (region) => region.display_name === "region_5a",
      )?.status,
    ).toBe("active");
    expect(
      coastalTrack.uuvs
        .filter((uuv) => uuv.status === "tracking")
        .map((uuv) => uuv.uuv_id),
    ).toEqual(["uuv_00", "uuv_01", "uuv_02", "uuv_03"]);
  });

  it("keeps physical platforms and the estimate continuous across every five-second frame", () => {
    const distance = (
      left: { x: number; y: number },
      right: { x: number; y: number },
    ) => Math.hypot(left.x - right.x, left.y - right.y);
    for (let index = 1; index < MOCK_FRAME_COUNT; index += 1) {
      const previous = createMockFrame(index - 1);
      const current = createMockFrame(index);
      expect(
        distance(previous.carrier!.position, current.carrier!.position),
      ).toBeLessThanOrEqual(30);
      expect(
        distance(
          previous.target_estimates[0]!.mean,
          current.target_estimates[0]!.mean,
        ),
        `estimate at ${current.sim_time_s}s`,
      ).toBeLessThanOrEqual(100);
      expect(
        distance(
          previous.target_estimates[0]!.prediction!.centerline_xy[0]!,
          current.target_estimates[0]!.prediction!.centerline_xy[0]!,
        ),
        `prediction origin at ${current.sim_time_s}s`,
      ).toBeLessThanOrEqual(350);
      current.uuvs.forEach((uuv) => {
        const before = previous.uuvs.find(
          (candidate) => candidate.uuv_id === uuv.uuv_id,
        )!;
        expect(
          distance(before.position, uuv.position),
          `UUV ${uuv.uuv_id} at ${current.sim_time_s}s`,
        ).toBeLessThanOrEqual(180);
      });
      current.usvs!.forEach((usv) => {
        const before = previous.usvs!.find(
          (candidate) => candidate.usv_id === usv.usv_id,
        )!;
        expect(
          distance(before.position, usv.position),
          `USV ${usv.usv_id} at ${current.sim_time_s}s`,
        ).toBeLessThanOrEqual(170);
      });
    }
  });

  it("keeps carrier relationships and active plan revisions coherent", () => {
    for (let index = 0; index < MOCK_FRAME_COUNT; index += 1) {
      const frame = createMockFrame(index);
      const ids = [
        ...(frame.carrier?.onboard_uuv_ids ?? []),
        ...(frame.carrier?.deployed_uuv_ids ?? []),
        ...(frame.carrier?.returning_uuv_ids ?? []),
      ];
      expect(new Set(ids).size).toBe(frame.uuvs.length);
      expect(frame.uuvs.every((uuv) => ids.includes(uuv.uuv_id))).toBe(true);
      const active = frame.plans.find((plan) => plan.status === "active");
      if (
        frame.plan_version === 0 ||
        (frame.sim_time_s >= 2400 && frame.sim_time_s < 2700) ||
        frame.sim_time_s >= 4500
      )
        expect(active).toBeUndefined();
      else expect(active?.version).toBe(frame.plan_version);
    }
  });
});
