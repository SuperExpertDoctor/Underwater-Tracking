import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { TargetEstimateView } from "../../types/frames";
import WorldModelEventOverlay from "./WorldModelEventOverlay";

it("places future event markers at the forecast position", () => {
  const targets = [
    {
      target_id: "T1",
      world_model: {
        events: [
          {
            event_id: "event-1",
            event_type: "area_exit_risk",
            horizon: "H3",
            predicted_position: { x: 20, y: 30 },
            confidence: 0.82,
            level: "strategic",
            summary: "预测轨迹将触及任务区边界",
          },
        ],
      },
    },
  ] as unknown as TargetEstimateView[];

  const { container } = render(
    <WorldModelEventOverlay
      targets={targets}
      project={(point) => ({ x: point.x + 1, y: point.y + 2 })}
      width={200}
      height={100}
    />,
  );

  expect(screen.getByLabelText("世界模型未来事件位置")).toBeInTheDocument();
  const marker = container.querySelector('[data-event-type="area_exit_risk"]');
  expect(marker).toHaveAttribute("data-horizon", "H3");
  expect(marker?.querySelector("polygon")).toHaveAttribute(
    "points",
    "21,26 27,32 21,38 15,32",
  );
});
