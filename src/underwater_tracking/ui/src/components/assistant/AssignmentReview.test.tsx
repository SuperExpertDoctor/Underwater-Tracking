import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { DirectiveStatus } from "../../services/assistantApi";
import AssignmentReview from "./AssignmentReview";

const preview: DirectiveStatus = {
  request_id: "assignment:job-1",
  status: "preview",
  directive: {
    directive_id: "S1:assign:T1:UUV-1",
    raw_text: "assignment: UUV-1 -> T1",
    target_scope: ["T1"],
    locked_members: {},
    target_priorities: {},
    minimum_quality: {},
    disabled_uuv_ids: [],
    return_uuv_ids: [],
    directive_type: "assignment",
    assignment_target_id: "T1",
    assignment_uuv_ids: ["UUV-1", "UUV-2"],
    confidence: 1,
    conflicts: [],
    status: "preview",
  },
};

describe("AssignmentReview", () => {
  it("makes the typed assignment preview explicit before applying it", () => {
    const onConfirm = vi.fn();
    render(<AssignmentReview job={preview} onConfirm={onConfirm} />);

    expect(screen.getByText("T1")).toBeInTheDocument();
    expect(screen.getByText("UUV-1、UUV-2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "确认应用指派" }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });
});
