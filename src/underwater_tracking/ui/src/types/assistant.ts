export interface ExpertDirectiveView {
  directive_id: string;
  raw_text: string;
  target_scope: string[];
  locked_members: Record<string, string[]>;
  target_priorities: Record<string, number>;
  minimum_quality: Record<string, number>;
  disabled_uuv_ids: string[];
  directive_type: "constraint" | "assignment";
  assignment_target_id: string | null;
  assignment_uuv_ids: string[];
  confidence: number;
  conflicts: string[];
  status: "preview" | "applied" | "rejected" | "needs_clarification";
}
