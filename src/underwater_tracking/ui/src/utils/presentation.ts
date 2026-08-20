/** The command UI is configured for one tracked target; keep backend IDs internal. */
export function displayTargetName(targetId: string | null | undefined): string {
  return targetId ? "target" : "—";
}
