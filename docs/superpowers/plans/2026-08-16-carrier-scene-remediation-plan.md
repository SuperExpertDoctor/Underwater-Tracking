# Carrier lifecycle and consistency remediation plan

> This follow-up plan addresses the merge-gate findings from the full-branch
> review of the carrier scene implementation. It preserves the existing
> LangGraph human-in-the-loop and plan-version flow while making deployment
> state executable and making the browser rendering contract testable.

## Goals

- Make `onboard -> deployed -> returning -> onboard` a real, deterministic
  engine lifecycle driven by explicit recovery/deployment commands and plan
  actions, with carrier-owned relationship lists derived from one source.
- Reject or exclude onboard/returning/failed UUVs consistently in human
  assignment, optimizer, commit validation, and simulation observation pools.
- Keep operational frames truth-safe and validate that UUV deployment states
  agree with carrier relationship lists.
- Fix carrier heading and UUV state visual cues, enlarge selection hit testing
  to the rendered sprite, and make Playwright exercise visible carrier and
  recovery paths plus image-failure fallback.
- Verify old JSONL frames, LangGraph re-planning, prompts/decision factors,
  Python 3.11 compatibility, and the complete frontend suite before merge.

## Task sequence

1. **Authoritative deployment lifecycle** — add an engine-owned deployment
   controller/state transition API, handle `rotate`/`return` and `track`
   plan-command actions, move returning UUVs toward the carrier, complete
   recovery on proximity, emit deployment events, and add engine integration
   tests for the full cycle and frame/log output.
2. **Canonical contracts and resource availability** — validate carrier/UUV
   relationship consistency at domain/UI boundaries, add one shared
   deployable-resource predicate, and apply it to directives, optimizer,
   commit, verification/allocator paths, plus UI assignment filtering and
   tests for onboard/returning rejection.
3. **Renderer correctness** — apply the carrier sprite orientation offset,
   retain active/failed/reserved cues over loaded UUV images, use a real
   view-aware recovery segment, and derive UUV hit testing from rendered
   dimensions with focused tests.
4. **Real browser integration coverage** — use an in-bounds returning carrier
   fixture with matching IDs, assert visible carrier/link/card behavior,
   exercise asset 404 fallback, and cover old frame normalization.
5. **Final verification and merge review** — run all Python/frontend tests,
   static checks, E2E/visual inspection, re-review the full branch, and
   fast-forward merge into `master`.

## Invariants

- `OperationalFrame` never contains target truth or evaluation-only state.
- A non-failed UUV appears in exactly one carrier relationship list and its
  `deployment_state` matches that list; failed UUVs are omitted from carrier
  lists and carry `failed`.
- Only deployed, non-failed, non-returning UUVs are assignment/planning
  resources.
- Live WebSocket and JSONL replay serialize/parse the same frame contract;
  old frames normalize missing carrier/deployment fields to compatibility
  values.
- Human assignment remains preview -> explicit apply -> next LangGraph
  strategic replanning; no UI scheduler or renderer may mutate plan state.
