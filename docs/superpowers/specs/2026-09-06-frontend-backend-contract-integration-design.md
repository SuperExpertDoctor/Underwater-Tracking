# Frontend Backend Contract Integration Design

## Decision

Merge \`origin/front\` into a feature branch based on \`branch1\`, then integrate the evidence-backed UI against the existing authoritative runtime frame pipeline. The front branch's presentation components and TypeScript contract additions are retained. Its mock frame data, mock live/replay hooks, mock mode switch, and mock-only acceptance path are excluded from the final tree.

## Boundaries

The UUV runtime owns deployment, scanning, passive observation, handoff, replacement, ownership, and lifecycle transitions. The estimator/world model owns public estimate freshness and prediction provenance. The LLM layer remains advisory and context-bound. The API publisher owns one sanitized public frame shared by snapshot HTTP, WebSocket, JSONL, and Replay. The UI only renders that frame and sends existing context-bound commands.

## Data Flow

\`\`\`text
SimulationEngine + MissionController + public estimator state
  -> OperationalExecutionSnapshot / public TargetEstimate
  -> frame_builder
  -> one sanitized OperationalFrame
  -> HTTP snapshot / WebSocket / JSONL / Replay
  -> real UI hooks
  -> evidence-backed map, timeline, status board, assistant panels
\`\`\`

The new public evidence includes per-region scan telemetry, current-cycle entry evidence, handoff evidence, confirmed deployment/passive-observation IDs, replacement batch metadata, and target estimate freshness. Missing legacy fields are shown as unknown/unavailable.

## Mock-Free Rule

The production UI contains no mock data source, no mock mode URL/environment switch, and no mock replay/stream hook. Operational-data acceptance tests connect to a real FastAPI server and real simulation run. Pure presentation tests may construct the smallest typed value needed to exercise a formatting branch, but they cannot be imported by or used as a runtime data source.

## Truth Boundary

Formal operational frames, ordinary HTTP/WebSocket/JSONL/Replay responses, world-model output, assistant context, and UI state contain no target truth/evaluation fields. Evaluation remains an explicit separate channel.

## Verification

Use the \`underwater-tracking\` Conda environment for Python checks, the UI package scripts for TypeScript checks, and the real \`seed=42\`, \`--duration-s 480\` accelerated audit twice. Compare transport frame identity, evidence fields, rendered status, truth sanitization, and deterministic trace digests.

