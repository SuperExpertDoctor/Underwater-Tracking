# USV Ray Frame Contract Design

## Goal

Keep USV passive-sonar bearings in the explicit platform-core GroupManager
update while making carrier-published OperationalFrame, JSONL, and WebSocket
serialization robust when the public SituationSnapshot has no USV positions.

## Decision

The public frame adapter will map only bearing rays whose observer ID exists in
`SituationSnapshot.uuvs`. USV observations remain in `Contact.bearing_rays`
and continue through the engine's internal GroupManager path; they are omitted
from the public `OperationalFrame.bearing_rays` because their public origin is
not available. An unknown observer is never reclassified as a UUV and never
gets a target-derived fallback position.

No truth fields or target positions are added to the operational snapshot or
frame contract. `domain/models.py` remains unchanged. This preserves the
truth-safe boundary and keeps the change compatible with existing consumers.

## Data Flow

1. Explicit platform-core sonar creates `PassiveSonarObservation` for deployed
   UUVs and USVs.
2. `SimulationEngine` adapts those observations to `BearingObservation` and
   passes all supported group-member bearings, including USV bearings, to
   `GroupManager.invoke`.
3. The engine stores those bearings in contact state for the carrier snapshot.
4. `frame_builder` resolves ray origins only from the snapshot's public UUV
   index and filters unsupported observers before constructing strict UI views.
5. The resulting frame is safe for Pydantic validation, JSONL logging, and hub
   publication.

## Error Handling and Tests

The builder treats a missing public observer origin as an unavailable public
projection, not as an exception and not as a guessed UUV. Tests cover direct
builder filtering, JSONL/WebSocket publisher behavior through the same frame
contract, and explicit engine integration showing that USV observation IDs
still contribute to the resulting belief.
