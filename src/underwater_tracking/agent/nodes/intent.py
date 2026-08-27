# src/underwater_tracking/agent/nodes/intent.py
"""Intent analysis node (spec 12.2, 8.1).

``IntentAnalysisNode`` builds the intent LLM payload from estimated data
only: the downsampled estimated trajectory (``sampled_belief_history``),
deterministic motion features from the foundation
``extract_motion_features`` (``trajectory_features``), a derived maneuver
summary (loiter segments, persistent maneuvers, suspected evasion),
recent belief uncertainty and observation quality, and any prior intent
hypotheses for the target. The snapshot's hidden truth never enters the
payload; only the fields the prompt may use are curated.

``__call__`` loops over every target present in the snapshot's group
reports and invokes ``IntentHypothesis`` per target, attaching model and
prompt versions plus the canonical request/response hashes to the state's
``llm_provenance`` (spec 16). All behavior is a pure function of the node's
inputs and the injected history provider: no randomness, no wall clock.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from underwater_tracking.agent.llm import (
    LLMCallMetadata,
    LLMContentError,
    StructuredLLM,
)
from underwater_tracking.agent.nodes.strategy import _content_error_feedback
from underwater_tracking.agent.prompts import (
    INTENT_PROMPT_VERSION,
    INTENT_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import IntentHypothesis, TrajectoryDiffResult
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
)
from underwater_tracking.prediction.features import extract_motion_features

# One downsampled estimated fix: (sim_time_s, x_m, y_m).
BeliefSample = tuple[int, float, float]
# Resolves the per-target estimated trajectory for a snapshot.
BeliefHistoryProvider = Callable[[SituationSnapshot, str], Sequence[BeliefSample]]
# Resolves the immutable SituationSnapshot from its storage reference.
SnapshotProvider = Callable[[str], SituationSnapshot]

# Deterministic maneuver heuristics over the extracted features.
_LOITER_FRACTION_MIN = 0.50
_PERSISTENT_HEADING_CHANGE_RAD = 2.0
_EVASION_DWELL_FRACTION_MIN = 0.30
_EVASION_CURVATURE_Q75_MIN = 0.005


class IntentAnalysisNode:
    """Semantic intent-analysis node (LangGraph node: state in, state out).

    ``build_payload`` is a pure payload builder (spec 12.2 input list);
    ``__call__`` runs the full analysis loop for every snapshot target.
    """

    def __init__(
        self,
        llm: StructuredLLM[IntentHypothesis],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = INTENT_PROMPT_VERSION,
        temperature: float = 0.2,
        max_samples: int = 40,
        belief_history: BeliefHistoryProvider | None = None,
        snapshot_provider: SnapshotProvider | None = None,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._max_samples = max_samples
        self._belief_history = belief_history
        self._snapshot_provider = snapshot_provider

    def build_payload(
        self,
        snapshot: SituationSnapshot,
        target_id: str,
        *,
        belief_history: Sequence[BeliefSample] | None = None,
        prior_hypotheses: Mapping[str, IntentHypothesis] | None = None,
        trajectory_diff: TrajectoryDiffResult | None = None,
        additional_evidence_ids: Sequence[str] = (),
    ) -> dict[str, object]:
        """Curated intent payload: estimated history and features only.

        Explicit ``belief_history`` overrides the injected provider; at
        least three increasing-time fixes are required to derive motion
        features. IDs are sorted; only the fields the prompt may use are
        serialized; the raw snapshot is never exposed to the LLM.
        """
        report = self._group_report(snapshot, target_id)
        samples = self._resolve_history(snapshot, target_id, belief_history)
        sampled = self._downsample(samples)
        features = extract_motion_features(
            np.asarray([sample[0] for sample in samples], dtype=float),
            np.asarray([[sample[1], sample[2]] for sample in samples], dtype=float),
        )
        evidence_ids = set(_intent_evidence_ids(snapshot, target_id))
        evidence_ids.update(additional_evidence_ids)
        payload: dict[str, object] = {
            "model": self._model_id,
            "temperature": self._temperature,
            "system_prompt": INTENT_SYSTEM_PROMPT,
            "scenario_id": snapshot.scenario_id,
            "sim_time_s": snapshot.sim_time_s,
            "target_id": target_id,
            "sampled_belief_history": [
                {"sim_time_s": t, "x": x, "y": y} for t, x, y in sampled
            ],
            "trajectory_history_source": "estimated_belief",
            "trajectory_features": features,
            "maneuver_summary": self._maneuver_summary(features),
            "belief_uncertainty": self._belief_uncertainty(report.belief),
            "observation_quality": self._observation_quality(report.quality),
            # Key areas are not part of the snapshot contract yet; kept as
            # an explicit empty list so the spec 12.2 input slot is visible.
            "key_area_proximity": (),
            "prior_intent_hypotheses": self._prior_intent_hypotheses(
                prior_hypotheses, target_id
            ),
            "evidence_ids": sorted(evidence_ids),
        }
        if trajectory_diff is not None:
            payload["trajectory_diff"] = self._trajectory_diff_payload(trajectory_diff)
        return payload

    def __call__(self, state: CarrierState) -> CarrierState:
        """Analyze every snapshot target and attach provenance to state."""
        snapshot = self._resolve_snapshot(state)
        snapshot_target_ids = tuple(
            dict.fromkeys(report.target_id for report in snapshot.group_reports)
        )
        target_filter = state.get("intent_target_ids")
        target_ids = (
            snapshot_target_ids
            if target_filter is None
            else tuple(
                target_id
                for target_id in snapshot_target_ids
                if target_id in set(target_filter)
            )
        )
        hypotheses: dict[str, IntentHypothesis] = {}
        provenance: dict[str, LLMCallMetadata] = {}
        for target_id in target_ids:
            diff = (state.get("prediction_diffs") or {}).get(target_id)
            gate = (state.get("prediction_diff_gates") or {}).get(target_id)
            additional_evidence_ids = tuple(
                evidence_id
                for evidence_id in (
                    None if diff is None else diff.diff_id,
                    None if gate is None else gate.suspicion_event_id,
                )
                if evidence_id is not None
            )
            payload = self.build_payload(
                snapshot,
                target_id,
                prior_hypotheses=state.get("intent_hypotheses"),
                trajectory_diff=diff,
                additional_evidence_ids=additional_evidence_ids,
            )
            hypothesis = self._invoke_intent(payload)
            hypotheses[target_id] = hypothesis
            provenance[f"intent:{target_id}"] = LLMCallMetadata(
                operation="intent",
                model=self._model_id,
                prompt_version=self._prompt_version,
                request_hash=canonical_digest(payload),
                response_hash=canonical_digest(hypothesis.model_dump(mode="json")),
                sim_time_s=snapshot.sim_time_s,
                scenario_id=snapshot.scenario_id,
            )
        return {
            "intent_hypotheses": {
                **state.get("intent_hypotheses", {}),
                **hypotheses,
            },
            "llm_provenance": {**state.get("llm_provenance", {}), **provenance},
        }

    @staticmethod
    def _trajectory_diff_payload(diff: TrajectoryDiffResult) -> dict[str, object]:
        return {
            "diff_id": diff.diff_id,
            "previous_prediction_id": diff.previous_prediction_id,
            "current_prediction_id": diff.current_prediction_id,
            "status": diff.status,
            "absolute_rms_m": diff.absolute_rms_m,
            "normalized_rms": diff.normalized_rms,
            "p90_distance_m": diff.p90_distance_m,
            "max_distance_m": diff.max_distance_m,
            "js_distance": diff.js_distance,
            "previous_leading_model": diff.previous_leading_model,
            "current_leading_model": diff.current_leading_model,
            "absolute_floor_m": diff.absolute_floor_m,
            "normalized_threshold": diff.normalized_threshold,
            "observation_ids": list(diff.current_evidence_ids),
        }

    def _invoke_intent(self, payload: dict[str, object]) -> IntentHypothesis:
        """One structured intent call; on schema failure, exactly ONE re-ask.

        Mirroring the strategy node's bounded correction (spec 8.3 content
        path), the detailed validation errors are appended as
        ``correction_feedback`` and the model answers exactly once more; a
        second content failure is a hard error — never an unbounded loop.
        Transport and config errors are untouched (the port retries those
        internally against its own budget).
        """
        try:
            hypothesis = self._llm.invoke_structured(
                "intent",
                payload,
                IntentHypothesis,
                prompt_version=self._prompt_version,
            )
            _validate_intent_evidence(hypothesis, payload)
            return hypothesis
        except LLMContentError as exc:
            hypothesis = self._llm.invoke_structured(
                "intent",
                {**payload, "correction_feedback": _content_error_feedback(exc)},
                IntentHypothesis,
                prompt_version=self._prompt_version,
            )
            _validate_intent_evidence(hypothesis, payload)
            return hypothesis

    def _resolve_snapshot(self, state: CarrierState) -> SituationSnapshot:
        provider = self._snapshot_provider
        if provider is None:
            raise ValueError("intent_analysis requires a snapshot provider")
        snapshot_ref = state.get("snapshot_ref")
        if not snapshot_ref:
            raise ValueError("intent_analysis requires snapshot_ref in state")
        return provider(snapshot_ref)

    def _resolve_history(
        self,
        snapshot: SituationSnapshot,
        target_id: str,
        explicit: Sequence[BeliefSample] | None,
    ) -> Sequence[BeliefSample]:
        if explicit is not None:
            samples = explicit
        elif self._belief_history is not None:
            samples = self._belief_history(snapshot, target_id)
        else:
            samples = None
        if samples is None or len(samples) < 3:
            raise ValueError(
                f"insufficient estimated trajectory history for target {target_id!r}"
            )
        return samples

    def _downsample(self, samples: Sequence[BeliefSample]) -> Sequence[BeliefSample]:
        """Uniformly downsample to at most ``max_samples`` fixes, keeping the last."""
        if len(samples) <= self._max_samples:
            return samples
        step = math.ceil(len(samples) / self._max_samples)
        downsampled = list(samples[::step])
        if downsampled[-1] != samples[-1]:
            downsampled.append(samples[-1])
        return downsampled

    def _maneuver_summary(self, features: Mapping[str, float]) -> dict[str, bool | float]:
        return {
            "loiter_segment": features["dwell_fraction"] >= _LOITER_FRACTION_MIN,
            "persistent_maneuver": (
                features["heading_change_rad"] >= _PERSISTENT_HEADING_CHANGE_RAD
            ),
            "suspected_evasion": (
                features["dwell_fraction"] >= _EVASION_DWELL_FRACTION_MIN
                and features["curvature_q75"] >= _EVASION_CURVATURE_Q75_MIN
            ),
            "dwell_fraction": features["dwell_fraction"],
        }

    def _belief_uncertainty(self, belief: TargetBelief) -> dict[str, float]:
        covariance = belief.covariance
        position_variance = sum(
            covariance[i][i] for i in range(min(len(covariance), 2))
        )
        return {
            "position_std_m": math.sqrt(max(position_variance, 0.0)),
            "fim_min_eigenvalue": belief.fim_min_eigenvalue,
            "fim_condition": belief.fim_condition,
        }

    def _observation_quality(self, quality: GroupQuality) -> dict[str, object]:
        return {
            "instant": quality.instant,
            "window_mean": quality.window_mean,
            "ewma": quality.ewma,
            "hard_guard_reasons": list(quality.hard_guard_reasons),
        }

    def _prior_intent_hypotheses(
        self,
        prior: Mapping[str, IntentHypothesis] | None,
        target_id: str,
    ) -> list[dict[str, object]]:
        if prior is None:
            return []
        hypothesis = prior.get(target_id)
        if hypothesis is None:
            return []
        return [
            {
                "label": hypothesis.label,
                "confidence": hypothesis.confidence,
                "evidence_ids": sorted(hypothesis.evidence_ids),
                "prompt_version": hypothesis.prompt_version,
            }
        ]

    def _group_report(self, snapshot: SituationSnapshot, target_id: str) -> GroupReport:
        for report in snapshot.group_reports:
            if report.target_id == target_id:
                return report
        raise ValueError(f"no group report for target {target_id!r}")


def _intent_evidence_ids(snapshot: SituationSnapshot, target_id: str) -> tuple[str, ...]:
    """Return source evidence, with a deterministic estimate for cold start."""
    report = next(
        report for report in snapshot.group_reports if report.target_id == target_id
    )
    evidence_ids = set(report.belief.source_observation_ids)
    if not evidence_ids:
        evidence_ids.add(
            f"estimate:{snapshot.scenario_id}:{target_id}:{snapshot.sim_time_s}"
        )
    return tuple(sorted(evidence_ids))


def _validate_intent_evidence(
    hypothesis: IntentHypothesis,
    payload: Mapping[str, object],
) -> None:
    """Keep provider evidence references inside the curated request boundary."""
    supplied = {
        str(evidence_id)
        for evidence_id in (payload.get("evidence_ids") or ())
    }
    unknown = set(hypothesis.evidence_ids) - supplied
    if unknown:
        raise LLMContentError(
            f"intent response cites evidence outside the request: {sorted(unknown)}"
        )
