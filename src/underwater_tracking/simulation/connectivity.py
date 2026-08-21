from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot
from typing import Literal

from underwater_tracking.domain.platforms import CommunicationLink, PlatformKind


@dataclass(frozen=True, slots=True)
class ConnectivityNode:
    platform_id: str
    kind: PlatformKind
    position_xy: tuple[float, float]
    surface_range_m: float
    acoustic_range_m: float


@dataclass(frozen=True, slots=True)
class ConnectivitySnapshot:
    links: tuple[CommunicationLink, ...]


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def build_connectivity(
    *,
    carrier_id: str,
    carrier_xy: tuple[float, float],
    nodes: tuple[ConnectivityNode, ...],
    carrier_positions: Mapping[str, tuple[float, float]] | None = None,
    carrier_ranges_m: Mapping[str, float] | None = None,
) -> ConnectivitySnapshot:
    """Build the carrier/platform communication graph.

    The original single-carrier arguments remain the compatibility path. A
    UUV-only run can additionally provide every carrier position and support
    radius so mother-ship-to-UUV acoustic links and the fleet surface mesh are
    represented in the same graph used by runtime health projections.
    """
    links: list[CommunicationLink] = []
    ordered = tuple(sorted(nodes, key=lambda node: node.platform_id))
    if carrier_positions is None:
        _append_legacy_carrier_links(links, carrier_id, carrier_xy, ordered)
    else:
        positions = dict(carrier_positions)
        positions.setdefault(carrier_id, carrier_xy)
        ranges = dict(carrier_ranges_m or {})
        for source_id in sorted(positions):
            source_xy = positions[source_id]
            source_range = ranges.get(source_id, float("inf"))
            for node in ordered:
                distance = _distance(source_xy, node.position_xy)
                if node.kind is PlatformKind.USV:
                    medium: Literal["surface", "acoustic"] = "surface"
                    limit = min(source_range, node.surface_range_m)
                else:
                    medium = "acoustic"
                    limit = min(source_range, node.acoustic_range_m)
                if distance <= limit:
                    links.append(
                        CommunicationLink(
                            source_id=source_id,
                            target_id=node.platform_id,
                            medium=medium,
                            distance_m=distance,
                        )
                    )
        ordered_carriers = tuple(sorted(positions))
        for index, left_id in enumerate(ordered_carriers):
            for right_id in ordered_carriers[index + 1 :]:
                distance = _distance(positions[left_id], positions[right_id])
                if distance <= min(
                    ranges.get(left_id, float("inf")),
                    ranges.get(right_id, float("inf")),
                ):
                    links.append(
                        CommunicationLink(
                            source_id=left_id,
                            target_id=right_id,
                            medium="surface",
                            distance_m=distance,
                        )
                    )
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            distance = _distance(left.position_xy, right.position_xy)
            if left.kind is PlatformKind.USV and right.kind is PlatformKind.USV:
                medium: Literal["surface", "acoustic"] = "surface"
                limit = min(left.surface_range_m, right.surface_range_m)
            else:
                medium = "acoustic"
                limit = min(left.acoustic_range_m, right.acoustic_range_m)
            if distance <= limit:
                links.append(
                    CommunicationLink(
                        source_id=left.platform_id,
                        target_id=right.platform_id,
                        medium=medium,
                        distance_m=distance,
                    )
                )
    return ConnectivitySnapshot(
        links=tuple(sorted(links, key=lambda link: (link.source_id, link.target_id)))
    )


def _append_legacy_carrier_links(
    links: list[CommunicationLink],
    carrier_id: str,
    carrier_xy: tuple[float, float],
    nodes: tuple[ConnectivityNode, ...],
) -> None:
    """Preserve the original single-carrier/USV connectivity contract."""
    for node in nodes:
        if node.kind is PlatformKind.USV:
            distance = _distance(carrier_xy, node.position_xy)
            if distance <= node.surface_range_m:
                links.append(
                    CommunicationLink(
                        source_id=carrier_id,
                        target_id=node.platform_id,
                        medium="surface",
                        distance_m=distance,
                    )
                )


def has_path(snapshot: ConnectivitySnapshot, source_id: str, target_id: str) -> bool:
    if source_id == target_id:
        return True
    adjacency: dict[str, set[str]] = {}
    for link in snapshot.links:
        adjacency.setdefault(link.source_id, set()).add(link.target_id)
        adjacency.setdefault(link.target_id, set()).add(link.source_id)
    queue = deque([source_id])
    visited = {source_id}
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor == target_id:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
