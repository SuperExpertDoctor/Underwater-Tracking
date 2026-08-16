from underwater_tracking.domain.platforms import PlatformKind
from underwater_tracking.simulation.connectivity import (
    ConnectivityNode,
    build_connectivity,
    has_path,
)


def node(
    platform_id: str,
    kind: PlatformKind,
    position_xy: tuple[float, float],
    surface_range_m: float,
    acoustic_range_m: float,
) -> ConnectivityNode:
    return ConnectivityNode(
        platform_id=platform_id,
        kind=kind,
        position_xy=position_xy,
        surface_range_m=surface_range_m,
        acoustic_range_m=acoustic_range_m,
    )


def test_usv_mesh_connects_carrier_to_uuv_over_multiple_hops() -> None:
    snapshot = build_connectivity(
        carrier_id="carrier_01",
        carrier_xy=(0.0, 0.0),
        nodes=(
            node("usv_00", PlatformKind.USV, (5000.0, 0.0), 6000.0, 2500.0),
            node("usv_01", PlatformKind.USV, (10000.0, 0.0), 6000.0, 2500.0),
            node("uuv_11", PlatformKind.UUV, (11000.0, 1000.0), 1.0, 2500.0),
        ),
    )

    assert has_path(snapshot, "carrier_01", "uuv_11")
    assert [(link.source_id, link.target_id, link.medium) for link in snapshot.links] == [
        ("carrier_01", "usv_00", "surface"),
        ("usv_00", "usv_01", "surface"),
        ("usv_01", "uuv_11", "acoustic"),
    ]


def test_distance_break_disconnects_group_leader() -> None:
    snapshot = build_connectivity(
        carrier_id="carrier_01",
        carrier_xy=(0.0, 0.0),
        nodes=(
            node("usv_00", PlatformKind.USV, (7000.0, 0.0), 6000.0, 2000.0),
            node("uuv_11", PlatformKind.UUV, (7000.0, 1000.0), 1.0, 2000.0),
        ),
    )

    assert not has_path(snapshot, "carrier_01", "uuv_11")
    assert [(link.source_id, link.target_id, link.medium) for link in snapshot.links] == [
        ("usv_00", "uuv_11", "acoustic")
    ]


def test_uuv_to_uuv_uses_acoustic_range() -> None:
    snapshot = build_connectivity(
        carrier_id="carrier_01",
        carrier_xy=(10000.0, 10000.0),
        nodes=(
            node("uuv_10", PlatformKind.UUV, (0.0, 0.0), 1.0, 1500.0),
            node("uuv_11", PlatformKind.UUV, (1000.0, 0.0), 1.0, 1500.0),
        ),
    )

    assert has_path(snapshot, "uuv_10", "uuv_11")
