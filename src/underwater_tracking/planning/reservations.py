# src/underwater_tracking/planning/reservations.py
"""Human-assignment reservation registry (spec 17.2, R4).

UUVs the operator has explicitly assigned to a target are reserved: the
allocator never assigns them elsewhere (``AllocationInput.reserved_uuv_ids``)
and the verification protocol never picks them as pingers. One registry is
owned by the CarrierRuntime and shared with the engine through
``SimulationEngine.set_reservations``; every consumer reads only the
immutable ``reserved_uuvs()`` projection, so the registry itself stays a
plain in-memory map.
"""

from __future__ import annotations

from collections.abc import Iterable


class ReservationRegistry:
    """In-memory map of uuv_id -> target_id with reverse lookup.

    A UUV is reserved for at most one target at a time; re-reserving the
    same UUV for a different target raises ValueError. The directive
    conflict validator rejects this earlier; the registry is the final
    guard.
    """

    def __init__(self) -> None:
        self._by_uuv: dict[str, str] = {}
        self._by_target: dict[str, set[str]] = {}

    def reserve(self, uuv_ids: Iterable[str], target_id: str) -> None:
        """Reserve every named UUV for ``target_id`` (idempotent per target)."""
        for uuv_id in uuv_ids:
            current = self._by_uuv.get(uuv_id)
            if current is not None and current != target_id:
                raise ValueError(
                    f"uuv {uuv_id!r} is already reserved for {current!r}"
                )
            self._by_uuv[uuv_id] = target_id
            self._by_target.setdefault(target_id, set()).add(uuv_id)

    def release(self, uuv_ids: Iterable[str]) -> None:
        """Release every named UUV (unknown ids are ignored)."""
        for uuv_id in uuv_ids:
            target_id = self._by_uuv.pop(uuv_id, None)
            if target_id is None:
                continue
            reserved = self._by_target.get(target_id)
            if reserved is not None:
                reserved.discard(uuv_id)
                if not reserved:
                    del self._by_target[target_id]

    def reserved_uuvs(self) -> frozenset[str]:
        """The frozen projection every consumer reads (never mutated)."""
        return frozenset(self._by_uuv)

    def reserved_for(self, target_id: str) -> frozenset[str]:
        """Every UUV currently reserved for ``target_id``."""
        return frozenset(self._by_target.get(target_id, ()))

    def is_reserved(self, uuv_id: str) -> bool:
        return uuv_id in self._by_uuv

    def items(self) -> list[tuple[str, tuple[str, ...]]]:
        """Deterministic ``(target_id, sorted uuv_ids)`` rows for the engine.

        ``SimulationEngine.set_reservations`` consumes the registry as a
        ``Mapping[str, Sequence[str]]`` duck type through ``items()``; the
        rows are sorted so the engine's reserved set is order-independent.
        """
        return [
            (target_id, tuple(sorted(uuv_ids)))
            for target_id, uuv_ids in sorted(self._by_target.items())
        ]
