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
        self._dedicated_by_target: dict[str, set[str]] = {}
        self._dedicated_by_uuv: dict[str, str] = {}

    def reserve(self, uuv_ids: Iterable[str], target_id: str) -> None:
        """Reserve every named UUV for ``target_id`` (idempotent per target)."""
        for uuv_id in uuv_ids:
            current = self._by_uuv.get(uuv_id)
            if current is not None and current != target_id:
                raise ValueError(
                    f"uuv {uuv_id!r} is already reserved for {current!r}"
                )
            dedicated_target = self._dedicated_by_uuv.get(uuv_id)
            if dedicated_target is not None and dedicated_target != target_id:
                raise ValueError(
                    f"uuv {uuv_id!r} is already dedicated to {dedicated_target!r}"
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

    def dedicate(self, uuv_ids: Iterable[str], target_id: str) -> None:
        """Reserve a frozen tracking group and project it as dedicated mode."""
        selected = set(uuv_ids)
        for uuv_id in selected:
            ordinary_target = self._by_uuv.get(uuv_id)
            if ordinary_target is not None and ordinary_target != target_id:
                raise ValueError(
                    f"uuv {uuv_id!r} is already reserved for {ordinary_target!r}"
                )
            dedicated_target = self._dedicated_by_uuv.get(uuv_id)
            if dedicated_target is not None and dedicated_target != target_id:
                raise ValueError(
                    f"uuv {uuv_id!r} is already dedicated to {dedicated_target!r}"
                )
        for uuv_id in self._dedicated_by_target.get(target_id, set()) - selected:
            del self._dedicated_by_uuv[uuv_id]
        self._dedicated_by_target[target_id] = selected
        for uuv_id in selected:
            self._dedicated_by_uuv[uuv_id] = target_id

    def release_dedicated(self, target_id: str) -> None:
        """Remove a target's dedicated-mode projection and free its members."""
        for uuv_id in self._dedicated_by_target.pop(target_id, ()):
            self._dedicated_by_uuv.pop(uuv_id, None)

    def reserved_uuvs(self) -> frozenset[str]:
        """The frozen projection every consumer reads (never mutated)."""
        return frozenset(self._by_uuv) | frozenset(self._dedicated_by_uuv)

    def reserved_for(self, target_id: str) -> frozenset[str]:
        """Every UUV currently reserved for ``target_id``."""
        return frozenset(self._by_target.get(target_id, ())) | frozenset(
            self._dedicated_by_target.get(target_id, ())
        )

    def dedicated_for(self, target_id: str) -> frozenset[str]:
        """The frozen members currently authorized for dedicated tracking."""
        return frozenset(self._dedicated_by_target.get(target_id, ()))

    def is_reserved(self, uuv_id: str) -> bool:
        return uuv_id in self._by_uuv or uuv_id in self._dedicated_by_uuv

    def items(self) -> list[tuple[str, tuple[str, ...]]]:
        """Deterministic ``(target_id, sorted uuv_ids)`` rows for the engine.

        ``SimulationEngine.set_reservations`` consumes the registry as a
        ``Mapping[str, Sequence[str]]`` duck type through ``items()``; the
        rows are sorted so the engine's reserved set is order-independent.
        """
        target_ids = set(self._by_target) | set(self._dedicated_by_target)
        return [
            (target_id, tuple(sorted(self.reserved_for(target_id))))
            for target_id in sorted(target_ids)
        ]

    def dedicated_items(self) -> list[tuple[str, tuple[str, ...]]]:
        """Deterministic target-group projection consumed by UUV-only execution."""
        return [
            (target_id, tuple(sorted(uuv_ids)))
            for target_id, uuv_ids in sorted(self._dedicated_by_target.items())
        ]
