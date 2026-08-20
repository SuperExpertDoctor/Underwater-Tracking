"""Bounded in-memory checkpoint storage for long-running group graphs."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class BoundedInMemorySaver(InMemorySaver):
    """Keep only the newest checkpoints and their dependent storage per thread."""

    def __init__(self, *, max_checkpoints: int = 2, **kwargs: Any) -> None:
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        super().__init__(**kwargs)
        self.max_checkpoints = max_checkpoints

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        result = super().put(config, checkpoint, metadata, new_versions)
        configurable = config["configurable"]
        self._prune_thread(
            configurable["thread_id"],
            configurable.get("checkpoint_ns", ""),
        )
        return result

    def put_writes(
        self,
        config: Any,
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        configurable = config["configurable"]
        self._prune_thread(
            configurable["thread_id"],
            configurable.get("checkpoint_ns", ""),
        )

    def _prune_thread(self, thread_id: str, checkpoint_ns: str | None = None) -> None:
        """Remove old checkpoint records, writes, and unreferenced blobs."""

        namespaces = self.storage.get(thread_id)
        if not namespaces:
            return
        selected_namespaces = (
            (checkpoint_ns,)
            if checkpoint_ns is not None
            else tuple(namespaces)
        )
        for namespace in selected_namespaces:
            checkpoints = namespaces.get(namespace)
            if not checkpoints:
                continue
            ordered_ids = list(checkpoints)
            retained_ids = set(ordered_ids[-self.max_checkpoints :])
            for checkpoint_id in ordered_ids:
                if checkpoint_id not in retained_ids:
                    del checkpoints[checkpoint_id]

            referenced_blobs: set[tuple[str, Any]] = set()
            for checkpoint_id in retained_ids:
                checkpoint_record = checkpoints.get(checkpoint_id)
                if checkpoint_record is None:
                    continue
                checkpoint = self.serde.loads_typed(checkpoint_record[0])
                referenced_blobs.update(
                    checkpoint.get("channel_versions", {}).items()
                )

            for write_key in list(self.writes):
                if (
                    write_key[0] == thread_id
                    and write_key[1] == namespace
                    and write_key[2] not in retained_ids
                ):
                    del self.writes[write_key]

            for blob_key in list(self.blobs):
                if (
                    blob_key[0] == thread_id
                    and blob_key[1] == namespace
                    and (blob_key[2], blob_key[3]) not in referenced_blobs
                ):
                    del self.blobs[blob_key]
