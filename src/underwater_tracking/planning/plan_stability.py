"""Deterministic change-cost evidence for rolling task-region planning."""

from __future__ import annotations

Bounds = tuple[float, float, float, float]


def rectangle_iou(left: Bounds, right: Bounds) -> float:
    """Return IoU for two ``(min_x, max_x, min_y, max_y)`` global rectangles."""
    intersection_width = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[2], right[2]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[1] - left[0]) * max(0.0, left[3] - left[2])
    right_area = max(0.0, right[1] - right[0]) * max(0.0, right[3] - right[2])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union
