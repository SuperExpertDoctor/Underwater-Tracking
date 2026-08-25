"""Configuration loader for the rule-based event world model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from underwater_tracking.world_model.models import RuleWorldModelConfig


DEFAULT_WORLD_MODEL_CONFIG = RuleWorldModelConfig()


def load_world_model_config(path: str | Path) -> RuleWorldModelConfig:
    """Load and strictly validate one YAML rule configuration."""

    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("world-model config root must be a mapping")
    return RuleWorldModelConfig.model_validate(raw)
