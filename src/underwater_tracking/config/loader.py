# src/underwater_tracking/config/loader.py
"""Configuration loading.

``load_app_config`` keeps its original contract unchanged: it validates the
scenario YAML plus the sibling ``tracking.yaml`` under the same config root.
Additively, when ``agent.yaml`` / ``llm.yaml`` exist next to
``tracking.yaml`` they are parsed into ``AppConfig.agent`` /
``AppConfig.llm``; when either file is absent the corresponding field stays
``None``, so existing callers, fixtures and tests are unaffected.
"""
from pathlib import Path
import yaml  # type: ignore[import-untyped]
from underwater_tracking.config.models import AppConfig

# Optional sections merged into the app config, keyed by the YAML filename
# next to ``tracking.yaml`` that provides them.
_OPTIONAL_SECTIONS: tuple[tuple[str, str], ...] = (
    ("agent", "agent.yaml"),
    ("llm", "llm.yaml"),
)


def load_app_config(path: str | Path) -> AppConfig:
    scenario_path = Path(path)
    config_root = scenario_path.parents[1]
    scenario_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    tracking_path = config_root / "tracking.yaml"
    tracking_data = yaml.safe_load(tracking_path.read_text(encoding="utf-8"))
    data: dict[str, object] = {**scenario_data, "tracking": tracking_data}
    for section, filename in _OPTIONAL_SECTIONS:
        section_path = config_root / filename
        if section_path.exists():
            data[section] = yaml.safe_load(section_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
