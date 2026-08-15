# src/underwater_tracking/config/loader.py
"""Configuration loading.

``load_app_config`` keeps its original contract unchanged: it validates the
scenario YAML plus the sibling ``tracking.yaml`` under the same config root.
Additively, when ``agent.yaml`` / ``llm.yaml`` exist next to
``tracking.yaml`` they are parsed into ``AppConfig.agent`` /
``AppConfig.llm``; when either file is absent the corresponding field stays
``None``, so existing callers, fixtures and tests are unaffected.

``configs/.env`` (git-ignored, next to ``tracking.yaml``) is resolved into
the environment: ``KEY=VALUE`` lines populate ``os.environ`` via
``setdefault``, so a real environment variable always wins over the file.
After validation, an ``llm`` section without an explicit ``api_key`` gets
one resolved from ``os.environ[api_key_env]`` when available.
"""
import os
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
    _load_env_file(config_root)
    scenario_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    tracking_path = config_root / "tracking.yaml"
    tracking_data = yaml.safe_load(tracking_path.read_text(encoding="utf-8"))
    data: dict[str, object] = {**scenario_data, "tracking": tracking_data}
    for section, filename in _OPTIONAL_SECTIONS:
        section_path = config_root / filename
        if section_path.exists():
            data[section] = yaml.safe_load(section_path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(data)
    if config.llm is not None and config.llm.api_key is None:
        key = os.environ.get(config.llm.api_key_env)
        if key is not None:
            config = config.model_copy(
                update={"llm": config.llm.model_copy(update={"api_key": key})}
            )
    return config


def _load_env_file(config_root: Path) -> None:
    """Resolve ``configs/.env`` into the environment (env vars always win).

    ``KEY=VALUE`` lines are parsed (blank lines and ``#`` comments are
    ignored) and applied with ``os.environ.setdefault``, so a real
    environment variable always overrides the file.
    """
    env_path = config_root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key:
            os.environ.setdefault(key, value.strip())
