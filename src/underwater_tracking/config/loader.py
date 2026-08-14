# src/underwater_tracking/config/loader.py
from pathlib import Path
import yaml
from underwater_tracking.config.models import AppConfig


def load_app_config(path: str | Path) -> AppConfig:
    scenario_path = Path(path)
    scenario_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    tracking_path = scenario_path.parents[1] / "tracking.yaml"
    tracking_data = yaml.safe_load(tracking_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate({**scenario_data, "tracking": tracking_data})
