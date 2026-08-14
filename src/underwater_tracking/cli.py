# src/underwater_tracking/cli.py
import argparse

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--steps", type=int, required=True)
    simulate.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    config = load_app_config(args.config)
    engine = SimulationEngine(config, seed=args.seed)
    for _ in range(args.steps):
        engine.step()
    return 0
