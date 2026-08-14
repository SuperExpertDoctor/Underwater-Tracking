# Underwater Tracking

Deterministic multi-UUV bearing-only tracking foundation: a headless
simulation loop that runs group tracking, prediction, planning, and
allocation with no ground-truth leakage into operational frames. Same
seed, same run, byte for byte.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Requires Python 3.11 or 3.12.

## Simulate

Run the deterministic default scenario for 360 steps at seed 42:

```powershell
python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42
```

## Output

Each run writes one JSONL line per simulation step to
`outputs/run-<seed>-<id>/frames.jsonl` (the `outputs/` directory is
git-ignored). Re-running with the same seed and configuration produces a
byte-identical log (run id and output path aside). Operational frames
never contain ground-truth fields; truth is available only through the
engine's evaluation sink.

## Test

```powershell
python -m pytest -q
```

## Verification

The foundation gates must all exit 0:

```powershell
python -m pytest tests/config tests/domain tests/simulation tests/tracking tests/prediction tests/planning tests/groups tests/integration tests/property -q
python -m ruff check src/underwater_tracking tests
python -m mypy src/underwater_tracking
python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42
```
