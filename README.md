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

## Agent run

Run the same scenario through the resilient LangGraph tracking assistant
(agent coupling with degradation handling):

```powershell
python -m underwater_tracking.cli agent-run --config configs/scenario/default.yaml --steps 540 --seed 42
```

The carrier keeps one SQLite database per run (plan and event repositories,
checkpointer, decision ledger). Group reports drive a carrier cycle every
observation step; committed plan commands flow back to the group manager.
The run finishes with `manifest.json` (carrier errors, decision count, and
the active plan record) plus `frames.jsonl`. The provider is the configured
LongCat-compatible HTTP endpoint; the API key is read from the configured
environment variable or git-ignored config.

## Command center

Start the LangGraph runtime and FastAPI/WebSocket transport in one process:

```powershell
python -m underwater_tracking.cli serve --config configs/scenario/default.yaml --seed 42
```

In a second shell, start the local React console. Its Vite development proxy
forwards `/api` and `/ws` to port 8000:

```powershell
npm --prefix src/underwater_tracking/ui run dev
```

### Carrier scene and deployment status

Every newly emitted operational frame includes the carrier view used by the
command-center map and the `载体舰 / 发送回收` sidebar card. The four source
images remain in the repository-root `assets/` directory:
`背景图.png`, `舰艇.png`, `UUV.png`, and `潜艇.png`. The browser serves their
stable copies from `/assets/scene/background.png`, `/assets/scene/carrier.png`,
`/assets/scene/uuv.png`, and `/assets/scene/submarine.png` respectively.

`onboard`, `deployed`, `returning`, and `failed` are backend-owned UUV
deployment states. The UI displays these frame values; it does not derive or
override deployment state. Older JSONL replays that predate carrier data remain
usable and show the compatible `等待载体态势` empty card.

Human assignment remains dynamic: assignment and recovery changes are carried
in subsequent frames, and the next LangGraph planning round re-plans from that
updated operational state rather than a separate UI scheduler.

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
