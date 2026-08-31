# Runtime Supervision And Live Progress Design

## Goal

Make `python main.py` stop all processes it owns when interrupted, and keep
the live command center advancing by default when initial planning is degraded.

## Runtime Ownership

`main.py` will launch the CLI backend as a child process rather than a daemon
thread. The parent owns both the backend process tree and the Vite process tree.
On normal exit, startup failure, SIGINT, or SIGTERM, it will stop the backend
tree and the Vite tree before returning. The backend retains its normal
cooperative shutdown path when it receives an interrupt directly.

The supervisor will use a bounded, platform-specific child-tree termination
helper. Windows uses `taskkill /T /F`; POSIX uses the session process group.
The helper tolerates a child that has already exited.

## Live Progress

`--steps 0` continues to mean an unbounded interactive run. It must not imply
bootstrap planning. Only `--bootstrap-planning` freezes physics until the first
planning epoch commits. If bootstrap is explicitly requested and fails, the
existing `awaiting_retry` behavior remains available.

Without explicit bootstrap, the simulation starts immediately and planning
errors remain visible through the existing health and operational-frame state;
they do not pin the simulation at time zero.

## Tests

Tests will verify that:

- `main.py` starts and shuts down its backend and Vite children on interrupt.
- Default zero-step runs do not enter bootstrap planning.
- Explicit bootstrap planning still waits and exposes retry state after failure.

Focused Python tests will run first, followed by the relevant test groups and
the full Python suite before the implementation commit.

## Scope

This change does not alter regional planning geometry or resource allocation
algorithms. Their failures are surfaced as degraded planning while the default
live simulation continues. Correcting those independent planning constraints is
separate from lifecycle and UI-progress behavior.
