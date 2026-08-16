# Remediation Task 2 report

## Scope

- Added the canonical Python deployability predicate and deterministic exclusion reasons.
- Enforced it in directives, optimizer/allocation input construction, commit validation,
  and active verification (pinger selection and in-position closure).
- Added carrier relationship disjointness and snapshot/frame membership validators.
- Normalized legacy payloads with omitted deployment and carrier relationship fields
  before validation, while preserving `carrier=None` compatibility.
- Added the browser mirror predicate and made manual assignment hide onboard and
  returning UUVs.

## TDD evidence

- New domain, directive, active-verification, optimizer/commit, frame, and UI tests
  were first run against the pre-change implementation and failed for the intended
  missing behaviors.
- The legacy carrier/frame normalization tests also failed before the compatibility
  normalizer was added.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/domain tests/api tests/agent --ignore=tests/agent/test_llm_port.py`
  - 145 passed, 56 skipped (one existing FastAPI deprecation warning).
- `npm --prefix src/underwater_tracking/ui test -- --run`
  - 12 files, 31 tests passed.
- `.venv/bin/python -m ruff check src tests`
  - passed.
- `.venv/bin/python -m mypy src`
  - passed: 80 source files.
- `npm --prefix src/underwater_tracking/ui run build`
  - passed; includes `tsc --noEmit`.

## Known environment baseline

`tests/agent/test_llm_port.py` requires `UNDERWATER_TRACKING_API_KEY`; without that
external credential the two config assertions fail. It is unrelated to Task 2 and
was excluded only from the focused verification command above. Project-wide
`mypy src tests` also has pre-existing strict test-annotation errors; production
`mypy src` passes.
