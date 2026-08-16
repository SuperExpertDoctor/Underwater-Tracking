# Remediation Task 4 report

## Scope

- Reworked the command-center Playwright fixture so the carrier, a returning
  `UUV-1`, and their recovery link are inside the same map bounds and use
  matching relationship IDs.
- Retained the assignment preview/apply and next-LangGraph-replan path with a
  separate deployable `UUV-2`, and retained row selection, detail drawer, and
  replay coverage.
- Verified the four scene images through their actual browser requests (HTTP
  200), then separately forced each one to HTTP 404 while the other three
  continued to return HTTP 200. Each mixed asset/vector case has its own
  canvas snapshot and asserts no application console errors.
- Added a carrierless, deployment-state-less legacy JSONL fixture. Replay and
  the `/api/replay` response both normalize it for frontend consumers:
  `UUV-legacy-deployed` becomes `deployed`; the legacy `returning` status
  becomes `returning`; `carrier` remains `null`.

## TDD evidence

- RED: the new backend compatibility test failed against the absent fixture
  (`IndexError` from an empty replay), and both new canvas assertions failed
  because their named snapshots did not yet exist.
- GREEN: added only the replay fixture and Playwright snapshot fixtures; no
  Canvas or backend production code changed.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/api/test_frame_pipeline.py tests/api/test_app.py`
  - 36 passed (one existing FastAPI/TestClient deprecation warning).
- `npm --prefix src/underwater_tracking/ui run test:e2e -- --workers=1`
  - 5 passed.
- `npm --prefix src/underwater_tracking/ui run test`
  - 13 files, 34 tests passed.
- `npm --prefix src/underwater_tracking/ui run build`
  - passed (`tsc --noEmit` and Vite production build).
- Visually inspected all five 1440×900 canvas snapshots: the asset path
  includes the carrier/returning-UUV/recovery-link region; each single-404
  path displays its mixed vector/background fallback.

## Important-review remediation

- Replaced the single wildcard all-assets-404 route with four parameterized
  browser cases: `background`, `carrier`, `uuv`, and `submarine`.
- Every case intercepts only its named scene asset with HTTP 404 and asserts
  the other three browser image requests return HTTP 200. The strict
  application-console-error assertion and visible carrier assertion remain.
- Added four 1440×900 mixed-state canvas snapshots and removed the obsolete
  all-assets-404 snapshot. Visual inspection confirms each named resource
  falls back independently while the other scene assets remain visible.
- Fresh focused E2E verification: `npm --prefix src/underwater_tracking/ui run
  test:e2e -- --workers=1 tests/e2e/command-center.spec.ts` — 5 passed.

## Scope control

- Production Canvas and backend code were left unchanged: the existing
  compatibility implementation satisfied the new regression tests.
- Pre-existing `.superpowers/sdd/remediation-task-2-report.md` modifications
  and `src/underwater_tracking.egg-info/` remain uncommitted and are excluded
  from this task's commit.
