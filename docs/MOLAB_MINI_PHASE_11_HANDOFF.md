# MoLab mini Phase 11 handoff

Date: 2026-08-20  
Scope: measured performance hardening and executable MoLab tuning matrix

## Outcome

The fixed eight-server catalog no longer starts every MCP stdio process tree at
once. Registration now uses two slots by default and rejects operational
overrides outside `1..4`. Individual registration and full configuration
initialization use the same bound. Tool calls remain ordered and unchanged.

A reusable `src.mini.benchmark` command records wall and monotonic duration,
CPU utilization samples, peak process-tree RSS, child-process counts before,
during, and after, plus NVML GPU utilization, memory, and temperature when a GPU
is present. Its artifacts omit prompts, scenarios, subprocess arguments, and
environment-variable contents.

## Retained before/after evidence

Both measurements used the same Python 3.12.13 environment, config hashes,
eight server names, five iterations, and real FastMCP servers. Each iteration
registered eight servers and 71 functions including the 16 lifecycle functions.

| Metric | Unbounded before | Bounded after |
|---|---:|---:|
| Registration slots | 8 effective | 2 |
| Median duration | 0.9220 s | 2.4220 s |
| p95 duration | 1.1566 s | 2.5098 s |
| Maximum sampled process-tree RSS | 1,817,395,200 B | 233,660,416 B |
| Peak descendant processes | 35 | 6 |
| Descendant processes after | 0 | 0 |

The bound reduced maximum sampled RSS by 87.1% and the peak descendant count by
82.9%, at a 1.5000-second median startup cost. The fourth unbounded sample had
the largest transient peak; raw per-iteration JSON remains under the ignored
`artifacts/mini/benchmarks/` tree for local inspection. Export those JSON files
with run artifacts before ending a MoLab session.

Lazy registration was measured but not selected. A complete bounded cold start
remains below three seconds, while lazy registration would defer schema and
configuration failures into generation and complicate deterministic preflight.

The synthesis worker pool already reuses exactly one generator per worker. A
new regression now proves reuse across six seeds, resets state both before and
after every attempt, and confirms the final MCP client cleanup. The fixed mini
graph path batches independent teacher parameter classifications; the disabled
legacy LLM-edge path is outside this profile. Dataset conversion streams compact
JSON arrays into atomic same-directory replacements and remains byte-identical
on rerun.

`src.mini.benchmark generation-workers` now makes the next MoLab gate directly
executable. It runs identical deterministic seeds at 1, 2, and 4 workers,
creates an isolated synthesis run for each row, and records raw resource,
throughput, p50/p95 trajectory latency, retry, failure, and process-cleanup
evidence plus a Markdown decision table. The command retains a partial report
when synthesis returns a non-completed state and enforces the current 1..4
worker safety contract.

## Commands run

```text
.test-work/phase10-full/Scripts/python.exe -m pytest -q \
  tests/unit/test_core_correctness.py tests/unit/test_mcp_async_paths.py \
  tests/integration/test_mcp_async.py tests/integration/test_mcp_lifecycle.py
# 17 passed in 63.77s

.test-work/phase10-full/Scripts/python.exe -m src.mini.benchmark \
  catalog-registration --config configs/mini/pipeline.toml \
  --label baseline-unbounded --samples 5 \
  --output artifacts/mini/benchmarks/phase11-catalog-before.json
# median 0.9220s

.test-work/phase10-full/Scripts/python.exe -m pytest -q \
  --basetemp .test-work/pytest-phase11-focus \
  tests/unit/test_mini_benchmark.py tests/unit/test_mcp_registration_bound.py \
  tests/unit/test_core_correctness.py tests/unit/test_mcp_async_paths.py
# 16 passed in 10.00s

.test-work/phase10-full/Scripts/python.exe -m src.mini.benchmark \
  catalog-registration --config configs/mini/pipeline.toml \
  --label bounded-2 --samples 5 \
  --output artifacts/mini/benchmarks/phase11-catalog-after.json
# median 2.4220s

ruff check src/mini/benchmark.py src/manager/mcp_client_manager.py \
  tests/unit/test_mini_benchmark.py tests/unit/test_mcp_registration_bound.py
# All checks passed

.test-work/phase10-full/Scripts/python.exe -m compileall -q src tests
# passed

.test-work/phase10-full/Scripts/python.exe -m pytest -q \
  --basetemp .test-work/pytest-phase11-unit tests/unit
# 79 passed, 1 skipped in 10.87s

.test-work/phase10-full/Scripts/python.exe -m pytest -q \
  --basetemp .test-work/pytest-phase11-integration \
  tests/integration -m "not model"
# 19 passed, 1 deselected in 61.76s

.test-work/phase10-full/Scripts/python.exe -m pytest -q \
  --basetemp .test-work/pytest-phase11-model tests/integration -m model
# 1 skipped (local model endpoint absent), 19 deselected in 1.56s

.test-work/phase10-full/Scripts/python.exe -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase12-focus-verified \
  tests/unit/test_mini_benchmark.py tests/unit/test_user_provided_cache.py \
  tests/integration/test_mini_resume.py tests/unit/test_data_split.py
# 23 passed in 3.08s

ruff check src/mini/benchmark.py tests/unit/test_mini_benchmark.py \
  tests/unit/test_user_provided_cache.py tests/integration/test_mini_resume.py
# All checks passed

.test-work/phase10-full/Scripts/python.exe -m compileall -q src tests
# passed

.test-work/phase10-full/Scripts/python.exe -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase12-unit-verified tests/unit
# 87 passed, 1 skipped in 10.72s (existing POSIX-only skip)

.test-work/phase10-full/Scripts/python.exe -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase12-integration \
  tests/integration -m "not model"
# 20 passed, 1 deselected in 59.49s

.test-work/phase10-full/Scripts/python.exe -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase12-model tests/integration -m model
# 1 skipped (local model endpoint absent), 20 deselected in 1.46s
```

## Files changed

- `src/manager/mcp_client_manager.py`: bounded registration admission.
- `src/mini/benchmark.py`: resource measurement, catalog benchmark,
  identical-seed generation worker matrix, and report rendering CLI.
- `tests/unit/test_mcp_registration_bound.py`: exact concurrency regression.
- `tests/unit/test_mini_benchmark.py`: measurement, comparison, safe worker
  matrix, and report-rendering regressions.
- `tests/unit/test_user_provided_cache.py`: exact teacher classification batch
  boundary regression.
- `tests/integration/test_mini_resume.py`: worker-local reuse, state reset, and
  final MCP cleanup regression.
- `docs/MOLAB_MINI_IMPLEMENTATION_PLAN.md`: Phase 11 status, evidence, and
  decision record.
- `docs/MOLAB_MINI_RUNBOOK.md`: reproducible benchmark commands.

There is no artifact or schema migration. Existing ignored benchmark artifacts
can be deleted or exported independently of synthesis runs.

## Remaining work and risks

Phase 11 remains in progress. No GPU/model endpoint is available locally, so the
following have not been measured and their defaults have not been changed:

- generation worker sweeps;
- vLLM sequence-count, utilization, and 8K/16K context sweeps;
- Qwen3 4B/8B student throughput and executable-quality comparison.

The exact next task is to run the ten-trajectory generation matrix on one MoLab
allocation with identical seeds at workers 1, 2, and 4 using the documented
benchmark command, then extend to 6 and 8 only if the live doctor reports enough
CPU capacity to revise the current four-worker safety contract. Retain the raw
JSON and Markdown report before changing any generation or vLLM setting.

The smoke suite was not run because this checkout has no `tests/smoke`
directory and the required live teacher endpoint/GPU is absent.

## Addendum (2026-08-21): serving-context benchmark instrument

The vLLM tuning gates had no executable instrument. `src.mini.benchmark` now
has a `serving-context` subcommand: one invocation measures one live server
configuration (context, `max_num_seqs`, and GPU utilization are launch
settings), gated on the doctor health check, and replays real
completed-trajectory prompts from a source run at deterministic length
percentiles so 8K-versus-16K comparisons use the real prompt distribution.
Requests pin temperature 0, the run seed, thinking off, and recorded
`max_tokens`; client concurrency is bounded at eight and latency includes
queue wait so p50/p95 reflects `max_num_seqs` effects. Reports retain prompt
lengths/hashes, token usage, sanitized per-request errors (over-context and
OOM evidence), NVML peaks, and never prompt text. The `compare` command now
dispatches on stage and renders serving comparisons.

Files changed: `src/mini/benchmark.py`, `tests/unit/test_mini_benchmark.py`,
`docs/MOLAB_MINI_RUNBOOK.md`, `docs/MOLAB_MINI_IMPLEMENTATION_PLAN.md`.

Verification: focused benchmark regressions pass (18 tests, including
percentile ordering, redaction, failure-as-data, bounds, and comparison
rendering); full local suite passes (97 unit, 20 non-model integration);
`ruff check` clean; `compileall` clean. End-to-end CLI smoke against a local
loopback stub endpoint verified the real HTTP path: healthy-endpoint
measurement (6 requests, concurrency 2), HTTP 400 over-context failures
recorded as data with statuses, and stage-dispatched comparison rendering.
Stub artifacts were deleted after the smoke; no generated artifact is tracked.

Remaining risk: the live MoLab sweeps are still unexecuted, so no serving
default has changed.
