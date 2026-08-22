# MoLab mini final local-implementation handoff

Date: 2026-08-22
Scope: all remaining locally actionable work for EnvFactory MoLab Mini
Verdict: **local implementation complete and MoLab-ready**. All MoLab/GPU/live
verification is intentionally deferred; every gate below lists its exact future
command and evidence requirement. No deferred gate is marked passed.

## Completed phases (implementation complete locally)

| Phase | Local state |
|---|---|
| 0 Baseline | Complete (Python 3.12 compile, test markers, ignores) |
| 1 Core correctness | Complete |
| 2 Async MCP lifecycle | Complete (live MCP tests pass on CPU) |
| 3 Mini catalog/config | Complete (8 servers / 55 tools offline check) |
| 4 Embeddings/graph | Complete locally; first live teacher-classified build is a Phase 5 gate |
| 5 Serving profile | Implementation complete; venvs/vLLM install/live graph/VRAM release deferred |
| 6 Resumable synthesis | Implementation complete; live smoke/pilot runs deferred |
| 7 Dataset conversion | Implementation complete; LlamaFactory dry load on MoLab deferred |
| 8 LoRA training | Profile/render/verify implemented; any GPU training deferred |
| 9 Executable evaluation | Implemented incl. LoRA provenance checks; live scoring deferred |
| 10 Notebooks/runbook | Both marimo notebooks + supervisor implemented; clean-session reproduction deferred |
| 11 Performance hardening | Bounded registration measured; catalog-registration, generation-workers, serving-context, and training-comparison instruments all executable; all live sweeps deferred |

Phase status in `docs/MOLAB_MINI_IMPLEMENTATION_PLAN.md` remains "In progress"
for phases 5-11 precisely because their exit criteria include live MoLab
evidence. Nothing was re-marked complete without that evidence.

## Final session changes

Commit `0128a8f` - feat: add offline training-comparison benchmark for
4B-vs-8B student runs.

- `src/mini/benchmark.py`: new `training-comparison` subcommand.
  `compare_training_runs` reads two completed runs' recorded artifacts
  (`training/training_summary.json`, checkpoint `trainer_state.json`,
  optional `evaluation/student_metrics.json`) fully offline and emits a
  redacted JSON report plus a Markdown decision table: global step, loss
  bounds, `train_runtime`, `train_samples_per_second`, adapter file counts,
  dataset-manifest sha256, loss trend, and the six executable-evaluation
  rates with absolute deltas. Rejects unsafe run IDs, identical baseline and
  candidate runs, missing artifacts, and non-object payloads; training-only
  comparisons render evaluation columns as n/a.
- `tests/unit/test_mini_benchmark.py`: four new regressions - full
  throughput/quality comparison with rendering assertions, training-only
  comparison, parametrized bad-run-ID rejection, identical-run rejection -
  plus the stage-guard entry for the new renderer.

Commit `a4677b6` - docs: document training-comparison instrument and defer
its MoLab-only runs.

- `docs/MOLAB_MINI_RUNBOOK.md`: executable command block for the comparison;
  protocol unchanged; explicit statement that the two training/evaluation
  runs it compares remain MoLab GPU gates.
- `docs/MOLAB_MINI_IMPLEMENTATION_PLAN.md`: §11 instrument evidence dated
  2026-08-22 and updated Phase 11 status-table row.
- `docs/MOLAB_MINI_PHASE_11_HANDOFF.md`: addendum with files changed and
  verification.
- `.gitignore`: `/__marimo__/` and `/examples/__marimo__/` session caches.

## Verification executed this session (Python 3.12.13)

All commands were run by the supervisor against the committed HEAD:

```text
python -m compileall -q src tests                 # clean
pytest -q tests/unit                              # 108 passed, 1 skipped (POSIX-only skip)
pytest -q tests/integration -m "not model"        # 24 passed, 1 deselected (model test: no endpoint)
pytest -q tests/smoke                             # 1 passed
pytest -q tests/unit/test_mini_benchmark.py       # 28 passed
ruff check src/mini src/manager src/serve tests/unit/test_mini_benchmark.py
                                                  # All checks passed
marimo check examples/molab_mini_generate.py \
             examples/molab_mini_train.py         # clean (no diagnostics)
marimo export html <both notebooks>               # both exported, zero failed cells
git diff --check                                  # clean
```

End-to-end CLI smoke for the new stage (synthetic run directories, real file
reading path): exit code 0, decision table rendered with correct deltas
(samples_per_second 9.00 vs 4.80 = -4.20; task_success 0.71 vs 0.83 = +0.12).

Known skips (all environment-tied, none hidden):

- 1 unit skip: POSIX-only test on Windows.
- 1 integration deselection: `-m model` test requires a live model endpoint.
- Smoke suite's live-teacher path cannot run locally; hermetic smoke passes.

## Repository hygiene

- Working tree clean at `a4677b6`; only untracked path is `.hermes/`
  (supervisor tooling, not project code).
- Zero files tracked under `artifacts/`; no secrets, keys, model weights, or
  large generated artifacts tracked (`git ls-files` verified).
- No unrelated user work modified; changes limited to benchmark module,
  its tests, docs, and `.gitignore`.

## Deferred MoLab-only gates with future commands

Run these only in a fresh MoLab GPU session (Python 3.12). Full sequences are
in `docs/MOLAB_MINI_RUNBOOK.md` and §12 of the implementation plan.

1. Environments + doctor (Phase 5)
   Command: `uv venv .venv-mini-runtime --python 3.12`,
   `uv pip install ... -r requirements-molab.txt`, then
   `python -m src.mini.doctor --config configs/mini/pipeline.toml --require-model`.
   Evidence: doctor exits zero; versions captured in an environment report.
2. First live graph build + cache reuse (Phases 4/5)
   Command: `python -m src.mini.build_graph --config configs/mini/pipeline.toml`
   twice.
   Evidence: manifest written; second build performs zero embedding computes
   and zero teacher classification requests; 100-seed sampling validation.
3. Live generation smoke + killed pilot resume (Phase 6)
   Commands: `synthesize --target 10 --workers 2`, then `--target 100
   --workers 4`, kill mid-flight, `--resume` with same run ID/target.
   Evidence: exactly 100 unique completed seeds; completed `--resume`
   performs no inference; >=80% yield or documented revised gate.
4. Dataset dry load (Phase 7)
   Command: `prepare_dataset --run-id <id>` then LlamaFactory load of both
   splits. Evidence: byte-identical rerun artifacts; zero seed overlap.
5. LoRA training (Phase 8)
   Commands: 20-step 4B smoke, memory probe, one-epoch train, restart-from-
   checkpoint verification via `llamafactory-cli train` with the rendered
   run-local YAML. Evidence: finite losses, checkpoint reload/resume advance,
   adapter promoted atomically, no secrets in logs.
6. Teacher/student evaluation (Phase 9)
   Commands: `evaluate --model-role teacher` then `--model-role student`.
   Evidence: side-by-side report from source JSON; student meets release
   gates (parse >=95%, valid tool name >=98%, task success within 15 pts).
7. Clean-session notebook reproduction (Phase 10)
   Evidence: both notebooks drive the CLI workflow end to end in a fresh
   session; artifacts exported before shutdown.
8. Performance sweeps (Phase 11) - instruments ready, measurements pending:
   - Generation workers: `benchmark generation-workers --workers 1 2 4
     [--target 10]`; extend to 6/8 only after doctor justifies raising the
     four-worker contract. Evidence: raw JSON + Markdown retained.
   - vLLM tuning/context: one `benchmark serving-context` invocation per
     server configuration, then `benchmark compare`. Evidence: p50/p95,
     token usage, OOM/over-context records, NVML peaks retained per config.
   - Student sizes: train + evaluate Qwen3-4B and Qwen3-8B per the runbook
     protocol, then `benchmark training-comparison --baseline-run-id <4b>
     --candidate-run-id <8b>`. Evidence: decision table plus both raw
     training/evaluation artifact sets retained before choosing a default.

## Risks

- The bounded-registration cost (~+1.5 s median cold start) trades against an
  87% RSS reduction; revisit only with live data.
- vLLM version pins age quickly on Blackwell; install with
  `uv --torch-backend=auto` and capture versions via doctor, never hard-pin.
- The training-comparison instrument reads `latest_checkpoint` paths recorded
  by prior runs; if a run's checkpoint directory was moved externally, rerun
  `src.mini.training verify` before comparing.
- Notebook-driven jobs depend on marimo subprocess supervision remaining
  idempotent across session restores; the clean-session gate (7 above) is the
  only proof.

## Exact next task for a MoLab session

Follow §12 of the implementation plan top to bottom: create both venvs, run
doctor, validate the catalog, build the graph twice, run the 10-trajectory
smoke, then the 100-trajectory pilot with a kill/resume, convert datasets,
and record every command output into the phase handoffs as you go.
