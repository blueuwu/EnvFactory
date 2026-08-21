# MoLab Mini Phase 9 handoff

Date: 2026-08-20  
Status: In progress; local implementation complete, live MoLab evaluation pending

## Files changed

- `src/mini/evaluate.py`: immutable held-out suite construction, endpoint and
  adapter health checks, isolated MCP execution, metrics, redacted traces, and
  side-by-side report generation.
- `src/mini/config.py` and `configs/mini/pipeline.toml`: explicit decoding,
  timeout, and bootstrap settings.
- `src/serve/vllm_molab.sh`: optional named vLLM LoRA serving with maximum-rank
  validation and adapter-aware health checks.
- `tests/unit/test_mini_evaluation.py`: parser, schema, execution, metric,
  split, immutability, and redaction regressions.
- `requirements.txt`: explicit JSON Schema validator dependency.
- `docs/MOLAB_MINI_RUNBOOK.md`: exact sequential teacher/student commands.

## Behavior changed

Evaluation seeds come only from the dataset manifest's validation split. The
suite records hashes rather than duplicating potentially sensitive scenarios.
Every task reconstructs its frozen reference history and schemas, creates a
fresh MCP session, verifies the initial state, executes only valid calls, saves
the final state, and closes all clients. Alternative valid tool sequences can
succeed when they reach the exact deterministic reference state; exact sequence
match remains a separate metric.

Teacher and student metrics use the same suite contract and decoding settings.
Student LoRA scoring requires an exposed adapter name distinct from the base
model, a matching health-completion model ID, a verified LoRA config, and a
served maximum rank no lower than the adapter rank. Explicit merged fallback is
supported only with base revision, adapter hash, merge dtype, and merged-model
hash provenance.

## Local verification

```text
python -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase9-focused \
  tests/unit/test_mini_evaluation.py tests/unit/test_mini_config.py
14 passed
```

## Pending live gates

- Run the teacher candidate against its healthy vLLM endpoint.
- Stop it and prove VRAM returned near baseline.
- Serve the verified student adapter through the named vLLM LoRA path.
- Prove the completion model is the adapter name and run student scoring.
- Review release gates and representative failures in `evaluation/report.md`.

The local machine has no prepared run, vLLM endpoint, LoRA artifact, or NVIDIA
runtime, so these acceptance checks cannot be substituted by CPU mocks.
