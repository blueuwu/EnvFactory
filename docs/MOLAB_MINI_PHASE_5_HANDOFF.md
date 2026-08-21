# MoLab Mini Phase 5 Handoff

Status: in progress as of 2026-08-19  
Development host: Windows, with tests executed under Python 3.12 through uv

## Files changed

- Runtime and training dependency profiles: `requirements-molab.txt` and
  `requirements-molab-train.txt`.
- Historical vLLM convenience extra: `setup.py` (stale 0.8.5 pin removed).
- Read-only compatibility report and model contract: `src/mini/doctor.py`.
- Localhost single-GPU process supervisor: `src/serve/vllm_molab.sh`.
- Loopback endpoint validation: `src/mini/config.py`.
- Unit and live model tests: `tests/unit/test_mini_doctor.py`,
  `tests/unit/test_mini_config.py`, and
  `tests/integration/test_model_server_contract.py`.
- Verified and pending MoLab commands: the implementation plan and runbook.

## Behavior changed

- Runtime and LlamaFactory dependencies now resolve in independent Python 3.12
  environments. The runtime uses uv's live-driver PyTorch backend selection and
  no longer inherits a stale vLLM 0.8.5 pin.
- The doctor emits human or JSON output, captures a complete package/version
  inventory, checks platform/resources/CUDA/Blackwell/path/disk requirements,
  and records only whether the configured API-key variable is present.
- Required model health performs authenticated `/v1/models` discovery, checks
  identity and reported context length, and makes a short non-thinking chat
  completion without retaining its text.
- The launcher keeps the API key out of argv, binds to `127.0.0.1`, exposes one
  GPU with tensor parallel size 1, writes run-local logs/PID/baseline state,
  stops gracefully, and checks that used VRAM returns near its baseline.
- Teacher endpoint configuration now rejects non-loopback URLs, credentials in
  URLs, missing ports, and paths other than the `/v1` API root.
- The deployment artifact-root override now rebases graph, manifest, embedding
  cache, and classifier-cache paths together, preserving the artifact contract.

## Commands and results

```text
Initial `python -m pytest -q`
Result: collection failed because the host default was Python 3.11 without
project dependencies. No tests ran.

Focused Phase 5 plus graph integration suite under uv/Python 3.12
Result: 12 passed, 1 skipped. The skip is the explicitly gated live model test.

Full suite under uv/Python 3.12
Result: 60 passed, 1 skipped in 65.17 seconds. The skip is the explicitly gated
live model test.

Python compileall and pylint error-only checks for the Phase 5 modules/tests
Result: passed.

Git for Windows bash syntax check for `src/serve/vllm_molab.sh`
Result: passed.

`git diff --check`
Result: passed (line-ending notices only).
```

## Tests not run

No NVIDIA GPU, Linux MoLab runtime, vLLM server, or downloaded Qwen model is
available on this development host. Therefore the two virtual environments
have not been installed on MoLab, the real model health test has not run, the
cold/warm live graph has not been built, and server-stop VRAM recovery has not
been measured. Phase 5 remains `In progress`.

## Migrations and risks

There is no artifact migration. Doctor JSON uses schema version 1. The graph
artifact remains absent until the live gate. `llamafactory==0.9.4` reflects the
current upstream release but is not yet a proven MoLab lock; retain the doctor
reports from both successful live environments before treating it as verified.

The launcher authenticates `/v1` with `VLLM_API_KEY` and binds to loopback.
vLLM documents that some non-OpenAI endpoints are not covered by API-key
authentication, so preserving the loopback bind is a security requirement.

## Exact next task

On a fresh GPU-backed MoLab session, follow the Phase 5 runbook: create both
environments, save both doctor reports, launch Qwen3 at 16K, run the live model
test, build the graph cold and forced-warm with zero cache misses, record the
100-seed validation, stop the server, and retain evidence that VRAM returned
within the configured tolerance. Only then mark Phase 5 complete.
