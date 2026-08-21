# MoLab Mini Phase 10 handoff

Date: 2026-08-20  
Status: In progress; local implementation complete, clean MoLab reproduction pending

## Files changed

- `src/mini/notebook.py`: redaction, safe run/export path validation, session
  warning, bounded log tailing, artifact checklist, and persistent named
  subprocess supervision.
- `examples/molab_mini_generate.py`: ordered generation marimo notebook.
- `examples/molab_mini_train.py`: ordered training/evaluation marimo notebook.
- `tests/unit/test_mini_notebook.py`: CPU lifecycle, safety, ordering, and CLI
  coverage regressions.
- `requirements-molab.txt`: marimo runtime command.
- `docs/MOLAB_MINI_RUNBOOK.md`: terminal equivalent for every notebook action.
- `docs/MOLAB_MINI_IMPLEMENTATION_PLAN.md`: Phase 10 status and local evidence.

## Behavior changed

Expensive notebook actions require explicit run buttons and launch existing CLI
entry points as subprocesses. Named live jobs cannot be replaced by reactive
reruns. PID, command, state, and log path survive notebook restarts beneath the
artifact root; environment values are not persisted. Stop sends graceful
termination before force-kill. Both notebooks warn at 10.5 elapsed hours and
require an explicit, absolute, separate rsync destination for export.

## Local verification

```text
python -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase10-focused-2 \
  tests/unit/test_mini_notebook.py
8 passed, 1 POSIX-only test skipped

python -m compileall -q src/mini/notebook.py \
  examples/molab_mini_generate.py examples/molab_mini_train.py
passed

marimo check examples/molab_mini_generate.py examples/molab_mini_train.py
passed with no notebook diagnostics

marimo export html examples/molab_mini_generate.py -o <temporary-output>
marimo export html examples/molab_mini_train.py -o <temporary-output>
both notebooks executed with no failed cells

python -m pytest -q tests/unit
75 passed, 1 POSIX-only test skipped

python -m pytest -q tests/integration -m "not model"
19 passed, 1 deselected

python -m pytest -q tests/integration -m model
1 skipped with its documented missing-endpoint prerequisite, 19 deselected

marimo run examples/molab_mini_generate.py --host 127.0.0.1 --port 8765 --headless
marimo run examples/molab_mini_train.py --host 127.0.0.1 --port 8766 --headless
both rendered interactively; no browser-console errors
```

Interactive checks confirmed every required control in rendered order, safe
doctor failure display when the Linux-only local environment was absent,
run-ID traversal rejection, and relative export-destination rejection. These checks used no
secret, model endpoint, GPU, upload, or artifact deletion.

## Disposable Linux verification

A clean `python:3.12-slim` container mounted the repository read-only and used
marimo 0.24.0. Both notebooks passed `marimo check` and headless HTML execution;
all nine focused tests passed on Linux, directly exercising POSIX graceful and
forced termination behavior.

The source was then copied into a disposable writable Linux workspace with
separate `.venv-mini-runtime` and `.venv-mini-train` directories. Browser-driven
checks proved:

- Generation doctor launched from `/work/.venv-mini-runtime/bin/python`; its
  completed failed/no-GPU state, PID, log path, return code, and bounded report
  were visible after explicit refresh.
- Catalog validation launched from the notebook and succeeded with eight
  servers, 55 metadata tools, and the expected catalog hash.
- Changing the reactive artifact-root field did not create a new catalog job.
- Training doctor launched from `/work/.venv-mini-train/bin/python`; refresh
  retained `gpu_exclusive: true` and an empty live-model-server list alongside
  its expected no-GPU/dependency failure report.
- Neither Linux notebook produced browser-console errors.

The Linux doctor also proved that
`ENVFACTORY_MINI_ARTIFACT_ROOT=/tmp/envfactory/artifacts/mini` rebases the graph,
manifest, embedding-cache, and classification-cache paths to writable POSIX
locations. No secret value was supplied or displayed.

The pre-edit full unit attempt could not collect because the active local
Python 3.11 environment lacks project dependencies (`networkx`, `numpy`, and
`fastmcp`). This is an environment limitation, not a Phase 10 regression.

## Pending clean-session gate

- Create the isolated runtime and training environments in fresh MoLab storage.
- Open both notebooks with the installed marimo command.
- Follow every generation and training/evaluation action in order.
- Confirm visible PID/log state, live tails, graceful stops, GPU exclusivity,
  and the 10.5-hour warning.
- Export a completed run to an explicit persistent destination and verify it.

Do not mark Phase 10 complete until that GPU-backed reproduction is recorded.
