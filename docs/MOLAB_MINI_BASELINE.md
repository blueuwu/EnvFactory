# MoLab Mini Baseline Report

Captured: 2026-08-19  
Repository commit: `eff3b22` (`main`)  
Host platform: Windows, local development environment (not MoLab)

## Source dependency baseline

The repository did not contain a lock file when this baseline was captured. The
following versions are therefore the declared constraints, not a fully resolved
environment:

| Dependency group | Declared version |
|---|---|
| `fastmcp` | `3.1.0` |
| `sglang` (optional) | `0.5.9` |
| `vllm` (optional) | `0.8.5` |
| All other runtime requirements | Unpinned in `requirements.txt` |
| `pytest` (development) | `>=8,<10` |

The clean Python 3.12 interpreter used for the compatibility check contained
only `pip==26.1.2`; project dependencies were intentionally not resolved as part
of the syntax-only baseline. A MoLab dependency lock must be captured after the
Blackwell-compatible runtime and training environments are verified.

## Baseline verification

| Check | Result | Evidence |
|---|---|---|
| Python runtime | Pass | `Python 3.12.13` |
| Source compilation | Pass | `python3.12 -m compileall` over `src/` and all eight mini tool files |
| Existing test discovery before scaffolding | No tests | `pytest`: `no tests collected` |
| Test discovery after scaffolding | Pass | 10 tests collected under Python 3.12.13 |
| Unit suite | Pass | 2 passed |
| Non-model integration suite | Pass | 8 passed |
| Tracked mini artifacts, secrets, pickle, or database files | Pass | No matching tracked files |
| GPU/MoLab checks | Not run | This host is not a MoLab GPU session |

This report is a baseline, not a reproducible dependency lock. The resolved
runtime versions will be recorded by the mini doctor and run manifest in later
phases.

## Phase 0 handoff

- Files changed: `setup.py`, `.gitignore`, `pytest.ini`, `tests/`, this baseline
  report, the runbook, and the Phase 0 status in the implementation plan.
- Behavior changed: package installation now rejects Python older than 3.12 and
  exposes a `dev` extra containing pytest.
- Tests run: Python 3.12 compile, pytest collection, unit tests, non-model
  integration tests, `git diff --check`, and a tracked-artifact scan; all pass.
- Tests not run: GPU, model-server, and MoLab tests do not apply to Phase 0 and
  are unavailable on this host.
- Artifact/schema migrations: none.
- Remaining risk: runtime dependencies are mostly unpinned and have not yet been
  resolved together on MoLab.
- Exact next task: begin Phase 1 by adding and testing the public synchronous MCP
  registration wrapper.
