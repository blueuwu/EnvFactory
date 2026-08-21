# MoLab Mini Phase 3 Handoff

Completed: 2026-08-19  
Host: Windows local development environment, Python 3.12.13

## Files changed

- Mini profile: `configs/mini/mcp_server.json` and
  `configs/mini/pipeline.toml`.
- Typed loading and validation: `src/mini/__init__.py` and
  `src/mini/config.py`.
- Offline catalog audit and CLI: `src/mini/catalog.py`.
- Dependency declaration: `requirements.txt` now directly declares Pydantic 2.
- Tests: `tests/unit/test_mini_config.py`,
  `tests/unit/test_mini_catalog_validation.py`, and
  `tests/integration/test_mini_catalog.py`.
- Status and commands: the implementation plan and runbook.

## Behavior changed

- The exact eight-server mini profile is now checked in with 55 metadata tools.
- TOML loading uses strict Pydantic models; unknown keys, invalid bounds,
  duplicate server names, and invalid split ratios fail before work starts.
- Repository-relative paths resolve from the package-discovered repository root
  rather than the shell directory. Callers may supply an explicit root.
- Only artifact location, embedding device, teacher endpoint, and teacher API-key
  environment-variable name accept deployment overrides. Secrets are never
  expanded into the resolved configuration.
- The catalog command detects duplicate JSON keys, path traversal/missing files,
  state/class/file mismatches, empty or duplicate tool lists, HTTP-client
  imports, missing lifecycle tools, and metadata/registration name mismatches.
- Catalog hashing frames normalized metadata and exact local tool bytes in
  configured server order.

## Commands and results

```text
Pre-edit full CPU suite
Result: 27 passed

python3.12 -m src.mini.catalog --config configs/mini/pipeline.toml --check
Result: passed; 8 servers, 55 metadata tools
Digest: 84d6dc1ab676859095a3ddfca8979978db0b083b949404a3f2f38334768b816d

Focused Phase 3 tests
Result: 14 passed

python3.12 -m compileall -q src tests <eight selected tool files>
Result: passed

python3.12 -m pytest -q
Result: 41 passed

git diff --check
Result: passed (line-ending notices only)
```

Pytest used a workspace-local `--basetemp` because the managed Windows sandbox
does not grant access to the interpreter's default user temp directory.

## Tests not run

No GPU, model-server, or MoLab notebook test was run because none is required by
the Phase 3 exit criteria. The catalog check itself is intentionally CPU-only,
credential-free, and offline.

## Migrations and risks

There is no artifact/schema migration. The catalog digest intentionally changes
when normalized metadata or any selected tool-file byte changes. HTTP-client
screening is import-based; future generated code that dynamically imports a
network client must extend the audit before catalog acceptance.

## Exact next task

Begin Phase 4 by introducing the embedding backend interface, local
sentence-transformers implementation, content-addressed SQLite cache, and graph
builder dry-run checks. Do not perform the first live teacher-classified graph
build until Phase 5 provides the local model endpoint.
