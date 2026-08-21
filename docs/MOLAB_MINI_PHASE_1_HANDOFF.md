# MoLab Mini Phase 1 Handoff

Completed: 2026-08-19  
Host: Windows local development environment, Python 3.12.13

## Files changed

- Core registration/configuration: `src/manager/mcp_client_manager.py`,
  `src/manager/llm_client_manager.py`, and `requirements.txt`.
- Generator initialization/config isolation: `src/gen/__init__.py`,
  `src/gen/env_gen/`, `src/gen/mcp_schema_gen.py`, and `src/gen/query_gen/`.
- Deterministic sampling: `src/graph/sampler.py` and
  `src/graph/tool_graph.py`.
- Verification/docs: `tests/conftest.py`,
  `tests/unit/test_core_correctness.py`, the implementation plan, and the
  runbook.

The pre-existing user edit in `README.md` was preserved and not modified as
part of Phase 1.

## Behavior changed

- `MCPClientManager.register_mcp_server(...)` is the public synchronous wrapper;
  the mixed-case spelling remains as a deprecated alias.
- MCP auto-init honors `SKIP_MCP_AUTO_INIT`. Invalid configuration is rejected
  clearly, failed multi-server initialization retains no partial catalog, and
  unregistered server use raises `MCPConfigurationError` instead of a `KeyError`.
- LLM HTTP clients and thread pools initialize lazily after configuration
  validation. Model-provider errors name the missing environment variables.
- Each `Gen` subclass initializes agents once.
- Query tool enablement is generator-local and constructors no longer share a
  default `QueryGenConfig` instance.
- Graph sampling uses one seed-local `random.Random` propagated through sampler
  calls, so concurrent calls cannot overwrite module-global RNG state.
- `ddgs` is now declared because an existing core import requires it.

## Commands and results

```text
Pre-edit: pytest -q tests/unit
Result: 2 passed

python3.12 -m compileall -q src tests
Result: passed

pytest --collect-only -q
Result: 17 tests collected

pytest -q tests/unit/test_core_correctness.py
Result: 7 passed

pytest -q
Result: 17 passed

git diff --check
Result: passed
```

The pytest commands used an ephemeral Python 3.12 runtime resolved from the
editable project plus the `dev` extra equivalent. The graph regression also ran
64 same-seed samples concurrently and compared output from two fresh Python
processes.

## Tests not run

No live model endpoint, GPU, MoLab session, or full real-server MCP lifecycle
test was run. Those require later serving/lifecycle phases and are not Phase 1
exit criteria. The synchronous registration boundary is tested with an async
registration double so it cannot launch the full repository catalog.

## Migrations and risks

There is no artifact or schema migration. Existing callers may see earlier,
more specific configuration exceptions. Real MCP process shutdown and
non-blocking async lifecycle behavior remain Phase 2 work.

## Exact next task

Begin Phase 2 by adding the native async MCP API and context-managed lifecycle,
then prove event-loop responsiveness and client cleanup with the listed stress
tests.
