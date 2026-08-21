# MoLab Mini Phase 4 Handoff

Completed: 2026-08-19  
Host: Windows local development environment, Python 3.12.13

## Files changed

- Embedding interface/backends/cache: `src/graph/embedding.py`.
- Shared graph integration and identity hashing: `src/graph/tool_node.py` and
  `src/graph/tool_graph.py`.
- Teacher classifier/cache: `src/mini/classification.py`.
- Graph builder, validation, manifest, and CLI: `src/mini/build_graph.py`.
- Strict backend/classifier names: `src/mini/config.py`.
- Optional local-model dependency: the `mini` extra in `setup.py`.
- Tests: `tests/unit/test_embedding_cache.py`,
  `tests/unit/test_user_provided_cache.py`,
  `tests/unit/test_mini_graph_build.py`, and
  `tests/integration/test_mini_graph.py`.
- Status and verified commands: the implementation plan and runbook.

## Behavior changed

- Graph embeddings now use an injectable two-method backend interface. The
  existing OpenAI-style HTTP path remains available, while the mini profile can
  lazily load `BAAI/bge-small-en-v1.5` through sentence-transformers on CPU.
- The SQLite embedding cache partitions by secret-free backend identity and
  exact UTF-8 text SHA-256. It deduplicates requests, restores input order,
  validates matrix shape/dimension/dtype/finiteness/norms, and persists only
  normalized float32 rows.
- Parameter nodes now use identity hashing, consistent with their identity
  equality. Same-named parameters remain distinct until explicit merging.
- With parameter merging disabled, graph construction computes only the legacy
  parameter vectors used for similarity edges. Name and description vectors
  are not requested.
- User-provided classification is cached by classifier identity and a canonical
  parameter/tool prompt hash. Teacher requests use temperature 0, top-p 1, the
  recorded run seed, strict JSON-schema output, and disabled Qwen thinking.
- `src.mini.build_graph` supports static `--dry-run`, safe warm reuse, and
  explicit `--force`. It atomically writes the graph and JSON manifest. A pickle
  is loaded only after current-input and output-hash verification.
- Static graph validation checks exact catalog tools/servers, lifecycle-tool
  exclusion, required-input fillability, and 100 deterministic samples against
  the six-tool/two-server limits and dependency rules.

## Commands and results

```text
python3.12 -m compileall -q src tests
Result: passed

Focused Phase 4 tests
Result: 14 passed

python3.12 -m src.mini.build_graph \
  --config configs/mini/pipeline.toml --dry-run
Result: passed; 8 servers, 55 tools
Only unmet live dependency: Qwen/Qwen3-14B teacher classification endpoint

python3.12 -m pytest -q
Result: 55 passed

pylint -E on new/modified Phase 4 modules and tests (excluding pre-existing
matplotlib no-member false positives in tool_graph.py)
Result: passed

git diff --check
Result: passed (line-ending notices only)
```

Pytest used workspace-local uv, temp, and matplotlib cache directories because
the managed Windows sandbox does not grant writes to the default user cache and
temp locations.

## Tests not run

The sentence-transformers model was not downloaded and no teacher/model server
was available. Therefore the first live 55-tool graph build, warm live-cache
proof, and live 100-seed acceptance checks were not run. The Phase 4 contract
explicitly defers those checks to Phase 5.

## Migrations and risks

There is no existing graph migration. A prior graph without the new manifest is
treated as incomplete and is never unpickled; rebuilding it requires explicit
`--force`. Cache identities deliberately include model/device or endpoint and
recorded decoding settings, so changing those values creates cache misses.

The graph pickle remains executable trusted-local data. Future code must retain
the manifest input fingerprint and output SHA-256 checks before loading it.

## Exact next task

Begin Phase 5 by creating the separate MoLab runtime/training environments,
adding the compatible vLLM serving profile and doctor command, then run the
first live graph build twice to prove cold construction and zero-request warm
reuse before recording the live 100-seed acceptance evidence.
