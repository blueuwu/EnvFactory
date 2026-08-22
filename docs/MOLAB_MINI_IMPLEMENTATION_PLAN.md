# EnvFactory Mini on MoLab: Complete Implementation Plan

Status: proposed implementation specification  
Owner: unassigned  
Last updated: 2026-08-19  
Target platform: one NVIDIA RTX PRO 6000 Blackwell GPU with 96 GB VRAM on MoLab  
Required Python: 3.12 or newer

## 1. Purpose

This document is the implementation contract for building a small, reproducible version of EnvFactory that can generate tool-use SFT data, fine-tune a student model, and evaluate it during MoLab's time-limited notebook sessions.

An implementation agent should be able to work from this document without relying on prior conversation. If implementation discoveries require changing a decision below, record the change in the decision log before changing code.

The mini version must reuse the existing EnvFactory core. It must not become a disconnected rewrite or duplicate the generated environments.

### Current implementation status

| Phase | Status | Evidence required to advance |
|---|---|---|
| 0. Baseline and compatibility | Complete | Python 3.12.13 compile passed; 10 tests collected, 2 unit and 8 non-model integration tests passed |
| 1. Core correctness | Complete | 7 Phase 1 regressions and full 17-test suite passed; concurrent/cross-process RNG evidence recorded |
| 2. Async MCP lifecycle | Complete | 5 live MCP async/lifecycle tests passed; 100-session stress returned tracked clients and descendant processes to baseline; full 27-test suite passed |
| 3. Mini catalog/config | Complete | Strict config tests passed; offline catalog check reported eight servers/55 tools; full 41-test suite passed |
| 4. Embeddings/graph implementation | Complete | 14 Phase 4 tests and full 55-test suite passed; eight-server dry run found only the deferred teacher endpoint |
| 5. MoLab serving and live graph gate | In progress | Focused local Phase 5/graph tests: 12 passed, one live model test skipped; full CPU suite: 60 passed, one live model test skipped. MoLab GPU, live graph/cache, and VRAM-release gates remain. |
| 6. Resumable synthesis | In progress | CPU fault-injection/resume tests pass; live MoLab 10/100 trajectory gates remain |
| 7. Dataset conversion | In progress | Local deterministic conversion tests and exact Qwen3 tokenizer probe pass; run-specific LlamaFactory dry load on MoLab remains |
| 8. LoRA training | In progress | Local profile/render/checkpoint-integrity tests pass; MoLab smoke, resume, memory probe, and full training remain |
| 9. Executable evaluation | In progress | Local executable-evaluation tests pass (strict tool-output parsing, fresh-state item execution, deterministic bootstrap metrics, source-linked reports, redaction) and the served-model identity probe guards the teacher boundary; live MoLab teacher/student report remains |
| 10. Notebooks/runbook | In progress | Two ordered marimo notebooks, tested process supervision, and action-complete CLI runbook are implemented locally; clean-session MoLab reproduction remains |
| 11. Performance hardening | In progress | Bounded-registration before/after report complete; worker, serving-context, and training-comparison (4B vs 8B) instruments implemented with executable benchmark commands; live worker, vLLM tuning, 8K/16K context, and student-model sweeps remain MoLab gates |

Update this table only after the corresponding phase exit criteria pass. Use `In progress`, `Blocked`, or `Complete`; if blocked, add the blocker and evidence immediately below the table.

## 2. Target outcome

The completed system must support this workflow:

```text
audited eight-server catalog
        |
        v
cached mini tool graph
        |
        v
Qwen3 teacher served locally through an OpenAI-compatible API
        |
        v
resumable generation of 2,000-5,000 validated trajectories
        |
        v
deterministic SFT train/validation datasets
        |
        v
LoRA fine-tuning of a Qwen3 4B or 8B student
        |
        v
held-out executable tool-use evaluation and run report
```

The first release is complete only when a fresh MoLab session can run the documented smoke workflow, resume an interrupted generation run, train a LoRA adapter, and produce evaluation metrics.

## 3. Platform constraints and resulting decisions

MoLab currently documents a default notebook allocation of 4 CPUs and 32 GB RAM, optional RTX PRO 6000 Blackwell access, sessions of at most 12 hours, shutdown after 90 idle minutes, and public-but-unlisted notebooks. Treat these as hard constraints until a live doctor command proves otherwise.

Sources:

- MoLab runtime: https://marimo.io/blog/reintroducing-molab
- GPU specification: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
- Current vLLM GPU installation guidance: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- Teacher model reference: https://huggingface.co/Qwen/Qwen3-14B

Consequences:

1. All long operations must checkpoint and resume.
2. CPU worker counts must be capped at four and normally remain at two or three.
3. Model serving and training run in separate phases; do not keep the teacher resident while training.
4. Bind the model server to `127.0.0.1`, not a public interface.
5. Never place API keys or tokens in a notebook cell, output, committed file, manifest, or command-line argument.
6. Treat notebook storage as ephemeral until persistence is verified. Export important artifacts before session end.
7. Use a Blackwell-compatible PyTorch/vLLM build with CUDA 12.8 or newer. Do not install the repository's current `vllm==0.8.5` extra on MoLab without an explicit compatibility test.

## 4. Scope

### 4.1 Included in the first release

- A fixed, audited mini catalog using existing generated environments.
- Local graph construction and graph caching.
- Local teacher inference through vLLM.
- Non-conversational SFT trajectory generation.
- Bounded concurrency and non-blocking MCP calls.
- Atomic per-trajectory persistence and manifest-based resume.
- Deterministic data validation and train/validation splitting.
- LoRA training for a 4B or 8B student.
- Executable held-out evaluation.
- Two MoLab marimo notebooks: generation and training/evaluation.
- Unit, integration, smoke, resume, and resource-lifecycle tests.
- A run report containing versions, configuration, timing, quality metrics, and resource peaks.

### 4.2 Explicitly excluded from the first release

- Web discovery of new MCP schemas.
- Automatic environment code generation and revision on MoLab.
- RL/VeRL training.
- Full-parameter fine-tuning.
- Multi-GPU tensor parallelism.
- More than eight MCP servers.
- External-network tools during trajectory execution.
- A public hosted inference endpoint.
- Distributed storage or a database service.

These exclusions are deliberate. Add them only after the first-release acceptance criteria pass.

## 5. Fixed mini catalog

Use exactly these environments for the initial profile:

| Server | Metadata | Tool implementation | Metadata tool count | Purpose |
|---|---|---|---:|---|
| Calculator | `envs/metadata/Calculator_metadata.json` | `envs/tools/Calculator.py` | 1 | Numeric transformation and simple dependency chains |
| Calendar | `envs/metadata/Calendar_metadata.json` | `envs/tools/Calendar.py` | 6 | Search, scheduling, and state mutation |
| CampusCard | `envs/metadata/CampusCard_metadata.json` | `envs/tools/CampusCard.py` | 6 | Accounts, balances, and transactions |
| HotelBooking | `envs/metadata/HotelBooking_metadata.json` | `envs/tools/HotelBooking.py` | 5 | Search-to-reservation workflows |
| MovieRecommender | `envs/metadata/MovieRecommender_metadata.json` | `envs/tools/MovieRecommender.py` | 3 | Filtering and recommendation |
| Retail | `envs/metadata/Retail_metadata.json` | `envs/tools/Retail.py` | 13 | Catalog, cart, order, and account workflows |
| Telecom | `envs/metadata/Telecom_metadata.json` | `envs/tools/Telecom.py` | 12 | Customer, billing, and service workflows |
| Weather | `envs/metadata/Weather_metadata.json` | `envs/tools/Weather.py` | 9 | Location/date lookup and read-only information |

Total metadata tools: 55.

All eight implementations currently use the standard library, Pydantic, and FastMCP only. The catalog intentionally excludes filesystem, Git, email, payment, and live HTTP tools.

`Calculator` is currently absent from `configs/mcp_server.json`; the mini configuration must add it explicitly.

Before accepting the catalog, add a test that imports each file, lists its MCP tools, verifies `load_scenario` and `save_scenario`, and confirms that metadata tool names match registered tool names after excluding lifecycle tools.

## 6. Default model and workload profile

### 6.1 Teacher

- Primary: `Qwen/Qwen3-14B`, BF16.
- Fallback if model loading exceeds host RAM: `Qwen/Qwen3-8B`, BF16.
- Runtime: current Blackwell-compatible vLLM.
- Tensor parallel size: 1.
- Model context: 16,384 tokens for the first release.
- GPU memory utilization: start at 0.85.
- Maximum concurrent sequences: start at 4; increase to 6 or 8 only after measurement.
- Server bind address: `127.0.0.1`.
- Server port: `8000`.
- API key: non-secret local placeholder.
- Thinking: enabled only where query solving or validation benefits from it; generated history must not retain unnecessary thinking text.

Do not enable YaRN for the mini profile. The native Qwen3 context is sufficient, and the Qwen model documentation warns that static YaRN can hurt shorter-context performance.

### 6.2 Generation defaults

- Mode: non-conversational SFT.
- Target trajectories: 2,000 for the first complete run; 5,000 only after the 2,000-run report passes.
- Smoke target: 10.
- Pilot target: 100.
- `pass_k`: 2.
- Maximum sampled tools per chain: 6.
- Maximum distinct servers per chain: 2.
- Generation workers: 4 initially.
- Maximum solve iterations: 8.
- Maximum state-machine iterations: 8.
- Query refinement: disabled in smoke and pilot; optional for the full run after quality comparison.
- User interaction and user tool use: disabled in the first release.
- Filtering: disabled for smoke, enabled only if a 100-trajectory comparison shows a measurable quality gain.
- Checkpoint interval: every completed trajectory, with a manifest flush every 10 completions.
- Maximum attempts per seed: 2.
- Default seed namespace: integer seeds `0..target-1` transformed through a recorded run seed.

### 6.3 Student training defaults

- Primary student: `Qwen/Qwen3-8B`.
- Faster smoke student: `Qwen/Qwen3-4B`.
- Method: BF16 LoRA first; use 4-bit QLoRA only if measured memory or training time requires it.
- Sequence cutoff: 8,192 tokens initially.
- Epochs: 1 for pilot, then 1-2 based on validation loss and executable evaluation.
- Per-device batch size: 1 or 2, selected by a memory probe.
- Gradient accumulation: choose an effective batch of 16 sequences.
- Gradient checkpointing: enabled.
- Preprocessing workers: 3.
- Dataloader workers: 2.
- Cache reuse: enabled after the first preprocessing pass.
- Checkpoint strategy: save adapter and trainer state often enough to lose no more than 30 minutes of work.
- Output: adapter only by default; merging is a separate optional export step.

## 7. Proposed repository layout

Create or modify the following files. Do not rename existing public modules unless a compatibility shim remains.

```text
configs/
  mini/
    mcp_server.json
    pipeline.toml
    llamafactory_sft.yaml
    dataset_info.example.json
docs/
  MOLAB_MINI_IMPLEMENTATION_PLAN.md
  MOLAB_MINI_RUNBOOK.md
examples/
  molab_mini_generate.py
  molab_mini_train.py
src/
  manager/
    mcp_client_manager.py
    llm_client_manager.py
  mini/
    __init__.py
    artifacts.py
    build_graph.py
    catalog.py
    config.py
    doctor.py
    evaluate.py
    manifest.py
    prepare_dataset.py
    synthesize.py
  serve/
    vllm_molab.sh
tests/
  conftest.py
  unit/
    test_mini_config.py
    test_mini_manifest.py
    test_mini_catalog.py
    test_embedding_cache.py
    test_data_split.py
  integration/
    test_mcp_async.py
    test_mcp_lifecycle.py
    test_mini_graph.py
    test_mini_resume.py
    test_model_server_contract.py
  smoke/
    test_mini_pipeline.py
requirements-molab.txt
requirements-molab-train.txt
```

Generated files belong under `artifacts/mini/` and must remain ignored by Git.

## 8. Configuration contract

### 8.1 `configs/mini/mcp_server.json`

It must contain only the eight fixed servers and use repository-relative POSIX-style paths. Every entry is stateful.

```json
{
  "mcpServers": {
    "Calculator": {"tool_path": "envs/tools/Calculator.py", "stateless": false},
    "Calendar": {"tool_path": "envs/tools/Calendar.py", "stateless": false},
    "CampusCard": {"tool_path": "envs/tools/CampusCard.py", "stateless": false},
    "HotelBooking": {"tool_path": "envs/tools/HotelBooking.py", "stateless": false},
    "MovieRecommender": {"tool_path": "envs/tools/MovieRecommender.py", "stateless": false},
    "Retail": {"tool_path": "envs/tools/Retail.py", "stateless": false},
    "Telecom": {"tool_path": "envs/tools/Telecom.py", "stateless": false},
    "Weather": {"tool_path": "envs/tools/Weather.py", "stateless": false}
  }
}
```

### 8.2 `configs/mini/pipeline.toml`

Implement typed loading for this shape. Unknown keys must fail fast. Relative paths resolve from the repository root, not the current shell directory.

```toml
schema_version = 1
run_seed = 42
artifact_root = "artifacts/mini"

[catalog]
mcp_config = "configs/mini/mcp_server.json"
metadata_dir = "envs/metadata"
servers = [
  "Calculator", "Calendar", "CampusCard", "HotelBooking",
  "MovieRecommender", "Retail", "Telecom", "Weather"
]

[graph]
path = "artifacts/mini/graph/graph.pkl"
manifest_path = "artifacts/mini/graph/manifest.json"
embedding_backend = "sentence_transformers"
embedding_model = "BAAI/bge-small-en-v1.5"
embedding_device = "cpu"
embedding_batch_size = 64
embedding_cache = "artifacts/mini/cache/embeddings.sqlite3"
enable_parameter_merge = false
enable_llm_edges = false
classify_user_provided = true
user_provided_classifier = "teacher"
user_provided_cache = "artifacts/mini/cache/user_provided.sqlite3"

[teacher]
provider = "vllm"
model = "Qwen/Qwen3-14B"
fallback_model = "Qwen/Qwen3-8B"
base_url = "http://127.0.0.1:8000/v1"
api_key_env = "VLLM_API_KEY"
max_model_len = 16384
gpu_memory_utilization = 0.85
max_num_seqs = 4

[generation]
mode = "sft_non_conv"
target_trajectories = 2000
workers = 4
pass_k = 2
max_nodes = 6
max_servers = 2
max_iterations = 8
max_solve_iterations = 8
max_attempts_per_seed = 2
enable_query_refinement = false
enable_user_interaction = false
enable_user_tool_use = false
enable_filteration = false
manifest_flush_every = 10

[dataset]
train_ratio = 0.90
validation_ratio = 0.10
split_seed = 42
minimum_tool_calls = 1
maximum_sequence_tokens = 8192
tokenizer_model = "Qwen/Qwen3-8B"
tokenizer_revision = "main"
allow_server_imbalance = false

[evaluation]
held_out_trajectories = 100
workers = 2
max_turns = 8
```

Environment variables override secrets and deployment-specific values only. Structural experiment settings must remain in the TOML file and be copied into each run manifest.

## 9. Artifact contract

Every invocation receives or creates a `run_id`. Default format:

```text
YYYYMMDDTHHMMSSZ-<git-short-sha>-<8-char-config-hash>
```

Artifact layout:

```text
artifacts/mini/
  graph/
    graph.pkl
    manifest.json
  cache/
    embeddings.sqlite3
  runs/<run_id>/
    run_manifest.json
    resolved_config.toml
    environment.json
    logs/
      events.jsonl
      model_server.log
    sampled_chains/
      <seed>.json
    trajectories/
      completed/<seed>.json
      failed/<seed>.json
    datasets/
      sft_all.json
      sft_train.json
      sft_validation.json
      dataset_manifest.json
    training/
      checkpoints/
      adapter/
      trainer_state.json
    evaluation/
      teacher.json
      student.json
      failures.jsonl
      report.md
```

Rules:

1. Write mutable JSON artifacts atomically through a temporary file in the same directory followed by `os.replace`.
2. JSONL event logs may append, flush after every record, and must tolerate a truncated final line.
3. A completed trajectory file is immutable. Never overwrite it during resume.
4. A retry writes only after the previous failure record is preserved in the manifest.
5. Hash graph inputs, resolved configuration, metadata files, MCP tool files, and model identifier.
6. Refuse to resume when a compatibility hash changes unless `--new-run` is used.
7. Do not load a remotely downloaded pickle. The graph pickle is trusted only when its manifest hash matches a graph created from the current audited inputs.

## 10. Run manifest schema

Implement the manifest as a versioned dataclass or Pydantic model. Required fields:

- `schema_version`
- `run_id`
- `state`: `created`, `running`, `interrupted`, `completed`, or `failed`
- `created_at`, `updated_at`, `completed_at`
- `git_commit`, `git_dirty`
- `config_sha256`, `graph_sha256`, `catalog_sha256`
- `python_version`, `platform`, `torch_version`, `cuda_version`, `vllm_version`
- `gpu_name`, `gpu_vram_bytes`, `host_ram_bytes`, `cpu_count`
- `teacher_model`, `teacher_revision`, `student_model`, `student_revision`
- `target_trajectories`
- `seeds`: ordered list
- `completed_seeds`, `failed_seeds`, `pending_seeds`
- `attempts_by_seed`
- `failure_summary`
- `started_monotonic_seconds`, `elapsed_seconds`
- `trajectory_rate_per_hour`
- `peak_gpu_memory_bytes`, `peak_host_rss_bytes`
- `dataset_counts`
- `training_summary`
- `evaluation_summary`

Never store tokens, keys, authorization headers, full environment variables, or raw secrets.

## 11. Implementation phases

Execute phases in order. A later phase must not begin until the previous phase's exit criteria pass.

### Phase 0: Baseline capture and compatibility gate

Tasks:

- [x] Create `docs/MOLAB_MINI_RUNBOOK.md` with commands as they are verified.
- [x] Add `python_requires=">=3.12"` to `setup.py` because the conversational generator uses Python 3.12 f-string grammar.
- [x] Add pytest to a development dependency group or documented development requirements.
- [x] Record the current dependency versions in a baseline report.
- [x] Run compile checks with Python 3.12.
- [x] Add a test discovery command and establish unit/integration markers.
- [x] Add `artifacts/mini/` and local model/cache directories to `.gitignore`.
- [x] Ensure all baseline changes preserve the full project's current behavior.

Exit criteria:

- Python 3.12 compiles every file under `src/`, selected `envs/tools/`, and new mini modules.
- `pytest --collect-only` finds the new suites.
- No secret or generated artifact is tracked.

### Phase 1: Correctness prerequisites in the existing core

Tasks:

- [x] Implement a public synchronous `register_mcp_server(...)` wrapper around `register_mcp_server_async(...)`.
- [x] Replace both nonexistent `register_MCP_server(...)` calls in `src/gen/env_gen/env_gen.py` with the public snake-case method.
- [x] Optionally retain `register_MCP_server` as a deprecated compatibility alias for downstream users.
- [x] Honor `SKIP_MCP_AUTO_INIT` at the bottom of `src/manager/mcp_client_manager.py`.
- [x] Remove duplicate subclass `load_agents()` calls. `Gen.__init__()` already dispatches to the subclass implementation.
- [x] Stop mutating shared `QueryGenConfig` inside tool classification. Store run-specific enablement on the generator/context.
- [x] Replace module-global random seeding in graph sampling with an explicit `random.Random` instance propagated through samplers.
- [x] Add clear exceptions when model or MCP configuration is missing instead of partially initializing a global singleton.

Tests:

- `test_register_mcp_server_sync_wrapper`
- `test_skip_auto_init`
- `test_agent_initialization_occurs_once`
- `test_query_configs_are_not_mutated_across_runs`
- `test_graph_sampling_is_reproducible_under_concurrency`

Exit criteria:

- `EnvGen` reaches registration without `AttributeError`.
- Two concurrent generators using the same source configuration do not alter each other's flags.
- Sampling the same seed produces the same ordered tool list in repeated processes.

### Phase 2: Non-blocking MCP API and lifecycle

The background MCP event-loop design may remain for compatibility, but asynchronous callers must not block their own event loop on `Future.result()`.

Required public API:

```python
async def aload_scenario(
    self, client_id: str, scenario: dict | None = None, check: bool = False
) -> str: ...

async def acall_tool(
    self, client_id: str, tool_name: str, tool_args: dict | str
) -> str: ...

async def asave_all_scenarios(
    self, client_ids: list[str]
) -> dict[str, dict | None]: ...

async def aclose_client(self, client_id: str) -> None: ...
```

Implementation requirements:

- [x] Add one internal submission helper that uses `asyncio.run_coroutine_threadsafe` and `asyncio.wrap_future` for asynchronous callers.
- [x] Move get-or-create logic into a coroutine that always runs on the MCP manager loop.
- [x] Keep existing synchronous methods as wrappers for notebooks and legacy code.
- [x] Never call a synchronous manager method from inside a coroutine running on the manager loop.
- [x] Load scenarios for different servers with `asyncio.gather` and bounded concurrency.
- [x] Save scenarios for different servers with `asyncio.gather` and bounded concurrency.
- [x] Preserve emitted tool-call order by default. Do not parallelize multiple calls from one model response unless dependency independence is explicitly proven.
- [x] Give registration, connection, scenario loading, tool calls, and shutdown separate configurable timeouts.
- [x] On timeout, cancel the submitted coroutine, record the operation/client/server, and clean up the affected session.
- [x] Verify whether a FastMCP child session requires explicit `close()`. Implement deterministic cleanup based on the installed FastMCP public API.
- [x] Avoid private `Client._connect()` if the pinned FastMCP version provides a supported equivalent.
- [x] Make shutdown idempotent.
- [x] Ensure a failed server registration closes the partially connected client.

Update all async query-generation and validation paths to use the async API, including scenario load, tool execution, final scenario save, and cleanup.

Tests:

- A ticker coroutine must continue advancing while a deliberately slow MCP call is in progress.
- Four Calculator sessions must remain isolated.
- Closing 100 short-lived sessions must return client counts and child-process counts to baseline.
- A timed-out call must not poison the next session.
- Repeated shutdown calls must succeed.

Exit criteria:

- No `Future.result()` is reachable from an async generation path.
- The lifecycle integration test has no leaked tracked clients or child processes.

Verified locally on 2026-08-19 with Python 3.12.13 and FastMCP 3.1.0. The
non-model integration suite passed 13 tests, including 5 Phase 2 live MCP tests;
the full suite passed 27 tests. Static regressions also verify that async query
generation and validation functions do not call synchronous MCP manager methods.

### Phase 3: Mini configuration and catalog validation

Tasks:

- [x] Create the exact MCP JSON and TOML files defined above.
- [x] Implement `src/mini/config.py` using typed models.
- [x] Implement repository-root discovery using the location of the package, with an explicit `--repo-root` override.
- [x] Implement environment-variable expansion only for allowed deployment fields.
- [x] Implement `src/mini/catalog.py` to load selected metadata in configured order.
- [x] Reject duplicate server names, missing paths, class-name mismatches, empty tool lists, and tools that import external HTTP clients.
- [x] Compare metadata names and registered MCP names.
- [x] Produce a catalog digest from normalized metadata and tool-file bytes.
- [x] Add `python -m src.mini.catalog --config ... --check`.

Exit criteria:

- Catalog check reports eight servers and 55 metadata tools.
- Catalog check requires no API key and makes no external network request.

Verified locally on 2026-08-19 with Python 3.12.13 and FastMCP 3.1.0. The
catalog CLI imported and introspected all eight local servers in-process without
starting stdio clients, reported 55 metadata tools, and produced digest
`84d6dc1ab676859095a3ddfca8979978db0b083b949404a3f2f38334768b816d`.
Fourteen focused Phase 3 tests and the full 41-test suite passed.

### Phase 4: Local embedding backend and graph build

Refactor embedding behind a small interface rather than adding mini-specific branches throughout graph code.

This phase implements and unit-tests the graph path. The first live graph build occurs at the end of Phase 5, after the local teacher endpoint exists, because the default `user_provided_classifier = "teacher"` requires that endpoint.

Required interface:

```python
class EmbeddingBackend(Protocol):
    @property
    def identity(self) -> str: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
```

Tasks:

- [x] Preserve the current HTTP backend.
- [x] Add a sentence-transformers backend using `BAAI/bge-small-en-v1.5` on CPU.
- [x] Add a content-addressed SQLite embedding cache keyed by backend identity plus exact UTF-8 text hash.
- [x] Deduplicate input texts before embedding and restore original order afterward.
- [x] Validate returned row count, dimension, finite values, and dtype.
- [x] Store normalized `float32` vectors.
- [x] Compute only embeddings required by the selected graph mode. With parameter merging disabled, do not compute unused name and description vectors.
- [x] Change `Parameter.__hash__` to identity hashing or add equality consistent with its hash; do not retain name-only collisions with identity equality.
- [x] Keep LLM edge construction disabled in the default mini profile.
- [x] Cache user-provided classification by normalized parameter/tool prompt hash.
- [x] Run teacher classification in non-thinking structured-output mode with a recorded decoding seed and settings.
- [x] Implement `python -m src.mini.build_graph --config ... [--force]`.
- [x] Save a graph manifest containing all input hashes, counts, thresholds, backend identity, duration, and output SHA-256.

Graph validation must check:

- Every selected metadata tool has one graph Tool node.
- Every Tool node belongs to an allowed server.
- Every required input is user-provided or reachable from an earlier tool according to validation rules.
- There are no references to lifecycle tools in sampled chains.
- A 100-seed sampling test produces valid chains of at most six tools and two servers.

Exit criteria:

- Embedding backend, deduplication, cache, hashing, and graph-manifest unit tests pass without a model server.
- A graph-builder dry run validates inputs and reports that teacher classification is the only unmet live dependency.
- The live graph acceptance checks listed above are deferred to and required by Phase 5.

Verified locally on 2026-08-19 with Python 3.12.13. Fourteen dedicated Phase 4
tests passed, including backend validation, embedding/classification cache
persistence, selective graph embeddings, manifest hashing, trusted warm reuse,
and the real eight-server dry run. The full 55-test suite passed. The dry run
reported all eight servers and 55 tools and named only the configured local
teacher classification endpoint as an unmet live dependency. No graph artifact
was created; the first teacher-classified graph build and its live 100-seed
acceptance result remain Phase 5 requirements.

### Phase 5: MoLab model-serving profile

Tasks:

- [ ] Create separate `.venv-mini-runtime` and `.venv-mini-train` Python 3.12 environments so vLLM and training-framework dependency resolution cannot silently break one another.
- [x] Create `requirements-molab.txt` for the runtime/generation environment without a stale hard pin to vLLM 0.8.5.
- [x] Create `requirements-molab-train.txt` for the separately installed and verified LlamaFactory training dependencies.
- [ ] Record both environments in the runbook and environment report.
- [ ] Share only the Hugging Face cache and artifact directory between the two environments; do not share their site-packages.
- [ ] Install vLLM using a Blackwell-compatible CUDA backend selected against the live driver.
- [ ] After a successful session, capture exact working packages in a lock or environment report.
- [x] Create `src/serve/vllm_molab.sh` with `set -euo pipefail`.
- [x] Validate required environment variables without printing secret values.
- [x] Use one GPU and tensor parallel size 1.
- [x] Bind to localhost and write logs under the current run directory.
- [x] Add configurable model, port, context, memory utilization, and max sequences.
- [x] Add a health check that validates `/v1/models` and one short chat completion.
- [x] Add graceful stop logic and wait for VRAM to return near baseline.
- [x] Implement `src/mini/doctor.py`.
- [ ] With the endpoint healthy, run the first live graph build from Phase 4.
- [ ] Re-run the build and verify embedding and user-classification cache hits.

Doctor output must include:

- Python executable and version.
- OS and architecture.
- CPU count and host RAM.
- Disk free space in repository, model cache, and artifact locations.
- GPU name, driver, total/free VRAM, and compute capability.
- PyTorch version, compiled CUDA version, and `torch.cuda.is_available()`.
- vLLM version and import status.
- Required path existence and writability.
- Model endpoint health and returned model identifier.
- Presence, but never values, of required secret variables.

Exit criteria:

- Doctor exits zero on a compatible MoLab runtime.
- A model health request succeeds at 16K configured context.
- First graph build completes and writes its manifest.
- Second graph build performs zero embedding computation and zero teacher-classification requests.
- A 100-seed graph sampling validation satisfies the server/tool limits and chain-validation rules.
- Stopping the server releases enough GPU memory to begin training.

### Phase 6: Resumable trajectory synthesis

Implement `python -m src.mini.synthesize` as the production entry point. Do not use `examples/sythesize_query.py` directly.

CLI contract:

```text
python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --run-id <optional> \
  --target 100 \
  --workers 4 \
  --resume
```

Tasks:

- [x] Validate doctor, graph manifest, catalog hash, model identity, and output directory before creating work.
- [x] Precompute the ordered seed list deterministically.
- [x] Pre-sample and atomically store each tool chain before model inference.
- [x] Use a bounded `asyncio.Queue`; do not create thousands of tasks at once.
- [x] Use one worker-local QueryGen instance per worker and explicitly reset all run-specific state before each seed.
- [x] Ensure every session and MCP client closes in `finally` blocks.
- [x] Persist successful trajectories individually.
- [x] Persist structured failures with stage, exception class, safe message, retryability, attempt, and timestamps.
- [x] Retry only transient model, timeout, or transport failures. Do not retry deterministic schema/configuration failures.
- [x] On SIGINT/SIGTERM, stop accepting work, allow in-flight atomic writes, mark the run interrupted, close clients, and stop cleanly.
- [x] Resume only pending and retryable failed seeds.
- [x] Prevent two processes from writing the same run using an advisory lock file containing PID and start time.
- [x] Detect stale locks conservatively and require `--recover-stale-lock` to replace one.
- [x] Emit one JSONL event per state transition and periodic human-readable progress.
- [x] Track attempted, completed, failed, retried, and valid counts separately.
- [x] Sample process RSS and GPU memory every 10 seconds without failing the run if monitoring is unavailable.

Do not count a trajectory as completed unless:

- It is valid JSON and conforms to `ToolQueryChain` serialization.
- It contains at least one accepted node and one tool call.
- Every tool name belongs to the mini catalog.
- Initial and final scenarios exist for every referenced server.
- Tool call and response steps alternate correctly.
- The final file has been atomically moved into `completed/`.

Exit criteria:

- A 10-trajectory smoke run completes.
- Killing a 100-trajectory run mid-flight and resuming yields exactly 100 unique completed seeds.
- Re-running `--resume` after completion performs no inference.

Implemented locally on 2026-08-19. Eight focused Phase 6 tests pass, including
immutable atomic artifacts, manifest invariants, lock recovery, transient retry
preservation, semantic compatibility refusal, and a deterministic ten-seed
fault-injection run that interrupts, resumes to exactly ten unique completed
files, then proves a completed resume does not initialize inference. The full
CPU suite passes 49 unit tests and 19 non-model integration tests. The live
MoLab/vLLM ten-trajectory smoke and killed 100-trajectory acceptance run remain
required before this phase can be marked complete. Python 3.12 compilation and
the full local suite also pass: 68 tests passed and the one live-model test was
skipped because no endpoint is available in this session.

### Phase 7: Dataset conversion and validation

Refactor reusable conversion logic out of `src/utils/data_process.py` without breaking its CLI.

Production CLI contract:

```text
python -m src.mini.prepare_dataset \
  --config configs/mini/pipeline.toml \
  --run-id <run-id>
```

Tasks:

- [x] Load completed files in numeric seed order.
- [x] Validate every trajectory before conversion.
- [x] Exclude failure artifacts and incomplete temporary files.
- [x] Calculate token lengths with the exact student tokenizer and chat template.
- [x] Record, do not silently discard, samples over the cutoff.
- [x] Split by trajectory seed before expanding trajectories into per-step SFT samples to prevent conversation leakage.
- [x] Keep the split deterministic.
- [x] Write compact JSON unless LlamaFactory requires another format.
- [x] Avoid retaining both all parsed trajectories and all expanded samples in memory where streaming is possible.
- [x] Write a dataset manifest with source run hash, tokenizer revision, counts, exclusions, role distribution, tool distribution, server combinations, and token-length percentiles.
- [x] Generate a small human-inspection report with 20 deterministic samples.

Dataset gates:

- Zero invalid tool names.
- Zero malformed tool-call JSON values.
- Zero train/validation seed overlap.
- At least 95% of retained samples fit the 8,192-token cutoff.
- No single server supplies more than 35% of samples unless the manifest explicitly records and accepts the imbalance.
- At least 80% trajectory generation yield in the 100-seed pilot. If lower, stop and diagnose before scaling.

Exit criteria:

- LlamaFactory can load both splits in a dry run.
- Re-running conversion produces byte-identical manifests and logically identical datasets.

Local evidence (2026-08-20): Phase 7 focused/config tests passed (12/12); all unit tests passed
(54/54); all non-model integration tests passed (19/19); the full local suite passed (73 passed,
one live-model skip). The Qwen3-8B tokenizer and chat template loaded at resolved revision
`b968826d9c46dd6066d109eabc6255188de91218`, and an exact-template token-count probe passed.
Synthetic reruns produced byte-identical dataset JSON, manifest, and inspection report files. The
run-specific LlamaFactory dry load remains for the MoLab training environment, so Phase 7 remains
in progress rather than complete.

### Phase 8: LoRA training profile

Create `configs/mini/llamafactory_sft.yaml` from the existing configuration with these semantic changes:

- `finetuning_type: lora`.
- LoRA target covers the model's attention and MLP linear layers, using the syntax supported by the verified LlamaFactory version.
- `cutoff_len: 8192`.
- `overwrite_cache: false` after the first successful preprocessing run.
- `preprocessing_num_workers: 3`.
- `dataloader_num_workers: 2`.
- BF16 enabled.
- Gradient checkpointing enabled.
- Effective batch size 16.
- Save adapter plus trainer state.
- Report to a local TensorBoard directory by default; external reporting is opt-in.
- Output path is run-specific and never overwritten implicitly.

Training procedure:

1. Stop vLLM and verify released VRAM.
2. Generate a run-specific `dataset_info.json` in the dataset directory with `env_factory_mini_train` and `env_factory_mini_validation` entries.
3. Render, never mutate, the shared YAML template into `training/resolved_llamafactory.yaml`; resolve dataset, output, logging, and checkpoint paths under the run directory.
4. Run a 20-step 4B smoke training job.
5. Validate loss is finite, checkpoints can reload, and resume advances from the saved step.
6. Run a memory probe for 8B with batch size 2; fall back to 1 if peak allocated memory exceeds 90 GB or an OOM occurs.
7. Train one epoch on the full training split.
8. Evaluate validation loss.
9. Save adapter, tokenizer metadata, resolved training arguments, version report, and data manifest hash.
10. Restart from the latest checkpoint for a short verification step before declaring the checkpoint usable.

Do not merge the adapter into base weights during the main training job. Merging creates a large artifact and should be an explicit export action.

Exit criteria:

- Smoke training and checkpoint resume pass.
- Full training finishes within a MoLab session or resumes correctly in the next one.
- No secret is present in trainer logs or configuration snapshots.

### Phase 9: Executable evaluation

Implement `python -m src.mini.evaluate`.

Evaluation data must be held out by seed before SFT expansion. Evaluate the teacher baseline and the student adapter under identical prompts, tool schemas, initial scenarios, maximum turns, and sampling parameters.

Metrics:

- Structured-output parse rate.
- Valid tool-name rate.
- Valid argument-schema rate.
- Tool-call execution success rate.
- Exact tool-sequence match against the selected reference trajectory.
- Final scenario exact match where deterministic.
- Task success determined by the existing validation/selector semantics.
- Average and percentile model latency.
- Average tool calls and turns per task.
- Token usage where available.
- Failure counts by parse, wrong tool, wrong arguments, execution, timeout, and final-state mismatch.

Evaluation rules:

- Temperature and other decoding settings must be recorded.
- Use a new MCP session per evaluation item.
- Do not reuse a mutated scenario.
- Preserve failed traces for debugging, redacting secrets.
- Calculate bootstrap confidence intervals for success rate when practical.
- Report teacher and student side by side.
- Evaluate sequentially: serve and evaluate the teacher, stop it and verify VRAM release, then serve the student's base model with the adapter.
- Prefer vLLM's supported LoRA serving path for the student (`--enable-lora` plus a named adapter module). Record the exposed model/adapter name and maximum LoRA rank. If the verified runtime cannot serve the adapter, perform an explicit merge into a separate export artifact and record the base revision, adapter hash, merge dtype, and merged-model hash.
- The student health check must prove that the adapter, not the unmodified base model, is active before scoring.

Provisional release gates, to be revised only after recording the teacher baseline:

- Student structured-output parse rate at least 95%.
- Student valid tool-name rate at least 98%.
- Student executable task success no more than 15 percentage points below the teacher.
- No cross-session state leakage.

Exit criteria:

- Evaluation is repeatable from a documented command.
- `evaluation/report.md` links every metric to its source JSON and lists representative failures.

Local implementation evidence (2026-08-20): `python -m src.mini.evaluate` now
freezes a held-out-seed suite before SFT-sample expansion, evaluates one fresh
MCP session per selected task, validates XML/JSON structure, tool identity and
argument schemas before execution, compares selected tool sequences and
deterministic final states, records latency/token/failure metrics and a
deterministic bootstrap interval, redacts failed traces, verifies named LoRA or
explicit-merge provenance, and renders the side-by-side source-linked report.
The model launcher supports vLLM `--enable-lora`, a named module, and a recorded
maximum LoRA rank. Focused CPU tests pass; the sequential live teacher/VRAM
release/student-adapter evaluation remains a MoLab acceptance gate, so Phase 9
is not yet declared complete.

### Phase 10: MoLab notebooks and runbook

Create two marimo notebooks stored as Python files.

`examples/molab_mini_generate.py` cell order:

1. Explanatory header and security warning.
2. Repository and artifact-root selection.
3. Read-only environment/doctor report.
4. Dependency installation instructions; do not silently reinstall on every reactive rerun.
5. Configuration preview with secrets redacted.
6. Model server start/stop controls.
7. Health check.
8. Catalog validation.
9. Graph build/cache status.
10. Smoke/pilot/full generation controls.
11. Live manifest metrics.
12. Dataset conversion and validation.
13. Artifact export checklist.

`examples/molab_mini_train.py` cell order:

1. Explanatory header and GPU exclusivity warning.
2. Doctor report and verification that vLLM is stopped.
3. Dataset/run selection.
4. Dataset manifest and sample inspection.
5. Training configuration preview.
6. Smoke training.
7. Full training/resume.
8. Adapter integrity check.
9. Teacher or student evaluation server launch as needed.
10. Evaluation controls and report display.
11. Artifact export checklist.

Notebook requirements:

- Expensive cells run only from explicit buttons or forms.
- Reactive dependency changes must not restart running jobs.
- Long jobs call CLI modules through subprocesses and stream logs rather than duplicating orchestration in notebook cells.
- Store subprocess PID and log path visibly.
- A stop button must send graceful termination before force-kill.
- Display session elapsed time and a warning at 10.5 hours.
- Provide commands equivalent to every notebook action in the runbook.

Local implementation evidence (2026-08-20): both ordered marimo Python
notebooks now orchestrate the existing CLI entry points through an idempotent,
persistent subprocess supervisor. The shared helper redacts environment secret
values, validates run/export paths, streams bounded log tails, records visible
PID/log state, sends graceful termination before force-kill, and warns at 10.5
hours. Focused CPU tests cover those invariants and statically verify notebook
order and CLI coverage. A clean GPU-backed MoLab session must still launch both
notebooks and follow the runbook before Phase 10 can be declared complete.
Python 3.12 verification totals are 75 unit tests passed with one POSIX-only
unit skip, plus 19 non-model integration tests passed; the one model-marked integration test skipped with its documented
missing-endpoint prerequisite. Marimo 0.24.0 reported no notebook diagnostics,
and both notebooks completed headless HTML execution without failed cells.
Interactive local marimo app sessions also rendered every ordered control,
showed no browser-console errors, rejected a traversal-shaped run ID and a
relative export destination, and surfaced missing local runtime executables as
sanitized notebook errors. The remaining clean-session gate is specifically the
Linux/GPU workflow and cannot be replaced by this CPU/Windows UI evidence.
Additional disposable Linux/Python 3.12 verification passed `marimo check`,
headless execution for both notebooks, and all nine focused Phase 10 tests.
In a live Linux marimo session, the generation notebook launched its doctor
through `.venv-mini-runtime/bin/python`, refreshed the completed PID/state/log,
validated the catalog at eight servers and 55 tools, and did not relaunch the
catalog when the artifact-root control changed. The training notebook launched
its isolated doctor and retained `gpu_exclusive: true` plus an empty live-server
list when its failed no-GPU report was refreshed. This narrows the remaining
gate to actual MoLab dependency installation, GPU/model work, and persistent
export.

### Phase 11: Performance hardening after correctness

Measure before applying each item and retain before/after reports.

- [x] Bound MCP registration rather than connecting all servers without a limit.
- [x] Add lazy server connection if eight-server startup is still material (measured and not selected).
- [x] Reuse worker-local agent/model objects after proving state reset and cleanup.
- [x] Batch independent graph LLM prompts used by the fixed mini profile.
- [x] Compact or stream large JSON outputs.
- [ ] Tune generation workers across 2, 4, 6, and 8.
- [ ] Tune vLLM `max_num_seqs` and GPU utilization while tracking p50/p95 latency and OOMs.
- [ ] Measure 8K versus 16K serving context using real prompt percentiles.
- [ ] Measure 4B versus 8B student training throughput and quality.

Do not optimize by parallelizing tool calls whose order may affect state.

Local implementation evidence (2026-08-20): a reusable Phase 11 benchmark
recorder measured five identical real eight-server catalog registrations before
and after introducing a two-registration semaphore (hard-capped at four). On
Windows 11/Python 3.12.13, median startup increased from 0.9220 s to 2.4220 s,
while maximum sampled process-tree RSS fell from 1,817,395,200 bytes to
233,660,416 bytes and peak descendant count fell from 35 to 6. Every sample
registered all eight servers and 71 tools including lifecycle tools, and every
sample returned to zero descendants. The committed Phase 11 handoff retains the
commands, interpretation, and the remaining MoLab-only benchmark matrix. Lazy
connection was not added: the bounded cold start remained below three seconds,
so deferring registration would move schema/configuration failures into active
trajectory work for little demonstrated benefit.

Additional local hardening evidence (2026-08-20): synthesis constructs exactly
one generator per worker, resets it before and after every trajectory attempt,
and closes all MCP clients after the worker pool stops. A dedicated six-seed,
two-worker regression proves object reuse, twelve reset boundaries, and final
client cleanup. The fixed profile's graph-time teacher classification batches
independent parameter decisions (32 per request by default); the optional
legacy LLM edge builder remains disabled. Dataset conversion already streams
compact JSON arrays through same-directory atomic replacements, with a
deterministic byte-for-byte rerun test. A new `generation-workers` benchmark
executes identical-seed 1/2/4-worker sweeps, records trajectory p50/p95,
throughput, retries, failures, CPU/RSS/GPU/process cleanup, and emits raw JSON
plus Markdown. The live sweep is not recorded here because it requires the
MoLab teacher endpoint; the worker-tuning checklist therefore remains open.

Serving-instrument evidence (2026-08-21): a `serving-context` benchmark now
makes the vLLM tuning gates directly executable. One invocation measures one
live server configuration (context, `max_num_seqs`, and GPU utilization are
launch settings), gated on a healthy doctor endpoint, and replays real
completed-trajectory prompts selected at deterministic length percentiles so
8K-versus-16K comparisons use the real prompt distribution. Requests pin
temperature 0, the run seed, thinking off, and recorded `max_tokens`; client
concurrency is bounded at eight and latency includes queue wait so p50/p95
reflects `max_num_seqs` effects. Reports retain prompt lengths/hashes, token
usage, sanitized per-request errors (over-context and OOM evidence), NVML
peaks, and never prompt text. Focused regressions cover percentile ordering,
redaction, failure-as-data recording, bounds, and comparison rendering; the
full local suite passes (97 unit, 20 non-model integration). The live 8K/16K
and `max_num_seqs` sweeps still require MoLab GPU sessions and remain open.

Training-comparison instrument evidence (2026-08-22): the last Phase 11 gate
without an executable instrument - "Measure 4B versus 8B student training
throughput and quality" - now has one. `python -m src.mini.benchmark
training-comparison` reads two completed runs' recorded artifacts offline
(`training_summary.json`, checkpoint `trainer_state.json`, and
`evaluation/student_metrics.json` when present) and emits a redacted JSON
report plus a Markdown decision table covering global step, loss bounds,
`train_runtime`, `train_samples_per_second`, adapter file counts, dataset
manifest hash traceability, and all six executable-evaluation rates with
absolute deltas. It rejects unsafe run IDs, identical baseline/candidate
runs, missing artifacts, and non-object payloads; it works with training-only
runs (evaluation columns render as n/a). Focused regressions cover throughput
and quality deltas, evaluation-free comparisons, bad-input rejection, stage
guards, and Markdown rendering; an end-to-end CLI smoke against synthetic run
directories verified the real file-reading path. The instrument itself is
offline; the two student training/evaluation runs it will compare remain
MoLab GPU gates and stay deferred.

## 12. Canonical clean-session workflow

The runbook and notebooks must implement the following logical command sequence. Replace `<run-id>` only where shown; do not invent separate undocumented entry points.

```bash
# One-time environment creation in a fresh MoLab session
uv venv .venv-mini-runtime --python 3.12
uv pip install --python .venv-mini-runtime/bin/python -e .
uv pip install --python .venv-mini-runtime/bin/python \
  --torch-backend=auto -r requirements-molab.txt

uv venv .venv-mini-train --python 3.12
uv pip install --python .venv-mini-train/bin/python -e .
uv pip install --python .venv-mini-train/bin/python \
  --torch-backend=auto -r requirements-molab-train.txt

# Runtime verification
.venv-mini-runtime/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml \
  --without-model

# Start vLLM through the runbook/notebook process supervisor, then verify it
.venv-mini-runtime/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml \
  --require-model

# Validate catalog and create/cache the graph
.venv-mini-runtime/bin/python -m src.mini.catalog \
  --config configs/mini/pipeline.toml --check
.venv-mini-runtime/bin/python -m src.mini.build_graph \
  --config configs/mini/pipeline.toml

# Generate a disposable smoke run
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --target 10 --workers 2

# Start a separate pilot run; record the run ID printed by the command
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --target 100 --workers 4

# Resume only with the pilot's original target and semantic configuration
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --run-id <run-id> --target 100 --workers 4 --resume

# Convert and validate data
.venv-mini-runtime/bin/python -m src.mini.prepare_dataset \
  --config configs/mini/pipeline.toml --run-id <run-id>

# Stop vLLM, confirm VRAM release, and use the generated run-specific YAML
.venv-mini-train/bin/llamafactory-cli train \
  artifacts/mini/runs/<run-id>/training/resolved_llamafactory.yaml

# Start the appropriate base/adapted model endpoint and evaluate
.venv-mini-runtime/bin/python -m src.mini.evaluate \
  --config configs/mini/pipeline.toml \
  --run-id <run-id> --model-role teacher
.venv-mini-runtime/bin/python -m src.mini.evaluate \
  --config configs/mini/pipeline.toml \
  --run-id <run-id> --model-role student
```

The implementation may require environment variables that point LlamaFactory to the run-specific dataset and output directory. The notebook must construct those variables from validated paths and show the resolved non-secret paths before launch. It must never modify the shared YAML in place.

## 13. Testing matrix

| Layer | Runs without GPU | Requires local MCP | Requires model server | Purpose |
|---|---:|---:|---:|---|
| Unit | Yes | No | No | Config, manifests, hashing, splitting, caches |
| Catalog integration | Yes | Yes | No | Tool/metadata parity and lifecycle tools |
| MCP async integration | Yes | Yes | No | Non-blocking calls, isolation, cleanup |
| Graph integration | Yes | No | Optional | Local embeddings and deterministic sampling |
| Model contract | No | No | Yes | OpenAI-compatible response and parser behavior |
| Pipeline smoke | Yes or small GPU | Yes | Yes | Ten complete trajectories |
| Resume fault injection | Yes or small GPU | Yes | Yes | Atomic writes and exact completion |
| Training smoke | GPU | No | No | Forward/backward, checkpoint, resume |
| Evaluation smoke | GPU | Yes | Yes | Executable end-to-end scoring |

Required standard commands:

```bash
python -m compileall -q src tests
pytest -q tests/unit
pytest -q tests/integration -m "not model"
pytest -q tests/integration -m model
pytest -q tests/smoke -m smoke
```

Every model/GPU test must have a marker and a clear skip reason when its prerequisites are absent. Unit tests must never require network access.

## 14. Benchmark protocol

Create a benchmark report for these stages:

1. Catalog initialization.
2. Cold graph build.
3. Warm graph build.
4. MCP load/call/save round trip for each server.
5. Ten-trajectory generation at workers 1, 2, and 4.
6. Hundred-trajectory pilot at the selected worker count.
7. Dataset conversion.
8. Twenty-step training smoke.
9. Held-out evaluation.

Record:

- Wall-clock and monotonic duration.
- CPU utilization and peak RSS.
- GPU utilization, peak allocated/reserved VRAM, and temperature if available.
- Request count, prompt/output tokens, latency p50/p95, retries, and timeouts.
- MCP process count before, during, and after.
- Successful trajectories per hour and cost if the platform exposes it.

Use the same seed set when comparing configurations. Do not compare throughput from different prompt or tool-chain distributions.

## 15. Failure recovery

### Model server failure

- Stop scheduling new work.
- Mark in-flight seeds retryable.
- Preserve completed files.
- Record the last successful health check and server log tail.
- Restart the server, verify model identity, and resume.

### MCP process failure

- Close the affected client and base connection.
- Retry the seed once with a fresh session if classified transient.
- If repeated, quarantine the server/seed pair and continue other seeds.
- Fail the run if one server exceeds a 20% failure rate in the pilot.

### Notebook/session termination

- Manifest and completed trajectories already reside on disk.
- In the next session, restore code and artifacts, run doctor, start the same model revision, and invoke `--resume` with the same run ID.
- Refuse resume if config, catalog, graph, or model compatibility hashes differ.

### Disk pressure

- Warn below 20 GB free.
- Stop starting new trajectories below 10 GB free.
- Never delete completed artifacts automatically.
- Offer an explicit export-and-prune command that prints exact targets and requires confirmation.

### Out-of-memory

- Capture the operation, model, prompt length, batch/concurrency, allocated/reserved VRAM, and stack trace.
- For serving, lower maximum sequences before lowering context.
- For training, lower per-device batch size before enabling more aggressive quantization.
- Resume rather than restarting the run from zero.

## 16. Security and safety requirements

- Bind inference only to localhost.
- Use environment-injected secrets and redact values from diagnostics.
- Never commit `.env`, Hugging Face tokens, model-provider keys, or artifact manifests containing environment dumps.
- Do not include Filesystem or GitServer in the mini catalog.
- Treat generated tool code and pickle files as executable/untrusted unless created from the audited local inputs.
- Do not automatically upload artifacts. Export requires an explicit user-configured destination and command.
- Do not log complete authorization-bearing request headers.
- Sanitize exception messages before writing shared notebook output.

## 17. Observability requirements

Use structured JSONL events with at least:

- timestamp
- run ID
- stage
- seed when applicable
- worker ID
- server and tool when applicable
- operation
- duration
- outcome
- retry count
- exception class and safe message
- prompt/output token usage when available

Human console output should be concise summaries derived from these events. Do not print full prompts or scenarios by default.

The final report must include:

- Resolved configuration and hashes.
- Environment versions.
- Catalog and graph statistics.
- Generation yield and throughput.
- Failure breakdown.
- Dataset statistics.
- Training summary and loss curve location.
- Teacher/student evaluation comparison.
- Peak resources.
- Known limitations and recommended next experiment.

## 18. Definition of done

The mini implementation is done when all statements below are true:

- [ ] A fresh Python 3.12 MoLab environment passes doctor.
- [ ] The exact eight-server catalog validates with 55 metadata tools.
- [ ] The graph builds locally, caches, and samples deterministically.
- [ ] Async MCP calls do not block the generation event loop.
- [ ] MCP clients and child processes return to baseline after stress tests.
- [ ] Ten-trajectory smoke and 100-trajectory pilot runs pass.
- [ ] An interrupted pilot resumes to exactly the requested unique count.
- [ ] Generation yield is at least 80% or an explicitly approved revised gate is documented.
- [ ] Dataset splits have no seed leakage and load in LlamaFactory.
- [ ] A student LoRA smoke job trains and resumes.
- [ ] The full adapter trains or resumes across sessions without data loss.
- [ ] Teacher and student executable evaluations complete on held-out seeds.
- [ ] Student meets the recorded release gates.
- [ ] Both marimo notebooks reproduce the CLI workflow without hidden state.
- [ ] The runbook has been followed once from a clean session.
- [ ] No secrets or generated large artifacts are tracked by Git.
- [ ] The final run report is complete.

## 19. Suggested commit sequence

Keep changes reviewable in this order:

1. `test: establish Python 3.12 and mini test scaffolding`
2. `fix: repair MCP registration and generator initialization`
3. `refactor: add non-blocking MCP APIs and lifecycle tests`
4. `feat: add typed mini catalog and configuration`
5. `feat: add local cached embeddings and mini graph builder`
6. `feat: add MoLab doctor and vLLM serving profile`
7. `feat: add resumable mini trajectory synthesis`
8. `feat: add deterministic dataset conversion and manifests`
9. `feat: add mini LoRA training profile`
10. `feat: add executable teacher and student evaluation`
11. `docs: add MoLab notebooks, runbook, and benchmark report`
12. `perf: tune measured mini pipeline bottlenecks`

Each commit must leave existing public entry points functional and include its relevant tests.

## 20. Agent handoff protocol

An agent beginning work should:

1. Read this entire document and the repository `README.md`.
2. Check `git status` and preserve unrelated user changes.
3. Identify the first unchecked task whose prerequisites are complete.
4. Run the nearest existing tests before editing.
5. Implement only that phase or a clearly bounded subset.
6. Add or update tests in the same change.
7. Run the phase exit criteria.
8. Update the checklist and decision log below with evidence.
9. Record commands and results in the handoff note.
10. Do not mark a phase complete when a required GPU/MoLab test has only been mocked locally.

Handoff notes must state:

- Files changed.
- Behavior changed.
- Tests run and exact outcomes.
- Tests not run and why.
- Artifact/schema migrations, if any.
- Remaining risks.
- Exact next task.

## 21. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-08-19 | Reuse EnvFactory instead of creating a parallel mini codebase | Avoid divergence and preserve upstream capabilities |
| 2026-08-19 | Use eight fixed, offline environments totaling 55 metadata tools | Sufficient diversity with deterministic, safe execution |
| 2026-08-19 | Begin with non-conversational SFT only | Smallest end-to-end path with useful output |
| 2026-08-19 | Use Qwen3-14B BF16 teacher with Qwen3-8B fallback | Fits 96 GB VRAM; fallback protects the 32 GB host-RAM constraint |
| 2026-08-19 | Train a 4B/8B LoRA student and omit full fine-tuning/RL | Fits one GPU and 12-hour resumable sessions |
| 2026-08-19 | Use 16K teacher context and 8K student cutoff initially | Preserves concurrency and covers the smaller catalog prompts |
| 2026-08-19 | Use local CPU embeddings with a persistent cache | Removes an external embedding API dependency |
| 2026-08-19 | Preserve tool-call order | Environment state makes speculative parallel execution unsafe |
| 2026-08-19 | Keep MCP and LLM singletons import-safe, validate configuration before use, and keep run-specific query flags and RNG state off shared objects | Prevent partial initialization and cross-run interference while preserving explicit server registration |
| 2026-08-19 | Give every stateful MCP client ID its own connected FastMCP stdio client and close it explicitly | FastMCP 3.1 `Client.new()` shares stdio session state; independent child sessions are required to isolate generated servers' process-global mutable state |
| 2026-08-19 | Select the MoLab PyTorch backend from the live NVIDIA driver with `uv --torch-backend=auto`, keep vLLM out of the training environment, and capture installed versions in doctor reports | Blackwell needs CUDA 12.8 or newer and static vLLM/PyTorch pins age quickly; isolated dependency solvers prevent serving and training upgrades from breaking each other |
| 2026-08-19 | Pass vLLM authentication only through `VLLM_API_KEY` and namespace launcher settings under `MOLAB_VLLM_*` | Keeps secrets out of process arguments and avoids collisions with vLLM's own internal networking environment variables |
| 2026-08-19 | Rebase all configured graph and cache outputs when `ENVFACTORY_MINI_ARTIFACT_ROOT` changes | A partial override made `artifact_root` disagree with its graph paths and caused the graph builder's containment check to reject an otherwise supported persistent-artifact deployment |
| 2026-08-19 | Treat immutable completed trajectory files as the resume source of truth and reconcile the atomic run manifest from them | A session can end after a completed file lands but before the next manifest checkpoint; disk reconciliation prevents duplicate inference and preserves exact unique-seed completion |
| 2026-08-20 | Make the student tokenizer identity and server-imbalance override explicit dataset configuration | Dataset conversion must use the exact tokenizer/chat template without hidden model defaults, and accepting a distribution gate exception must be a deliberate recorded setting |
| 2026-08-20 | Render pinned LlamaFactory 0.9.4 profiles from one shared LoRA template, keep checkpoints under the run, and atomically promote only adapter files after verification | Run-local paths prevent cross-run overwrite, `lora_target: all` covers Qwen attention/MLP linear layers in the verified syntax, and adapter promotion preserves resumable trainer state without merging base weights |
| 2026-08-20 | Keep marimo cells as UI-only CLI orchestrators and persist named subprocess state beneath the artifact root | Reactive reruns cannot replace live jobs, session restarts can rediscover PID/log state, and lifecycle/redaction/path safety remain independently testable without marimo or a GPU |
| 2026-08-20 | Bound MCP catalog registration to two concurrent stdio process trees, with an operational hard cap of four | Five-run before/after measurement reduced peak descendants from 35 to 6 and maximum sampled process-tree RSS from 1.82 GB to 234 MB; the 1.50-second median startup cost is acceptable under the four-CPU/32-GB MoLab constraint |
| 2026-08-21 | Recover stale run locks via atomic rename-and-reverify, treat access-denied PID probes as live, add `--new-run`, enforce disk-pressure thresholds at runtime, add a pilot server-failure circuit breaker, and emit graph-stage event logs | Conformance review found the stale-recovery unlink could delete a freshly replaced lock, access-denied probes misclassified live PIDs as dead on Windows, resume lacked the documented compatibility override, disk thresholds existed only on paper, and the 20% pilot failure-rate abort was unimplemented |
| 2026-08-21 | Measure one vLLM server configuration per `serving-context` invocation and replay real completed-trajectory prompts at deterministic length percentiles with pinned decoding | Context, `max_num_seqs`, and GPU utilization are launch settings that require a restart between measurements; real percentile-spanned prompts make 8K-vs-16K results distribution-relevant, and pinned temperature/seed/thinking-off keeps latency deltas attributable to the server configuration |

Future decision changes must append a row; do not rewrite history.
