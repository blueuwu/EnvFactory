# EnvFactory on MoLab — Master Implementation & Reproduction Plan

**Status:** execution-ready master specification  
**Target:** marimo MoLab, 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM  
**Primary objective:** build a reliable, restart-safe MoLab implementation of EnvFactory that reuses the upstream repository where useful, fixes its correctness/runtime issues, exercises the paper’s core mechanisms end to end, fine-tunes Qwen3 models, and performs executable-environment RL.  
**Audience:** a coding agent or engineer starting cold. This document is intended to be sufficient to decide what to do next, what files to touch, what tests to run, what outputs to write, and what counts as success.

---

# 0. Executive decision

We are **not** doing either extreme:

```text
EXTREME A
rewrite EnvFactory from scratch

EXTREME B
clone upstream, run two scripts, call it a reproduction
```

The project uses a third approach:

```text
PINNED UPSTREAM ENVFACTORY
        +
SMALL AUDITABLE PATCH SET
        +
MOLAB ORCHESTRATION / RESUME / SERVING LAYER
        +
INDEPENDENT VALIDATION OF CORE MECHANISMS
        +
NEW / REGENERATED ENVIRONMENTS
        +
FULL TOOLGRAPH / QUERYGEN EXPERIMENTS
        +
QWEN3 SFT
        +
EXECUTABLE GRPO
        +
ABLATIONS / HELD-OUT EVALUATION
```

This gives us the best tradeoff.

We **reuse** upstream implementation when it is already useful:

- metadata generation
- EnvGen orchestration
- generated MCP environment format
- validation/revision machinery
- ToolGraph types and current algorithms
- QueryGen presets and existing dialogue machinery
- data converters
- `Gen.validate()` sandbox execution
- LLaMA-Factory data format
- released environments as initial infrastructure fixtures

We **patch or extend** upstream when required:

- broken registration API calls
- unsafe/global configuration mutation
- global RNG behavior
- blocking async MCP paths
- session/process leaks
- single-GPU serving scripts
- Blackwell-incompatible package pins
- graph edge caching and candidate caps
- per-seed generation persistence
- provenance/manifests
- deterministic environment checks
- VeRL/GRPO integration

We **independently demonstrate** the paper-critical ideas rather than merely inheriting artifacts:

1. regenerate at least three known environments from their sketches/metadata without copying the released implementation;
2. synthesize at least five new/modified environments and pass executable validation;
3. audit the dependency ToolGraph at parameter level;
4. compare TopologySampler with RandomWalkSampler;
5. compare semantic graph edges with semantic+LLM logical refinement;
6. generate non-conversational and conversational trajectories;
7. verify pass-k candidates in isolated stateful MCP sessions;
8. fine-tune Qwen3-1.7B with full SFT;
9. scale SFT to Qwen3-4B if the memory gate passes;
10. implement executable state-based reward;
11. run GRPO on Qwen3-1.7B;
12. evaluate Base vs SFT vs SFT+RL on held-out environments.

The project is allowed to reuse upstream code. The project is **not** allowed to rely only on upstream-generated environments/datasets/checkpoints for its central result.

---

# 1. What counts as success

There are four levels. They are cumulative.

## Level 0 — Infrastructure proof

A fresh MoLab session can:

- bootstrap the repository;
- install compatible dependencies;
- pass the runtime doctor;
- launch a local Qwen model;
- start MCP servers;
- load → mutate → save scenario state;
- close all clients/processes;
- stop the model server and recover VRAM;
- resume a deliberately interrupted test run.

This is the first hard gate.

## Level 1 — Pipeline proof

Using the audited small profile:

```text
MCP environments
→ metadata/catalog audit
→ dependency graph
→ topology chains
→ QueryGen
→ processed SFT dataset
→ Qwen3-1.7B SFT
→ executable held-out evaluation
```

No RL required yet.

## Level 2 — Research-shaped reproduction

Add:

- EnvGen regeneration proof;
- new environment synthesis;
- full semantic + LLM ToolGraph;
- topology-vs-random ablation;
- conversational QueryGen;
- query refinement ablation;
- full Qwen3-1.7B SFT;
- Qwen3-4B full SFT if feasible;
- held-out environment generalization;
- comparison with released EnvFactory model(s) where practical.

## Level 3 — Executable RL reproduction

Add:

```text
RL dataset
→ executable trajectory/state reward
→ single-GPU GRPO
→ SFT+RL held-out eval
```

Primary RL target:

```text
Qwen3-1.7B full policy
```

Stretch target:

```text
Qwen3-8B LoRA policy
```

Do **not** make Qwen3-4B full GRPO the primary milestone on one 96 GB GPU.

---

# 2. Scope tiers

Use a tier in every run config.

## Tier A — pipeline works

Goal: establish a reliable system.

Target:

```text
8-environment golden catalog
then 20-environment small profile

20–300 generated trajectories during development
~300–1,000 SFT step examples
Qwen3-1.7B full SFT
no RL required
```

Tier A is complete only if restart/resume and executable eval work.

## Tier B — paper-shaped result

Target:

```text
20–60+ training environments
held-out environment split defined before data generation
~3,000–5,000 generated trajectories across presets
~5k–15k usable SFT examples
~1k+ RL turn examples if generation budget allows

Qwen3-1.7B full SFT
Qwen3-4B full SFT
Qwen3-1.7B GRPO
mandatory ablations
```

This should be the primary public project target.

## Tier C — scale / closer fidelity

Stretch:

- 60–85 environments;
- released-dataset-scale SFT/RL volumes;
- 8B LoRA/QLoRA;
- multi-turn live-tool GRPO;
- public benchmarks such as BFCL/τ-bench if environment and session budget permit.

Tier C must not block completion of Tier B.

---

# 3. Reproduction modes

Every run records:

```yaml
reproduction_mode:
  molab_pragmatic | repo_current | paper_shaped
```

## `molab_pragmatic`

Default.

- prioritize correctness and single-GPU viability;
- use current Blackwell-compatible serving/training packages;
- topology recursion depth 3 initially;
- full Qwen3-1.7B SFT;
- 4B full SFT after memory proof;
- pass-k 2 for smoke, 4 for serious generation;
- GRPO n=4 for smoke, n=8 after stability.

## `repo_current`

Match checked-in upstream behavior where sensible.

Examples from the audited code:

```text
TopologySampler max_servers = 3
TopologySampler max_recursion_depth = 5
optional parameter skip probability = 0.60
already-satisfiable parameter skip probability = 0.90
ToolGraph semantic edge threshold = 0.85
merge parameter threshold = 0.92
checked-in SFT YAML epochs = 1
```

## `paper_shaped`

Prefer paper-described choices.

Important known divergence to preserve explicitly:

```text
topology recursion:
paper ≈ 3
current repo = 5

SFT:
paper = 3 epochs
checked-in repo YAML = 1 epoch

RL initialization:
paper starts RL from SFT epoch-1 checkpoint
```

Never silently mix modes.

Every result table must include `reproduction_mode`.

---

# 4. MoLab design constraints

Treat these as hard until `doctor` verifies otherwise:

```text
GPU: 1× RTX PRO 6000 Blackwell, 96 GB
GPU architecture: SM120
CPU: 4
host RAM: 32 GB
session maximum: ~12 hours
idle limit: ~90 minutes
notebook format: marimo .py
persistent storage: limited / verify every session
```

Consequences:

1. **12-hour wall clock is the primary architecture constraint.**
2. Every expensive stage must be restart-safe.
3. Every unit of generation must become an artifact as soon as it succeeds.
4. Model serving and ordinary SFT run in separate GPU ownership phases.
5. RL owns the GPU itself and may colocate rollout + actor.
6. Host RAM is more likely to fail from MCP process proliferation than VRAM is during generation.
7. CPU concurrency begins low and is increased only from measurements.
8. Old vLLM/SGLang pins from upstream must not be trusted on SM120.
9. No notebook cell contains the implementation of a multi-hour job.
10. GitHub is source of truth for code; a remote artifact store is source of truth for large outputs.

---

# 5. Repository strategy

Do not vendor or rewrite the entire upstream repository.

Create:

```text
envfactory-molab/
```

and pin upstream under:

```text
third_party/EnvFactory/
```

Either:

- git submodule, or
- clone-at-bootstrap pinned to a recorded commit.

The default design is clone-at-bootstrap plus auditable patches.

## 5.1 Recommended tree

```text
envfactory-molab/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   ├── profile.yaml
│   ├── split.json
│   ├── profiles/
│   │   ├── golden8.yaml
│   │   ├── small20.yaml
│   │   ├── medium.yaml
│   │   └── full.yaml
│   ├── serve_vllm.sh
│   ├── serve_sglang.sh
│   ├── sft_qwen3_1_7b_full.yaml
│   ├── sft_qwen3_4b_full.yaml
│   ├── sft_qwen3_8b_lora.yaml
│   ├── grpo_qwen3_1_7b.yaml
│   └── graph.yaml
├── notebooks/
│   ├── nb00_bootstrap.py
│   ├── nb01_core_audit.py
│   ├── nb02_env_synthesis.py
│   ├── nb03_tool_graph.py
│   ├── nb04_serve_model.py
│   ├── nb05_trajectory_synth.py
│   ├── nb06_data_process.py
│   ├── nb07_sft.py
│   ├── nb08_rl_grpo.py
│   └── nb09_eval_report.py
├── emlab/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── doctor.py
│   ├── config.py
│   ├── catalog.py
│   ├── profiles.py
│   ├── serving.py
│   ├── jobs.py
│   ├── checkpoint.py
│   ├── manifest.py
│   ├── artifacts.py
│   ├── sync.py
│   ├── logging.py
│   ├── graph_audit.py
│   ├── sampler_audit.py
│   ├── env_health.py
│   ├── query_audit.py
│   ├── reward.py
│   ├── evaluation.py
│   ├── verl_glue/
│   │   ├── __init__.py
│   │   ├── tool_adapter.py
│   │   ├── single_turn.py
│   │   └── multi_turn.py
│   ├── patches/
│   │   ├── P01_*.patch
│   │   └── ...
│   └── ui.py
├── scripts/
│   ├── bootstrap.sh
│   ├── run_env_gen.py
│   ├── run_graph.py
│   ├── run_query_gen.py
│   ├── run_data_process.py
│   ├── run_sft.sh
│   ├── run_rl.sh
│   ├── evaluate.py
│   └── stop_gpu_jobs.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   ├── golden/
│   └── gpu/
├── docs/
│   ├── MASTER_IMPLEMENTATION_PLAN.md
│   ├── MOLAB_RUNBOOK.md
│   ├── DECISIONS.md
│   ├── REPRODUCTION_NOTES.md
│   └── REPORT_TEMPLATE.md
├── artifacts/                  # gitignored
└── third_party/
    └── EnvFactory/
```

## 5.2 Rule: notebooks orchestrate, modules implement

Marimo notebooks:

- configure;
- display;
- launch subprocesses;
- tail logs;
- inspect artifacts;
- plot metrics.

They must not contain unique production logic.

Anything that matters to correctness belongs in `emlab/`, a CLI script, or a tested patch.

---

# 6. Environment / dependency strategy

Use **three separate virtual environments**.

## `.venv-gen`

Purpose:

- EnvGen
- MCP client/server management
- ToolGraph
- QueryGen
- data processing

Contains:

- upstream EnvFactory dependencies
- FastMCP / MCP
- networkx
- litellm/openai clients
- pandas/pyarrow
- sentence-transformers if local CPU embeddings are used
- no model-serving stack unless strictly necessary

## `.venv-serve`

Purpose:

- vLLM or SGLang only

Contains:

- Blackwell-compatible PyTorch
- current vLLM **or** current SGLang
- compatible attention kernels

This environment is allowed to diverge sharply from upstream serving pins.

## `.venv-train`

Purpose:

- LLaMA-Factory / TRL
- PEFT
- VeRL
- training PyTorch stack

The generation process talks to the server over OpenAI-compatible HTTP, so sharing a Python/CUDA environment between generator and server creates dependency pain for almost no benefit.

---

# 7. Configuration contract

Create a single typed root config.

Suggested shape:

```yaml
schema_version: 1
reproduction_mode: molab_pragmatic
tier: A
run_seed: 42

upstream:
  repo: LARK-AI-Lab/EnvFactory
  commit: "<PINNED_SHA>"

paths:
  workspace: /root/work
  artifact_root: /root/work/artifacts
  hf_cache: /root/work/hf_cache

profile:
  name: golden8
  mcp_config: configs/generated/mcp_server.golden8.json

graph:
  embedding_backend: bge_m3
  build_edge_threshold: 0.85
  merge_param_threshold: 0.92
  enable_parameter_merge: false
  enable_llm_edges: true
  edge_candidate_top_k: 20
  max_servers: 3
  recursion_depth: 3

generation:
  preset: SFT_NON_CONV
  target: 100
  pass_k: 4
  max_nodes: 10
  workers: 4
  max_attempts_per_seed: 2

teacher:
  engine: vllm
  model: "<resolved>"
  max_model_len: 16384
  gpu_memory_utilization: 0.85

sft:
  model: Qwen/Qwen3-1.7B
  full_finetune: true
  cutoff_len: 8192

rl:
  enabled: false
  rollout_n: 4
```

Unknown structural keys must fail fast.

Secrets do not belong in this YAML.

---

# 8. Artifact, run, and provenance contract

This is mandatory from day one.

## 8.1 Run IDs

Every invocation belongs to:

```text
YYYYMMDDTHHMMSSZ-<gitshortsha>-<config8>
```

## 8.2 Artifact tree

```text
artifacts/
├── cache/
│   ├── embeddings.sqlite3
│   ├── graph_edges.sqlite3
│   └── user_provided.sqlite3
├── graph/
│   ├── graph.pkl
│   ├── graph.json
│   ├── graph.graphml
│   └── graph_manifest.json
├── env_health/
├── runs/<run_id>/
│   ├── run_manifest.json
│   ├── resolved_config.yaml
│   ├── environment.json
│   ├── logs/
│   │   └── events.jsonl
│   ├── sampled_chains/
│   ├── trajectories/
│   │   ├── completed/
│   │   └── failed/
│   ├── datasets/
│   ├── training/
│   └── evaluation/
└── state/
```

## 8.3 Atomicity rules

1. Mutable JSON writes:
   - temp file in same directory;
   - flush/fsync;
   - `os.replace`.

2. Completed trajectory files are immutable.

3. A retry never overwrites the previous failure record without preserving it.

4. JSONL logs:
   - append;
   - flush every record;
   - reader tolerates a truncated last line.

5. A run cannot resume if compatibility hashes changed, unless explicitly started as a new run.

## 8.4 Manifest fields

At minimum:

```text
schema_version
run_id
state
git_commit
git_dirty
upstream_commit
config_sha256
catalog_sha256
graph_sha256
python version
torch version
CUDA version
vLLM/SGLang version
VeRL version
GPU name/VRAM
host RAM
CPU count
provider/model revisions
generation seeds
completed/failed/pending
attempts
failure summary
elapsed time
throughput
peak VRAM
peak host RSS
dataset counts
training summary
evaluation summary
```

Never store secrets.

## 8.5 Compatibility hashes

Hash:

- resolved structural config;
- selected environment metadata;
- selected environment tool files;
- graph thresholds/model;
- upstream commit;
- teacher model identifier;
- student model identifier.

A changed compatibility hash means a new experiment.

---

# 9. Stage checkpoint contract

Every long stage exposes:

```python
class StageState:
    stage: str
    run_id: str
    profile: str
    units_total: list[str]
    units_done: list[str]
    units_failed: list[str]
    artifacts: list[str]
    resolved_versions: dict
```

Status for individual work units:

```text
PENDING
RUNNING
SUCCESS
FAILED_RETRYABLE
FAILED_FINAL
```

On startup:

- stale `RUNNING` → `PENDING`;
- `SUCCESS` → skip;
- retryable failures → retry up to config budget;
- final failures → leave quarantined.

---

# 10. Storage / remote survival strategy

Code and configs:

```text
GitHub
```

Large generated artifacts:

```text
Hugging Face dataset/model repositories
or another user-configured artifact store
```

Recommended:

| Artifact | Destination |
|---|---|
| code/config/notebooks | GitHub |
| generated env code/metadata/checkpoints | HF dataset repo |
| graph/cache | HF dataset repo |
| trajectories | HF dataset repo in compressed shards |
| processed datasets | HF dataset repo |
| model checkpoints | private HF model repo |
| logs | compressed HF dataset repo |

Do not trust sandbox storage alone.

Sync is allowed to lag local progress, but successful units should be safely on local disk immediately.

---

# 11. Secrets policy

Inference servers bind to:

```text
127.0.0.1
```

Keys:

- only environment variables / protected local `.env`;
- `.env` mode `0600`;
- never committed;
- never placed in notebook source;
- never printed;
- never included in manifests;
- never included in command arguments when avoidable.

All logs run through a secret scrubber before remote upload.

---

# 12. Marimo authoring rules

1. One definition per variable name across the reactive notebook where practical.
2. Do not use `asyncio.run()` in a marimo cell.
3. Use top-level `await` or subprocesses.
4. Every expensive action is behind an explicit UI trigger.
5. Reactive changes must not relaunch a training/generation job.
6. Long jobs execute in a subprocess.
7. Notebook UI tails logs and reads manifests.
8. Every notebook begins with session state:
   - hours remaining;
   - GPU free;
   - RAM free;
   - disk free;
   - child processes;
   - active GPU lock;
   - stage progress.
9. Every notebook action has a CLI equivalent in `docs/MOLAB_RUNBOOK.md`.
10. Notebook code is never the only place a workflow exists.

---

# 13. Phase 0 — cold bootstrap and runtime doctor

Goal:

> turn a cold MoLab sandbox into a verified EnvFactory workspace.

## 13.1 Bootstrap tasks

- [ ] create workspace directories;
- [ ] clone this repo;
- [ ] clone upstream at pinned SHA;
- [ ] check `git status`;
- [ ] apply patch series idempotently;
- [ ] create `.venv-gen`, `.venv-serve`, `.venv-train`;
- [ ] install dependencies;
- [ ] run Python compile checks;
- [ ] record resolved packages;
- [ ] verify HF/artifact credentials if configured;
- [ ] verify writeable artifact directories.

## 13.2 Doctor checks

Output must include:

- Python executable/version;
- OS/architecture;
- CPU count;
- host RAM;
- free disk;
- GPU name;
- driver;
- CUDA runtime;
- compute capability;
- free/total VRAM;
- torch version;
- serving engine import/version;
- train environment import/version;
- upstream commit;
- patch list;
- required path existence;
- model endpoint health if requested;
- secret presence as booleans only.

Expected GPU capability:

```python
torch.cuda.get_device_capability() == (12, 0)
```

but doctor should report rather than assume when first run.

## 13.3 Smoke tests

T1:
```text
upstream imports
```

T2:
```text
FastMCP client and MCP server SDK coexist
```

T3:
```text
Calculator starts and answers one call
```

T4:
```text
Calendar load → mutate → save records mutation
```

T5:
```text
two Calendar sessions loaded with different states do not leak
```

T6:
```text
GPU matmul works
```

T7:
```text
one configured provider completion works
```

T8:
```text
artifact remote write/delete works if remote sync enabled
```

T9:
```text
model server starts and stops; VRAM returns near baseline
```

T10:
```text
MCP child-process count returns to baseline after repeated session creation/close
```

Exit criteria:

```text
all mandatory tests green
```

---

# 14. Phase 1 — patch and harden upstream core

Do this **before** serious generation.

The Mini audit identified real correctness issues that must become first-class patches.

## 14.1 Registration correctness

Patch:

- implement/retain public `register_mcp_server(...)`;
- route old invalid `register_MCP_server(...)` references to it;
- optionally retain deprecated compatibility alias.

Tests:

```text
test_register_mcp_server_sync_wrapper
test_envgen_registration_reaches_server_registration
```

## 14.2 MCP auto-init

Honor:

```text
SKIP_MCP_AUTO_INIT
```

Importing modules during tests must not unexpectedly spawn all MCP servers.

Test:

```text
test_skip_auto_init
```

## 14.3 Agent initialization

Audit `Gen.__init__()` and subclasses.

Remove duplicate `load_agents()` invocation.

Test:

```text
test_agent_initialization_occurs_once
```

## 14.4 QueryGen configuration isolation

Do not mutate shared `QueryGenConfig` instances.

Every generator must receive:

- immutable config, or
- deep copy / run-specific state.

Test:

```text
test_query_configs_are_not_mutated_across_runs
```

Two concurrent QueryGen instances must not change each other.

## 14.5 RNG isolation

Never call global:

```python
random.seed(...)
```

inside production graph/query generation.

Use:

```python
rng = random.Random(seed)
```

Propagate explicit RNG through samplers.

Tests:

```text
same graph + same config + same seed = identical chain
concurrent calls do not perturb output
different seed changes output distribution
```

## 14.6 Non-blocking async MCP API

The MCP manager may retain its background event loop, but async generation must not block on `Future.result()`.

Add/standardize:

```python
async def aload_scenario(...)
async def acall_tool(...)
async def asave_all_scenarios(...)
async def aclose_client(...)
```

Implementation:

- submit coroutine via `asyncio.run_coroutine_threadsafe`;
- await with `asyncio.wrap_future`;
- all actual get/create work runs on manager loop;
- sync wrappers remain for old paths;
- async code never calls sync manager methods;
- registration/load/call/save/close each has its own timeout;
- timeout cancels and cleans affected session;
- shutdown is idempotent.

## 14.7 Bounded MCP concurrency

Add configurable limits:

```text
MCP_MAX_CONCURRENT_CLIENTS
MCP_MAX_CONCURRENT_REGISTRATIONS
MCP_MAX_CONCURRENT_SCENARIO_LOADS
```

Start conservatively.

Do not parallelize sequential tool calls from the same model response unless independence has been proven.

## 14.8 Lifecycle stress tests

Required:

- ticker coroutine continues during deliberately slow MCP call;
- four concurrent isolated Calculator/Calendar sessions;
- 100 short-lived sessions return tracked client count to baseline;
- child process count returns to baseline;
- timed-out call does not poison next session;
- shutdown can run twice;
- failed registration closes partial connections.

Exit criteria:

```text
no Future.result() in async generation path
no tracked client leak
no child-process growth after stress loop
```

---

# 15. Master patch set

Keep every patch separate and reversible.

Suggested patches:

| ID | Area | Purpose |
|---|---|---|
| P01 | serving SGLang | single-GPU, remove cluster/offline assumptions |
| P02 | serving vLLM | single-GPU, current Blackwell setup, prefix caching |
| P03 | setup/dependencies | stop forcing stale serving pins |
| P04 | MCP registration | correct public registration API |
| P05 | MCP auto-init | respect skip flag |
| P06 | Gen initialization | remove duplicate agent init |
| P07 | QueryGen config | remove shared mutation |
| P08 | sampler RNG | explicit RNG |
| P09 | MCP async API | non-blocking wrappers |
| P10 | MCP lifecycle | bounded concurrency, close_all, timeouts, process metrics |
| P11 | provider config | env-configurable providers if needed |
| P12 | ToolGraph | edge candidate top-k |
| P13 | ToolGraph | persistent edge cache hook |
| P14 | QueryGen | deterministic seed→output mapping and `.error.json` |
| P15 | data process | provenance sidecar |
| P16 | deterministic envs | fix/exclude unseeded random/time behavior where needed |
| P17 | serialization | JSON/GraphML backup for graph |
| P18 | logging | structured/scrubbed error and usage logging |

Each patch:

- includes rationale;
- has a test;
- appears in `patches_applied.json`;
- is pinned to upstream SHA;
- must apply idempotently.

---

# 16. Phase 2 — catalog, profiles, and train/eval split

Use progressive environment profiles.

## 16.1 `golden8`

Use the audited eight-environment set:

```text
Calculator
Calendar
CampusCard
HotelBooking
MovieRecommender
Retail
Telecom
Weather
```

Purpose:

- fast core correctness;
- diverse state patterns;
- known metadata/tool counts;
- manageable system prompts;
- no need to boot 85 processes.

Expected metadata tool total from the audited plan:

```text
55
```

The catalog check must verify actual current count rather than blindly trusting this number.

## 16.2 `small20`

After golden8:

Select ~20 environments spanning:

- stateless computation;
- lookup/search;
- hierarchical data;
- account/balance state;
- booking;
- CRUD;
- catalog/cart/order;
- multi-server dependencies.

Only include environments that pass `env_health`.

## 16.3 `medium`

~45 environments.

## 16.4 `full`

all verified registered environments.

## 16.5 Catalog validator

Check:

- metadata exists;
- implementation exists;
- module imports;
- class/server name consistency;
- tool names match metadata;
- `load_scenario` and `save_scenario` exist;
- lifecycle tools excluded from model-visible set;
- no duplicate server names;
- server names obey routing convention;
- profile paths are repo-relative and stable.

## 16.6 Held-out environments

For Tier B, define held-out environments **before generating training data**.

Persist:

```text
configs/split.json
```

Never mutate this split after training data is generated.

Produce:

```text
mcp_server.<profile>.train.json
mcp_server.<profile>.heldout.json
```

Training generation never sees held-out environments.

Evaluation generation never writes into training folders.

This is the cleanest environment-generalization test.

---

# 17. Phase 3 — EnvGen: reuse + proof

This project does not need to regenerate all 85 existing environments.

It **does** need to prove environment synthesis works.

Use two tracks.

## Track A — reuse upstream environments

Use released environments for:

- bootstrap;
- graph debugging;
- QueryGen infrastructure;
- early SFT.

This is allowed and encouraged.

## Track B — EnvGen proof

Mandatory Tier B gate:

### Known regeneration

Pick at least three heterogeneous known environments, e.g.:

```text
Calendar
Retail
HotelBooking
```

For each:

```text
existing sketch
→ metadata generation
→ executable env generation
→ validation/revision
→ independent validation
```

Do not copy the released `envs/tools/<Name>.py` into generated output.

Compare behavior, not source-code textual equality.

### New/modified environment synthesis

Create at least five new or meaningfully modified environments.

Examples:

- task tracker;
- event ticketing;
- package delivery;
- cloud resource inventory;
- restaurant reservation;
- expense management;
- CRM.

Avoid tools requiring live external network access during execution.

Goal:

```text
schema sketch
→ metadata
→ executable stateful MCP
→ validation
→ repair
→ verified catalog entry
```

## 17.1 Metadata requirements

Metadata must specify:

- class name;
- service description;
- tools;
- tool descriptions;
- JSON-like input schemas;
- required fields;
- output schemas;
- mutation semantics;
- understandable ID descriptions.

Lint:

- duplicate tool name;
- required fields missing from properties;
- missing descriptions;
- invalid output shape;
- ambiguous opaque identifier behavior;
- inconsistent names.

## 17.2 Environment implementation contract

Every generated stateful env should have:

```python
class Scenario(BaseModel):
    ...

class API:
    def load_scenario(...)
    def save_scenario(...)
    ...
```

MCP wrappers delegate to the API.

All mutable state must be captured by scenario serialization.

## 17.3 Independent validation

Do not rely solely on EnvGen's own validator.

Every generated environment must pass:

1. `py_compile`;
2. module import;
3. MCP boot/list tools;
4. metadata/tool parity;
5. lifecycle tools present;
6. scenario save→load→save equality;
7. read operation tests;
8. create/update/delete where supported;
9. invalid identifier path;
10. deterministic same-state/same-call result;
11. session isolation;
12. no network egress at tool-call time;
13. process cleanup.

For mutating calls:

```text
load S0
call T(args)
save S1
assert expected state transition
```

## 17.4 Validation-repair loop

Upstream's revision mechanism may be reused.

Store every revision:

```text
r0 code
validation_r0.json
repair prompt hash
r1 code
validation_r1.json
...
```

Recommended maximum revisions:

```text
3 during ordinary run
5 during debugging / important proof env
```

Failed environments are quarantined and excluded from profiles.

## 17.5 Env health artifact

Create:

```text
artifacts/env_health/<profile>.json
```

Fields:

```text
env
hash
metadata parity
scenario roundtrip
determinism
isolation
network egress
tests passed
tests failed
status
```

Tier A requirement:

```text
golden8 100% or documented upstream defect
```

Tier B profile requirement:

```text
≥95% selected env health pass
```

---

# 18. Phase 4 — ToolGraph

Reuse upstream graph types and build path, but add audits/caches and make graph behavior explicit.

## 18.1 Graph semantics

Model:

```text
Tool nodes
Parameter nodes

Input Parameter → Tool
Tool → Output Parameter

semantic/logical dependency edges
```

A tool sequence is valid only if required inputs can be:

- supplied by user;
- satisfied by defaults/optional semantics;
- produced by an earlier tool.

## 18.2 Parameter identity

Retain upstream-compatible names, but internally preserve unique path identity:

```text
server/tool/input|output/schema_path
```

Bare-name identity can collide across unrelated concepts.

## 18.3 Schema flattening tests

Test:

- scalar;
- array scalar;
- object;
- nested object;
- array of objects;
- required flags;
- original schema path.

## 18.4 Embeddings

Primary paper-shaped setting:

```text
BAAI/bge-m3
```

Cache embeddings.

For parameter pair similarity, preserve current known weighting where applicable:

```text
0.6 * cosine(name/type embedding)
+
0.4 * cosine(description embedding)
```

Default semantic edge threshold:

```text
0.85
```

Parameter merge threshold if merge enabled:

```text
0.92
```

During Tier A, parameter merging can remain disabled.

## 18.5 Embedding abstraction

Implement:

```python
class EmbeddingBackend(Protocol):
    @property
    def identity(self) -> str: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
```

Backends:

1. BGE-M3 local/API;
2. small CPU sentence-transformer fallback for quick smoke;
3. HTTP backend if already present upstream.

Production experiment logs which backend was used.

## 18.6 Embedding cache

SQLite/content-addressed:

```text
backend identity
+
sha256(exact input text)
```

Requirements:

- dedupe texts before inference;
- restore original order;
- finite-value validation;
- normalized float32 storage;
- cache-hit metrics.

## 18.7 Semantic output→input dependencies

For candidate source output and destination input:

- compute similarity;
- if threshold passes, create dependency relation;
- record edge provenance.

Provenance:

```text
SEMANTIC
LLM
MANUAL
```

## 18.8 LLM logical refinement

Enable for the research run.

Input per server:

- tool names;
- descriptions;
- inputs/outputs;
- current adjacency.

Output:

```json
{"adjacency_map": {...}}
```

Validate tool names before applying.

Default policy:

- LLM may add missing logical edges;
- pruning is optional and must be separately logged.

## 18.9 Cost control

LLM edge adjudication can explode.

Mandatory:

- edge candidate top-k;
- SQLite edge cache;
- graph build metrics.

Record:

```text
candidate pairs
embedding-filtered pairs
LLM calls
accepted edges
cache hits
wall clock
tokens
cost if API
```

## 18.10 User-provided parameter classification

Classify inputs as:

```text
USER_PROVIDED
SYSTEM_DERIVED
UNCERTAIN
```

Examples:

User-provided:

- destination city;
- date;
- desired quantity;
- search query.

System-derived:

- opaque event ID;
- order ID;
- record handle.

Classification artifact must retain prompt/model/cache key.

Failed parsing → `UNCERTAIN`, never silently `False`.

## 18.11 Fillability audit

Every required parameter must be:

- user-provided, or
- reachable from producer output.

Report:

```text
required parameters total
user-provided
tool-produced
both
unfillable
```

Unfillable parameters block research-grade graph acceptance.

Do not automatically mark them user-provided merely to make the graph pass.

## 18.12 Serialization

Save:

```text
graph.pkl
graph.json
graph.graphml
graph_manifest.json
```

Never depend only on pickle across environments.

---

# 19. Phase 4b — topology sampler and mandatory ablation

This is a central experiment, not a utility test.

Implement/retain:

```text
RandomWalkSampler
TopologySampler
```

## 19.1 Topology sampling invariants

Before consumer tool enters sequence, required inputs should already be satisfiable.

Backward dependency filling recursively chooses producer tools.

Current repo-shaped defaults identified in prior audit:

```text
max_servers = 3
max_recursion_depth = 5
optional_skip_probability = 0.60
already_valid_skip_probability = 0.90
```

Paper-shaped mode:

```text
max_recursion_depth = 3
```

Start MoLab pragmatic mode at depth 3.

## 19.2 Determinism

Sampler accepts explicit `random.Random`.

Persist:

```text
seed
graph hash
start node
sampler mode
sampler config
selected chain
```

## 19.3 Sampler tests

- required opaque ID adds producer;
- user-provided field does not force producer;
- optional arg may be omitted;
- producer precedes consumer;
- max depth respected;
- max server count respected;
- no duplicate tool unless explicitly allowed;
- impossible dependency surfaces clearly;
- same seed is deterministic.

## 19.4 Mandatory Random vs Topology experiment

Use the **same 100–500 start seeds**.

Compare:

```text
chain generation success
unsatisfied required parameter count
executable-chain rate
average tool count
average server count
dependency depth
QueryGen success rate
final trajectory validation rate
```

This result belongs in the final README/report.

Tier B does not complete without it.

---

# 20. Phase 5 — local model serving

Default engine order:

```text
vLLM first
SGLang second
API-only fallback
```

Reason:

- single-GPU Blackwell compatibility is more important than historical pin fidelity;
- generation uses OpenAI-compatible HTTP either way.

## 20.1 Version rule

Do not pin to the old upstream serving versions merely for fidelity.

At each bootstrap:

- use a known-working lock if already established;
- otherwise resolve current compatible stack;
- run smoke;
- freeze exact versions after success.

## 20.2 Teacher candidates

Suggested progression:

```text
fast iteration:
Qwen3-8B

middle:
Qwen3-14B

higher-quality local teacher:
Qwen3-30B-A3B-Thinking class model
prefer FP8 on Blackwell if available/verified
```

Do not let the entire project depend on the 30B teacher loading successfully.

API teacher is acceptable for EnvGen or some QueryGen stages if recorded.

## 20.3 Initial server settings

Start conservative:

```text
tp = 1
max context = 16k
GPU memory utilization = 0.80–0.85
max sequences = 4
```

After measurement:

```text
context 32k if needed
max sequences 8–16 if stable
GPU utilization up to ~0.9 if safe
```

## 20.4 Prefix caching

Enable when supported.

QueryGen repeatedly sends large, similar tool/schema prefixes; prefix caching can materially help.

## 20.5 Health check

Before production generation:

1. `/v1/models`;
2. one short completion;
3. one structured JSON response;
4. one Qwen tool-call response;
5. one 8k-ish prompt;
6. concurrency mini-benchmark;
7. record peak VRAM;
8. stop server;
9. confirm VRAM release.

## 20.6 GPU lock

Use:

```text
artifacts/run/gpu.lock
```

Owners:

```text
serve
sft
rl
```

Serving and SFT cannot own GPU simultaneously.

VeRL RL owns GPU end-to-end.

Stale lock recovery requires checking PID before replacing.

---

# 21. Phase 6 — QueryGen

Reuse upstream QueryGen, but put a production wrapper around it.

## 21.1 Presets

Retain:

```text
SFT_NON_CONV
SFT_CONV
RL_NON_CONV
```

Current audited shapes:

```text
SFT_NON_CONV:
  pass_k ~4
  non-conversational

SFT_CONV:
  pass_k ~4
  refinements + user interaction/user tools enabled

RL_NON_CONV:
  pass_k ~4
  filtering enabled
```

Treat exact upstream config as source of truth at pinned commit and write it into the run manifest.

## 21.2 Debug presets

For smoke:

```text
pass_k = 2
max tools = 4–6
workers = 2
target = 10–20
```

For serious:

```text
pass_k = 4
max tools = 10–15 depending profile/context
workers start = 4
```

## 21.3 Production wrapper

Create:

```bash
python -m emlab.querygen \
  --config configs/profile.yaml \
  --preset SFT_CONV \
  --target 100 \
  --resume
```

Production behavior:

- validate doctor/catalog/graph;
- generate deterministic full seed list;
- persist seeds;
- pre-sample tool chain;
- persist sampled chain before LLM inference;
- bounded queue;
- worker-local QueryGen object;
- reset run-specific state every seed;
- close all MCP sessions in `finally`;
- atomically save success;
- write structured failure;
- retry only transient failures;
- handle SIGTERM/SIGINT gracefully;
- advisory run lock;
- resource monitoring.

## 21.4 Candidate isolation

For pass-k:

```text
candidate0 starts S0
candidate1 starts S0
candidate2 starts S0
candidate3 starts S0
```

Never reuse a mutated candidate state.

This gets an explicit integration test.

## 21.5 Scenario/query quality checks

Before solve:

- scenario validates;
- every involved server can load it;
- no target is already trivially satisfied;
- hidden backend IDs do not leak into user query;
- query is not a list of tool names;
- selected tools make semantic sense.

## 21.6 Conversational behavior

Research run should exercise:

- user simulator;
- user-only tools;
- clarification;
- cross-turn state;
- implicit references.

Assistant must not call a user-only tool.

User simulator must not reveal hidden system-derived IDs unless learned during interaction.

## 21.7 Realism refinements

Track independent before/after artifacts for:

1. implicit reference;
2. action compression;
3. ambiguity introduction;
4. goal expansion.

A refinement that makes the task unanswerable is rejected.

## 21.8 Candidate evaluator

Use executable signals first:

- tool-call parse validity;
- tool execution;
- final state;
- target intent completion;
- exception count;
- redundant calls;
- interaction length.

LLM judge is supplementary, not sole correctness oracle.

## 21.9 Redundant-step filtering

Retain both:

```text
raw_steps
filtered_steps
```

Possible method:

- dependency-aware heuristic;
- optional replay-with-step-removed check.

Never destroy raw trajectory.

## 21.10 Argument masking

Retain/extend upstream `masked_arguments`.

Use masks for values that should not be exact-match-critical:

- generated IDs;
- timestamps;
- harmless limits;
- equivalent ordering controls.

Masks are part of reward and evaluation.

## 21.11 Per-trajectory schema

Store:

```json
{
  "trajectory_id": "...",
  "seed": 42,
  "reproduction_mode": "...",
  "graph_hash": "...",
  "sampler": {},
  "tool_chain": [],
  "servers": [],
  "initial_scenario": {},
  "query": "...",
  "turns": [],
  "raw_steps": [],
  "filtered_steps": [],
  "final_scenario": {},
  "candidate_scores": [],
  "masked_arguments": {},
  "success": true,
  "provider": "...",
  "model": "...",
  "usage": {},
  "latency": {}
}
```

## 21.12 Generation progression

D0:

```text
10 trajectories
golden8
pass_k 2
```

D1:

```text
100 trajectories
golden8/small20
pass_k 4
manual inspect 20
```

D2:

```text
500–1,000 trajectories
small20
mix NON_CONV + CONV
```

D3 Tier B:

```text
~3,000–5,000 trajectories
small/medium profile
SFT_NON_CONV + SFT_CONV + RL_NON_CONV
```

Do not scale if pilot generation yield is poor.

Suggested pilot gates:

```text
valid tool chain ≥85%
converter acceptance ≥80%
```

Investigate rather than lower gates silently.

---

# 22. Mandatory QueryGen ablation

Use the same environment/task seed families.

Compare:

```text
A: SFT_NON_CONV
B: SFT_CONV
C: SFT_CONV + full query refinements
```

Metrics:

- trajectory success;
- final state success;
- tool calls;
- turns;
- clarification rate;
- redundant calls;
- generated prompt length;
- student performance after matched-size training subsets if budget permits.

This makes the project more than a port.

---

# 23. Phase 7 — data processing

Reuse upstream converter where correct.

Do not duplicate it unnecessarily.

Patch around:

- provenance;
- deterministic input order;
- better manifesting;
- streaming where needed.

## 23.1 SFT unit

Preserve upstream format and semantics.

Each useful step pair becomes an Alpaca-like example with:

```text
instruction
input
output
system
history
```

Failed tool calls and removed steps remain filtered according to audited converter behavior.

## 23.2 RL unit

One interaction turn/sample.

Preserve:

```text
prompt
data_source
agent_name
ability
reward_model.ground_truth
extra_info.mcp_factory_kwargs
```

The extra environment state is critical for executable reward.

## 23.3 Split rules

Two separate kinds of holdout:

### Environment holdout

Defined by `configs/split.json`.

Used for final generalization evaluation.

### Trajectory seed split

Within training environments:

- split by trajectory before expanding into SFT steps;
- never let steps from same conversation cross train/validation.

## 23.4 Decontamination

Before training:

- normalized duplicate query hash;
- near-duplicate query check;
- same initial-scenario hash check where appropriate.

Report removals.

## 23.5 Token-length audit

With exact student tokenizer/template, measure:

```text
4k
8k
16k
```

coverage.

Do not choose cutoff purely from upstream config.

## 23.6 Dataset manifest

Include:

- source run IDs;
- environment split hash;
- graph hash;
- trajectory counts;
- step counts;
- converter config;
- tokenizer revision;
- token percentiles;
- removed/failed counts;
- server/tool distributions;
- conversational fraction;
- mutating-task fraction.

## 23.7 Data acceptance

- zero invalid tool names;
- zero malformed retained tool calls;
- zero split leakage;
- manual render of at least 20 examples;
- LLaMA-Factory dry-load succeeds;
- rerun produces deterministic manifest.

---

# 24. Phase 8 — SFT

Primary scientific target:

```text
Qwen3-1.7B full fine-tuning
```

Secondary:

```text
Qwen3-4B full fine-tuning
```

Stretch:

```text
Qwen3-8B LoRA
```

## 24.1 Why this order

1.7B lets us prove:

- data quality;
- chat template;
- full optimizer path;
- checkpoint survival;
- executable improvement.

4B is closer to paper backbone scale but materially tighter.

8B full AdamW does not sensibly fit the single-GPU memory budget without aggressive compromises.

## 24.2 SFT smoke

Before a long run:

- 20 examples;
- 20 optimizer steps;
- checkpoint;
- reload;
- resume five more steps;
- evaluate 10 tasks.

## 24.3 LR decision

Do not assume the 1.7B should automatically use `1e-5`.

Run a small controlled sweep:

```text
1e-6
3e-6
1e-5
```

Same:

- dataset subset;
- seed;
- number of updates;
- eval set.

Select on:

- validation loss;
- tool syntax;
- executable task success.

Record decision in `DECISIONS.md`.

## 24.4 Initial 1.7B config

Starting shape:

```yaml
model_name_or_path: Qwen/Qwen3-1.7B
stage: sft
do_train: true
finetuning_type: full
template: qwen3
mask_history: true

cutoff_len: 8192
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
gradient_checkpointing: true

learning_rate: <selected_from_sweep>
bf16: true

save_strategy: steps
save_steps: 100
save_total_limit: 2
overwrite_output_dir: false
```

Effective batch can be tuned after memory/throughput measurement.

## 24.5 Epoch policy

Modes:

```text
repo_current:
1 epoch

paper_shaped:
3 epochs
retain epoch-1 checkpoint for RL init

molab_pragmatic:
1 epoch pilot
then 3-epoch final if validation supports it
```

## 24.6 Qwen3-4B memory gate

Before training:

1. model load;
2. one forward/backward;
3. 10 steps;
4. peak memory;
5. save/reload.

Required headroom:

```text
prefer ≥10% VRAM free at peak
```

Fallback order:

1. batch size 1;
2. gradient checkpointing;
3. shorter sequence;
4. efficient optimizer if compatible;
5. only then LoRA if full FT is not viable.

## 24.7 8B

Use LoRA/QLoRA only as a stretch comparison.

Do not substitute 8B LoRA for the full-FT 1.7B scientific milestone.

## 24.8 Checkpoints

Save trainer state, not just model weights.

Need:

- optimizer;
- scheduler;
- RNG state;
- tokenizer/config;
- resolved YAML;
- dataset manifest hash.

Push important checkpoints remotely.

---

# 25. Phase 9 — unified executable evaluation / reward library

Implement correctness once in:

```text
emlab/reward.py
emlab/evaluation.py
```

Use it for:

- SFT evaluation;
- released-model evaluation;
- RL reward;
- final report.

## 25.1 Signals

### Format

Can the output be parsed into legal tool calls?

### Tool identity

Are selected tools valid and relevant?

### Arguments

Compare with masks.

### Execution

Do calls execute without exceptions?

### Final state

Replay:

```text
load initial state
execute predicted calls
save predicted final state
```

Compare to reference.

### Length/redundancy

Penalize unnecessary actions modestly.

## 25.2 Composite reward

Default configurable form:

```text
R =
  alpha_format * R_format
+ alpha_state  * R_state
+ alpha_traj   * R_trajectory
- alpha_len    * P_length
```

Initial practical weighting can resemble:

```text
format 0.1
state  0.7
trajectory/tool-call shaping 0.2
```

but all coefficients are config, not constants buried in code.

## 25.3 State similarity

Do not require raw exact equality for every environment.

Start with deterministic strict equality where possible.

When necessary, normalize:

- dict ordering;
- correctness-insensitive timestamps;
- masked fields;
- semantically equivalent generated identifiers.

State similarity for partial shaping:

- compare expected leaf paths;
- penalize missing expected changes;
- penalize unexpected mutations.

## 25.4 Alternative valid trajectories

This is important.

A policy may reach the correct state via a valid sequence different from reference.

Therefore trajectory reward must not dominate state reward.

Implement:

- masked call matching;
- dependency-order matching;
- precision/recall over required operations;
- tolerance for independent read-call reordering.

Test:

```text
reference path A succeeds
alternate path B reaches same final state
B receives high reward
```

## 25.5 Reward unit tests

- unparseable output → low/zero;
- exact correct → near max;
- correct final state with alternative valid sequence → high;
- wrong mutation → low;
- extra harmless read → small penalty only;
- masked argument difference → no penalty;
- tool execution exception handled;
- unexpected extra mutation penalized;
- read-only task still receives meaningful reward.

---

# 26. Phase 10 — GRPO with VeRL

Before implementing glue:

**re-check the current VeRL fork and current VeRL docs/code.**

If EnvFactory integration exists at implementation time:

- audit it;
- reuse where correct;
- keep our reward tests.

If not:

implement minimal glue on stock current VeRL.

## 26.1 RL stage A — single-turn replay RL

Start here.

Input already contains dialogue history up to the current turn.

Policy generates the tool-call sequence for that turn.

Then:

```text
parse calls
→ fresh MCP sandbox
→ load initial state
→ execute calls
→ save state
→ composite reward
```

No live tool feedback during rollout.

Advantages:

- simpler;
- lower latency;
- matches current RL sample shape;
- easier single-GPU GRPO proof.

## 26.2 RL stage B — live multi-turn tools

After A works.

Policy:

```text
generate
→ tool call
→ execute MCP
→ tool result
→ continue generation
```

Use VeRL's current tool/agent-loop API.

This is higher fidelity but not a prerequisite for the first GRPO result.

## 26.3 Primary model

```text
Qwen3-1.7B
initialized from SFT epoch-1 checkpoint in paper-shaped mode
```

## 26.4 RL smoke config

Start:

```text
GRPO
rollout n = 4
train batch = 8
prompt max = 4096
response max = 2048
microbatch = 1
LR = 1e-6
small update count
```

After stable:

```text
n = 8
response max = 4096 if needed
larger effective batch
```

## 26.5 RL memory policy

1.7B full actor should be primary.

4B full RL is not the default target because actor + optimizer + rollout + reference is too tight with 32 GB host RAM.

8B LoRA is a possible stretch because frozen base + adapter can be much cheaper.

## 26.6 Environment isolation in RL

Every rollout has a unique client/session namespace.

No rollout reuses another rollout's mutated state.

All clients close in `finally`.

Track:

```text
active clients
child processes
host RSS
```

during training.

Any monotonic process/RSS growth is a blocker.

## 26.7 RL feasibility gate

Before a serious run:

- 20 rollout groups;
- 5 optimizer updates;
- validation;
- checkpoint;
- process exit;
- restart;
- resume.

No OOM.

No leaked Ray workers.

No MCP leak.

No reward exception.

## 26.8 RL success gate

At least:

- ≥100 updates or a justified comparable training budget;
- reward trend not collapsed;
- held-out executable final-state success improves over SFT or demonstrates a clear reward/behavior effect;
- full config/checkpoint/provenance preserved.

---

# 27. Phase 11 — held-out evaluation

Create a fixed evaluation suite.

## 27.1 Held-out environments

Primary generalization suite:

```text
100+ tasks
from held-out environments
```

Stratify by:

- read/write;
- dependency depth;
- one/multiple servers;
- tools per task;
- conversational/non-conversational;
- user interaction required;
- opaque-ID dependency.

## 27.2 Same-environment held-out seeds

Secondary suite:

- training environments;
- unseen scenario/query seeds.

Measures task generalization separate from environment generalization.

## 27.3 Checkpoints compared

At minimum:

```text
Base Qwen3-1.7B
SFT epoch 1
SFT final
SFT + GRPO
```

Also, when practical:

```text
released EnvFactory-1.7B
released EnvFactory-4B
our Qwen3-4B SFT
```

All evaluated with identical:

- prompt template;
- tool schemas;
- initial states;
- decoding settings;
- max turns.

## 27.4 Metrics

Primary:

```text
task success
final-state success
composite executable score
tool-call parse rate
tool-name validity
argument correctness
tool execution success
```

Secondary:

```text
exact/soft sequence match
redundant call rate
mean tool calls
mean turns
clarification rate
exception rate
latency
tokens
```

## 27.5 Statistics

Report:

- N;
- mean;
- bootstrap 95% CI where practical;
- failure category counts.

No result table without sample count.

---

# 28. Mandatory ablations

Tier B final report contains these.

## A1 — RandomWalk vs Topology

Question:

> does dependency-aware sampling create more satisfiable/executable tasks?

Metrics:

- unsatisfied required inputs;
- chain execution validity;
- QueryGen yield;
- final trajectory success.

## A2 — Semantic graph vs Semantic + LLM refinement

Question:

> do LLM-added logical dependencies improve graph/task coverage?

Metrics:

- isolated tools;
- edge count;
- fillability;
- valid chain rate;
- generation success.

## A3 — QueryGen mode

Compare:

```text
NON_CONV
CONV
CONV + refinements
```

## A4 — Reward form

Compare:

```text
trajectory/tool-call only
vs
state-heavy composite reward
```

At minimum run offline reward analysis if full RL duplicate run is too expensive.

## A5 — SFT vs SFT+RL

Primary learning claim.

## A6 — Reused vs newly generated environments

Optional but valuable:

- train/eval with reused upstream env subset;
- inspect comparable trajectories from regenerated/new envs;
- show EnvGen-created environments are usable by the same graph/query/training pipeline.

---

# 29. Testing pyramid

## Unit — no GPU/network

- config;
- manifests;
- hashes;
- atomic writes;
- lock behavior;
- seed lists;
- schema flattening;
- similarity weighting;
- sampler;
- state comparison;
- masked call matching;
- dataset split;
- reward.

## Integration — local MCP

- registration;
- load/call/save;
- metadata parity;
- lifecycle;
- isolation;
- timeout cleanup;
- 100-session leak test;
- deterministic env behavior.

## Model contract

- OpenAI-compatible endpoint;
- JSON parsing;
- tool-call format;
- reasoning/tool parsing if enabled.

## Pipeline smoke

```text
one chain
→ QueryGen
→ trajectory
→ converter
```

## Resume fault injection

Kill generation mid-run.

Restart.

Expected:

- exact requested unique seeds;
- no completed seed reruns;
- no corrupt partial final artifact.

## Training smoke

- forward;
- backward;
- save;
- reload;
- resume.

## RL smoke

- rollout;
- MCP reward;
- optimizer update;
- checkpoint;
- resume.

## Golden fixture

Handwritten Calendar-like fixture with known:

- initial state;
- tool schema;
- correct sequence;
- alternative correct sequence;
- wrong mutation;
- expected reward.

Never let all correctness depend on LLM-generated tests.

---

# 30. Failure taxonomy

## Core/MCP

```text
REGISTER
AUTO_INIT
CONNECTION
TIMEOUT
CLIENT_LEAK
PROCESS_LEAK
STATE_LOAD
STATE_SAVE
STATE_ISOLATION
```

## EnvGen

```text
METADATA_PARSE
METADATA_SCHEMA
CODE_SYNTAX
IMPORT
MCP_START
TOOL_SCHEMA_MISMATCH
TOOL_RUNTIME
STATE_TRANSITION
STATE_SERIALIZATION
NONDETERMINISM
NETWORK_EGRESS
REPAIR_EXHAUSTED
```

## Graph

```text
EMBEDDING
EDGE_JUDGE
CLASSIFICATION_PARSE
UNFILLABLE_PARAM
BAD_LLM_EDGE
SERIALIZATION
INVALID_CHAIN
```

## QueryGen

```text
SCENARIO_INVALID
CHAIN_INVALID
MCP_LOAD
ASSISTANT_TOOL_ERROR
USER_SIM_ERROR
MAX_STEPS
NO_VALID_CANDIDATE
EVALUATOR_PARSE
STATE_MISMATCH
MODEL_TIMEOUT
```

## Training

```text
OOM
NAN_LOSS
CHECKPOINT_WRITE
CHECKPOINT_RESUME
ROLLOUT_SERVER
RAY_WORKER
REWARD_EXCEPTION
TOOL_SESSION_LEAK
```

Every failure artifact includes category.

---

# 31. Failure recovery rules

## Model server crash

- stop scheduling;
- mark in-flight retryable;
- preserve completions;
- store last health check/log tail;
- restart;
- verify model ID;
- resume.

## MCP failure

- close affected session;
- retry once fresh if transient;
- quarantine repeated server/seed pair;
- if one server fails >20% in pilot, stop and investigate.

## Session termination

Next session:

1. bootstrap/doctor;
2. pull state/artifacts;
3. verify hashes;
4. launch same model revision;
5. resume exact run ID.

## Disk pressure

Thresholds should be configurable.

Never delete completed artifacts automatically.

## OOM

Serving fallback order:

1. lower concurrent sequences;
2. lower GPU utilization;
3. lower context;
4. smaller/quantized teacher.

SFT fallback order:

1. microbatch 1;
2. gradient checkpointing;
3. shorter context;
4. efficient optimizer;
5. LoRA only if necessary.

RL fallback:

1. rollout n 4;
2. smaller train batch;
3. shorter response;
4. lower rollout memory fraction;
5. reference/offload tuning;
6. do not jump immediately to a different algorithm.

---

# 32. Observability

Structured JSONL event fields:

```text
timestamp
run_id
stage
unit_id/seed
worker_id
server
tool
operation
duration
outcome
retry
exception category
safe message
prompt tokens
output tokens
```

Resource sampler every ~30 seconds:

- process RSS;
- free host RAM;
- GPU utilization;
- VRAM;
- open FDs;
- child process count.

Important canary:

```text
child process count
```

If it rises across completed trajectories, stop.

---

# 33. Benchmark protocol

Benchmark:

1. cold bootstrap;
2. golden8 catalog init;
3. 100 MCP round trips;
4. cold graph build;
5. warm cached graph build;
6. topology sampling 1,000 seeds;
7. 10 trajectories workers 1/2/4;
8. 100 trajectory pilot;
9. data conversion;
10. SFT 20-step smoke;
11. SFT throughput;
12. reward replay throughput;
13. GRPO smoke;
14. held-out eval.

Record:

- wall time;
- CPU/RSS;
- GPU/VRAM;
- requests/tokens;
- p50/p95 model latency;
- MCP process count;
- trajectory/hour;
- reward evaluations/sec;
- API cost if applicable.

Comparisons use identical seed sets.

---

# 34. Implementation milestones

## M0 — repository + doctor

Exit:

- cold bootstrap works;
- compile/test discovery works;
- GPU and persistence report exists.

## M1 — upstream correctness patches

Exit:

- registration fixed;
- auto-init controllable;
- config isolation green;
- explicit RNG green;
- async MCP stress test green.

## M2 — golden8

Exit:

- catalog validates;
- 55-tool expected count confirmed or updated from current pinned commit;
- lifecycle and isolation tests green.

## M3 — environment proof

Exit:

- three known environments regenerated;
- five new/modified environments generated;
- ≥95% pass independent health checks;
- one intentionally broken env repaired successfully.

## M4 — graph

Exit:

- graph builds;
- cache works;
- fillability audit has no unexplained required-param holes;
- JSON backup written.

## M5 — topology ablation

Exit:

- 100+ fixed seeds compared;
- topology shows measurable reduction in invalid/unsatisfied chains or the contrary result is documented.

## M6 — serving

Exit:

- stable local model;
- health/structured/tool response works;
- stop releases VRAM.

## M7 — QueryGen smoke

Exit:

- 10 trajectories;
- kill/resume test;
- candidate state isolation verified.

## M8 — 100 trajectory pilot

Exit:

- generation/conversion yield gates pass;
- 20 trajectories manually inspected;
- throughput/cost measured.

## M9 — SFT dataset

Exit:

- deterministic train/val;
- token audit;
- no leakage;
- LLaMA-Factory dry load.

## M10 — 1.7B full SFT

Exit:

- LR sweep decision;
- full run;
- checkpoint survives session;
- beats base on at least one executable held-out metric.

## M11 — conversational/data scale

Exit:

- NON_CONV + CONV + refinement data;
- QueryGen ablation recorded;
- Tier B data volume.

## M12 — 4B SFT

Exit:

- memory gate;
- full SFT or documented full-FT infeasibility and controlled fallback;
- held-out result.

## M13 — reward library

Exit:

- all reward golden tests pass;
- alternative valid trajectory accepted;
- replay throughput measured.

## M14 — GRPO smoke

Exit:

- 20 rollout groups;
- 5 updates;
- checkpoint/resume;
- no leaks/OOM.

## M15 — GRPO result

Exit:

- meaningful training budget;
- held-out Base/SFT/SFT+RL table;
- reward curve;
- failure analysis.

## M16 — final reproduction report

Contains:

- method;
- hardware;
- deviations;
- data stats;
- all ablations;
- model comparisons;
- known limitations;
- reproducibility manifests.

---

# 35. Exact first session

Do not attempt serious EnvGen or training yet.

1. Create `envfactory-molab`.
2. Pin upstream SHA.
3. Create three venvs.
4. Run doctor.
5. Add patch harness.
6. Apply serving dependency patches.
7. Apply registration/auto-init patches.
8. Apply config/RNG patches.
9. Implement async MCP wrappers.
10. Start Calculator.
11. Call Calculator.
12. Start Calendar.
13. load state.
14. mutate state.
15. save state.
16. start second Calendar session.
17. prove isolation.
18. run 100-session cleanup test.
19. start small local Qwen server.
20. make one tool-formatted request.
21. stop server.
22. confirm VRAM release.
23. write baseline benchmark.
24. push code/config.
25. sync run artifacts.

Do not proceed if client/process leak exists.

---

# 36. Exact second session

Goal: catalogs + EnvGen proof start.

1. Validate golden8 metadata/implementations.
2. Confirm registered model-visible tool count.
3. Create env health runner.
4. Run independent checks over golden8.
5. Select three known regeneration environments.
6. Regenerate metadata for first known env.
7. regenerate executable env.
8. run EnvGen validation/revision.
9. run independent validator.
10. compare behavior with released env.
11. repeat for remaining known environments as session allows.
12. draft at least one new sketch.
13. run it through full EnvGen.
14. record model/provider/cost/time.
15. update decision log.

---

# 37. Exact third session

Goal: graph + sampler.

1. Finish required EnvGen proof envs if needed.
2. implement embedding backend abstraction.
3. implement/cache BGE embeddings.
4. build golden8 graph.
5. classify user-provided params.
6. semantic edge build.
7. run LLM edge refinement.
8. graph fillability audit.
9. serialize pickle + JSON/GraphML.
10. confirm warm rebuild cache hit.
11. audit TopologySampler current behavior.
12. ensure explicit RNG.
13. generate 100 fixed RandomWalk chains.
14. generate same 100 Topology chains.
15. calculate satisfiability/executability.
16. write ablation report.

---

# 38. Exact QueryGen session checklist

Before run:

- graph hash fixed;
- profile fixed;
- seed list persisted;
- model health green;
- RAM > safety threshold;
- child process count baseline recorded.

For every seed:

- sample chain;
- save chain;
- validate required inputs;
- generate scenario;
- validate/load scenario;
- generate query;
- run pass-k candidates from fresh state;
- store full traces;
- evaluate;
- select;
- replay selected candidate;
- persist success atomically;
- close all clients.

After batch:

- child process count baseline;
- manifest flush;
- trajectory yield;
- converter acceptance;
- manual review sample;
- remote sync.

---

# 39. Exact pre-SFT checklist

- fixed train/val environment split;
- fixed trajectory split;
- no duplicated trajectory between splits;
- data converter deterministic;
- tokenizer revision recorded;
- 20 rendered examples manually correct;
- length distribution measured;
- model chat template verified;
- 20-step train smoke;
- checkpoint reload;
- held-out base evaluation saved.

---

# 40. Exact pre-RL checklist

- SFT checkpoint verified;
- RL dataset round-trips;
- environment determinism checks green;
- reward unit tests 100%;
- state replay test green;
- alternative-valid-trajectory reward test green;
- 20 reward replays no leaks;
- VeRL installed/current API audited;
- single rollout group works;
- one optimizer update;
- save;
- resume.

---

# 41. Definition of done — Tier A

- [ ] cold MoLab bootstrap reproducible;
- [ ] core upstream correctness patches tested;
- [ ] async MCP path non-blocking;
- [ ] no client/process leaks;
- [ ] golden8 validates;
- [ ] local model serves on one GPU;
- [ ] graph builds and caches;
- [ ] topology chains sample deterministically;
- [ ] 100-trajectory pilot resumes safely;
- [ ] dataset conversion deterministic;
- [ ] Qwen3-1.7B full SFT completes;
- [ ] executable held-out eval produced;
- [ ] all artifacts have provenance.

---

# 42. Definition of done — Tier B

Everything Tier A plus:

- [ ] three known environments regenerated without copying tool implementation;
- [ ] five new/modified environments synthesized and validated;
- [ ] full parameter-level graph audited;
- [ ] BGE-M3 graph path works;
- [ ] LLM logical edge refinement measured;
- [ ] RandomWalk vs Topology ablation complete;
- [ ] non-conv vs conv/refinement QueryGen ablation complete;
- [ ] 3k–5k trajectory-scale dataset or justified equivalent;
- [ ] Qwen3-1.7B full SFT final;
- [ ] Qwen3-4B full SFT attempted after memory gate;
- [ ] held-out environment generalization evaluated;
- [ ] reward library complete;
- [ ] Qwen3-1.7B GRPO smoke;
- [ ] GRPO training result;
- [ ] Base vs SFT vs SFT+RL reported;
- [ ] final README/report contains exact deviations from paper/repo behavior.

---

# 43. Strong public-project deliverables

The GitHub/HF release should contain:

1. master README;
2. MoLab notebooks;
3. runbook;
4. patch set against pinned upstream;
5. EnvGen proof artifacts;
6. new generated environments;
7. environment health report;
8. ToolGraph JSON/visualization;
9. topology-vs-random results;
10. trajectory samples;
11. dataset card;
12. Qwen3-1.7B SFT checkpoint;
13. Qwen3-4B checkpoint if completed;
14. Qwen3-1.7B GRPO checkpoint;
15. held-out evaluation set;
16. Base/SFT/RL report;
17. reproducibility manifests.

---

# 44. Final report structure

```text
1. Problem
2. What EnvFactory contributes
3. What we reused
4. What we changed for MoLab
5. Core bugs/hardening found
6. Environment synthesis reproduction
7. ToolGraph construction
8. Topology sampling ablation
9. Query generation
10. Generated data statistics
11. SFT setup
12. SFT results
13. Executable reward
14. GRPO setup
15. RL results
16. Held-out environment evaluation
17. Ablations
18. Compute / throughput / cost
19. Deviations from paper
20. Limitations
21. Reproduction instructions
```

Be explicit about reuse.

A good sentence is:

> We reuse the public EnvFactory implementation as a pinned substrate, add correctness and MoLab compatibility patches, independently regenerate and synthesize executable environments, audit and ablate its dependency-aware sampling and QueryGen behavior, and reproduce SFT plus executable-environment GRPO on a single RTX PRO 6000.

Do not claim a from-scratch implementation if it is not one.

---

# 45. Agent handoff protocol

Any coding agent taking over must:

1. read `MASTER_IMPLEMENTATION_PLAN.md`;
2. read `DECISIONS.md`;
3. check `git status`;
4. inspect current run manifest;
5. identify the earliest incomplete milestone;
6. run nearest relevant tests before editing;
7. implement only that bounded milestone/subtask;
8. add tests in same change;
9. run exit criteria;
10. update status/evidence;
11. commit reviewable changes;
12. write handoff note.

Handoff note:

```text
Current milestone:
Files changed:
Behavior changed:
Tests run:
Exact results:
Tests not run:
Artifacts created:
Known failures:
Resource observations:
Next exact command/task:
```

Never mark a GPU test complete because a mock passed.

---

# 46. Decision log items to record early

At first successful bootstrap:

- upstream SHA;
- Python;
- PyTorch/CUDA;
- vLLM/SGLang version;
- attention backend;
- teacher model;
- storage destination;
- golden/small profiles;
- graph embedding backend;
- recursion depth/reproduction mode.

After pilots:

- teacher size;
- max context;
- QueryGen worker count;
- pass-k;
- LR for 1.7B;
- SFT cutoff;
- whether 4B full FT is viable;
- GRPO rollout n.

Never rewrite old decisions. Append.

---

# 47. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Blackwell serving incompatibility | high | separate serving venv; vLLM→SGLang→API fallback |
| MCP process leak | critical | async/lifecycle patches; 100-session test; child-process monitor |
| host RAM exhaustion | critical | golden/small profiles; bounded clients/workers |
| 12h termination | high | per-unit atomic outputs; frequent model checkpoints; remote sync |
| ToolGraph LLM cost explosion | high | embedding filter, top-k, edge cache |
| environment nondeterminism | high for RL | deterministic env health gate; exclude/fix offenders |
| poor trajectory quality | high | 10/100 pilots, executable evaluation, manual review |
| split leakage | critical for reported results | immutable env split before generation; trajectory-level split |
| reward overfits exact reference path | high | state-heavy reward; alternate-valid tests |
| VeRL API mismatch | medium/high | inspect current version first; minimal integration; single-turn mode first |
| 4B full SFT OOM | medium | memory gate; sequence/microbatch tuning |
| key leakage | critical | local env only; scrub logs; upload allowlists |
| upstream changes | medium | pinned SHA; patch test suite |
| pickle incompatibility | medium | JSON/GraphML backup + version manifest |

---

# 48. Things not to do

Do not:

- boot 85 servers in the first session;
- use upstream old serving pins blindly;
- store secrets in notebooks;
- call global `random.seed()` in concurrent code;
- allow candidate trajectories to share state;
- run serious SFT before held-out split is frozen;
- run RL before environment determinism is checked;
- use exact tool-sequence equality as the only reward;
- mark all opaque IDs user-provided to fix graph fillability;
- scale QueryGen before inspecting first 20 outputs;
- keep vLLM running while launching ordinary SFT;
- rely on notebook storage as the only model checkpoint;
- run multi-turn VeRL before single-turn replay RL works;
- optimize performance before lifecycle correctness;
- claim paper fidelity when using materially different settings without labeling them.

---

# 49. Highest-value compact result if time becomes constrained

If the project has to stop early, preserve the scientific core rather than scale.

Best compact outcome:

```text
8–20 verified environments
+
3 regenerated known envs
+
5 new EnvGen envs
+
full dependency graph
+
RandomWalk vs Topology ablation
+
500–1,000 verified trajectories
+
non-conv vs conv QueryGen comparison
+
Qwen3-1.7B full SFT
+
executable held-out evaluation
+
small Qwen3-1.7B GRPO run
+
Base vs SFT vs SFT+RL
```

This is substantially stronger than:

```text
85 reused environments
+
released dataset
+
one LoRA training command
```

because it demonstrates the actual mechanisms.

---

# 50. Immediate next action

The first coding agent should begin at:

```text
M0 → M1
```

Specifically:

1. create repository skeleton;
2. pin upstream;
3. implement doctor;
4. create three virtual environments;
5. install/verify the generation stack;
6. make Calculator and Calendar work;
7. implement the core correctness patches;
8. implement non-blocking MCP API;
9. run lifecycle stress tests;
10. only then touch EnvGen/ToolGraph/QueryGen.

The project should not consume meaningful generation or training budget until the MCP lifecycle is demonstrably leak-free and restart-safe.
