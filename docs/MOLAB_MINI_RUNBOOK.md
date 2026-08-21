# EnvFactory Mini on MoLab Runbook

This runbook records commands only after they are implemented and verified.
Commands shown in the implementation plan but not yet available remain outside
this file until their phase is complete.

## Phase 0: compatibility and test discovery

EnvFactory Mini requires Python 3.12 or newer. Install the package with its
development dependencies in a Python 3.12 environment:

```bash
python3.12 -m pip install -e ".[dev]"
```

Compile the existing source tree and the fixed eight-server mini catalog:

```bash
python3.12 -m compileall -q src \
  envs/tools/Calculator.py \
  envs/tools/Calendar.py \
  envs/tools/CampusCard.py \
  envs/tools/HotelBooking.py \
  envs/tools/MovieRecommender.py \
  envs/tools/Retail.py \
  envs/tools/Telecom.py \
  envs/tools/Weather.py
```

Discover and run the CPU-only baseline suites:

```bash
python3.12 -m pytest --collect-only -q
python3.12 -m pytest -q tests/unit
python3.12 -m pytest -q tests/integration -m "not model"
```

The compile command was verified locally with Python 3.12.13 on 2026-08-19. The
clean local interpreter did not have the development extra installed, so the
same collection and test arguments were verified in an ephemeral Python 3.12
environment with `uv run --no-project --python python3.12 --with
'pytest>=8,<10'`. GPU, model-server, and clean MoLab-session checks are
intentionally not claimed here.

## Phase 1: core configuration and correctness

For normal MCP auto-initialization, point `MCP_CONFIG_PATH` at a valid JSON
configuration before importing generator modules:

```bash
export MCP_CONFIG_PATH=configs/mcp_server.json
```

For commands that register a bounded server set explicitly, disable catalog
auto-initialization:

```bash
export SKIP_MCP_AUTO_INIT=true
```

Missing model-provider variables and attempts to use an unregistered MCP server
now fail with configuration-specific exceptions before a network call. The LLM
client opens its HTTP clients and worker pool lazily on first configured use.

Run the Phase 1 regressions and complete CPU suite with:

```bash
python3.12 -m pytest -q tests/unit/test_core_correctness.py
python3.12 -m pytest -q
```

The suite was verified locally on 2026-08-19: 7 Phase 1 tests passed and all 17
collected tests passed. The RNG regression covers 64 concurrent samples and two
fresh Python processes using the same seed.

## Phase 2: async MCP lifecycle

Async generation and validation code must use `aload_scenario`, `acall_tool`,
`asave_all_scenarios`, and `aclose_client`. The synchronous manager methods are
compatibility wrappers for non-async notebooks and legacy entry points only.
Stateful client IDs own independent stdio sessions and must be closed when their
trajectory or validation attempt finishes.

The lifecycle timeouts and batch bound can be tuned without changing code:

```bash
export MCP_REGISTRATION_TIMEOUT_SECONDS=120
export MCP_CONNECTION_TIMEOUT_SECONDS=30
export MCP_LOAD_SCENARIO_TIMEOUT_SECONDS=120
export MCP_TOOL_CALL_TIMEOUT_SECONDS=30
export MCP_SHUTDOWN_TIMEOUT_SECONDS=10
export MCP_BATCH_CONCURRENCY=4
```

Run the Phase 2 checks with:

```bash
python3.12 -m pytest -q tests/integration/test_mcp_async.py
python3.12 -m pytest -q tests/integration/test_mcp_lifecycle.py
python3.12 -m pytest -q tests/integration -m "not model"
python3.12 -m pytest -q
```

Verified locally on 2026-08-19 with Python 3.12.13 and FastMCP 3.1.0: all 5
live Phase 2 tests passed, the 13-test non-model integration suite passed, and
the full 27-test suite passed. The stress case opened and closed 100 Calculator
sessions in batches and returned both tracked-client and descendant-process
counts to their pre-test baselines. No GPU or model server is required.

## Artifact hygiene

Mini outputs belong under `artifacts/mini/`. Local virtual environments, model
downloads, and caches must stay in the ignored locations documented in
`.gitignore`. Never store API keys or tokens in commands, notebooks, reports, or
run manifests.

## Phase 3: mini configuration and catalog

The checked-in mini profile is `configs/mini/pipeline.toml`. Relative paths are
always resolved from the discovered repository root, so this check can be run
from another working directory. Pass `--repo-root` when package-location
discovery is not appropriate.

Only these deployment-specific overrides are accepted; structural experiment
settings remain in TOML:

```bash
export ENVFACTORY_MINI_ARTIFACT_ROOT=/persistent/envfactory/artifacts/mini
export ENVFACTORY_MINI_EMBEDDING_DEVICE=cpu
export ENVFACTORY_MINI_TEACHER_BASE_URL=http://127.0.0.1:8000/v1
export ENVFACTORY_MINI_TEACHER_API_KEY_ENV=VLLM_API_KEY
```

Validate the exact fixed catalog without an API key or network access:

```bash
python3.12 -m src.mini.catalog \
  --config configs/mini/pipeline.toml \
  --check
```

The command was verified locally on 2026-08-19. It reported eight servers, 55
metadata tools, and catalog SHA-256
`84d6dc1ab676859095a3ddfca8979978db0b083b949404a3f2f38334768b816d`.
It statically rejects HTTP-client imports before importing each audited local
tool file, then lists the in-process FastMCP registrations and checks lifecycle
and metadata tool-name parity.

## Phase 4: local embeddings and graph build

The graph builder uses the local sentence-transformers backend from the `mini`
extra. Install it before a live build (the static dry run remains model-free):

```bash
python3.12 -m pip install -e ".[dev,mini]"
```

Validate catalog inputs, output paths, backend identity, and deferred live
dependencies without downloading a model, contacting the teacher, or writing
artifacts:

```bash
python3.12 -m src.mini.build_graph \
  --config configs/mini/pipeline.toml \
  --dry-run
```

After Phase 5 starts and verifies the configured localhost teacher endpoint,
build the graph or safely reuse its trusted cache:

```bash
python3.12 -m src.mini.build_graph \
  --config configs/mini/pipeline.toml
```

Use `--force` only when intentionally replacing a graph after its inputs
change. Without it, an incomplete graph/manifest pair, an input fingerprint
mismatch, or an output hash mismatch fails closed. Pickle loading occurs only
after the current input fingerprint and manifest-recorded graph hash match.

Run the Phase 4 checks with:

```bash
python3.12 -m pytest -q \
  tests/unit/test_embedding_cache.py \
  tests/unit/test_user_provided_cache.py \
  tests/unit/test_mini_graph_build.py \
  tests/integration/test_mini_graph.py
python3.12 -m pytest -q
```

Verified locally on 2026-08-19 with Python 3.12.13: 14 dedicated Phase 4 tests
and the full 55-test suite passed. The real dry run reported eight servers, 55
tools, and only the Phase 5 teacher endpoint as unmet. It wrote no graph or
manifest. The first live build is intentionally deferred until the local
teacher endpoint is available.

## Phase 5: isolated MoLab environments and local model serving

The runtime and training dependency solvers must remain separate. They share a
Hugging Face cache and the configured artifact directory, but never a virtual
environment or site-packages. Choose persistent locations available in the
current MoLab session before creating the environments:

```bash
export HF_HOME=/persistent/envfactory/huggingface
export ENVFACTORY_MINI_ARTIFACT_ROOT=/persistent/envfactory/artifacts/mini

uv venv .venv-mini-runtime --python 3.12
uv pip install --python .venv-mini-runtime/bin/python -e .
uv pip install --python .venv-mini-runtime/bin/python \
  --torch-backend=auto -r requirements-molab.txt

uv venv .venv-mini-train --python 3.12
uv pip install --python .venv-mini-train/bin/python -e .
uv pip install --python .venv-mini-train/bin/python \
  --torch-backend=auto -r requirements-molab-train.txt
```

The runtime requirements deliberately leave vLLM unpinned so `uv` can select a
current PyTorch/CUDA wheel against the live driver. NVIDIA Blackwell requires
CUDA 12.8 or newer. The training profile currently selects LlamaFactory 0.9.4;
update it only through a separately tested dependency change. Official
references: [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
and [LlamaFactory releases](https://github.com/hiyouga/LlamaFactory/releases).

Read the local API key without echoing it or placing it in shell history, then
run both environment profiles. JSON reports include the complete installed
package/version map but only secret-variable presence, never secret values:

```bash
read -rsp "Local vLLM API key: " VLLM_API_KEY
export VLLM_API_KEY
printf '\n'

.venv-mini-runtime/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml \
  --environment-profile runtime \
  --without-model \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/environment-runtime.json"

.venv-mini-train/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml \
  --environment-profile train \
  --without-model \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/environment-train.json"
```

The doctor fails closed unless it finds Linux, Python 3.12+, at least four
CPUs, 30 GiB host RAM, 10 GiB free disk, a CUDA 12.8+ PyTorch build, and a
compute-capability 12.0 GPU with at least 90 GiB VRAM. It warns below 20 GiB
free disk. The runtime profile additionally imports vLLM and checks
sentence-transformers and `VLLM_API_KEY`; the training profile checks
LlamaFactory instead.

Start the server with the runtime environment's executable. Launcher settings
use a `MOLAB_VLLM_*` namespace because vLLM reserves some `VLLM_*` names for
its own networking. The only intentional vLLM variable is `VLLM_API_KEY`,
which vLLM supports for authentication without a command-line secret.

```bash
export MOLAB_VLLM_RUN_DIR="$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/phase5-live-gate"
export MOLAB_VLLM_MODEL=Qwen/Qwen3-14B
export MOLAB_VLLM_PORT=8000
export MOLAB_VLLM_MAX_MODEL_LEN=16384
export MOLAB_VLLM_GPU_MEMORY_UTILIZATION=0.85
export MOLAB_VLLM_MAX_NUM_SEQS=4

bash src/serve/vllm_molab.sh start
bash src/serve/vllm_molab.sh status
bash src/serve/vllm_molab.sh health
```

`start` binds only to `127.0.0.1`, exposes GPU 0 only, uses tensor parallel
size 1, and writes its PID, pre-launch GPU baseline, and log beneath
`$MOLAB_VLLM_RUN_DIR/logs`. `health` validates the returned model identity,
any endpoint-reported context length, and one eight-token chat completion with
thinking disabled. Run the live model test after health passes:

```bash
RUN_MINI_MODEL_TESTS=1 .venv-mini-runtime/bin/python -m pytest -q \
  tests/integration/test_model_server_contract.py
```

Build the graph once cold, then force one rebuild to prove both persistent
caches serve every request. The builder itself performs the required
100-seed/six-tool/two-server acceptance validation on each construction:

```bash
.venv-mini-runtime/bin/python -m src.mini.build_graph \
  --config configs/mini/pipeline.toml

.venv-mini-runtime/bin/python -m src.mini.build_graph \
  --config configs/mini/pipeline.toml --force

.venv-mini-runtime/bin/python - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["ENVFACTORY_MINI_ARTIFACT_ROOT"]) / "graph" / "manifest.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["embedding_cache"]["misses"] == 0
assert manifest["classification_cache"]["misses"] == 0
print(manifest["counts"], manifest["embedding_cache"], manifest["classification_cache"])
PY
```

Finally stop the server. The launcher sends `SIGTERM`, waits before using
`SIGKILL`, and exits nonzero unless used VRAM returns within 512 MiB of the
recorded baseline:

```bash
bash src/serve/vllm_molab.sh stop
```

The launcher, doctor, and fake model-contract tests were verified locally on
2026-08-19. The live MoLab install, Qwen endpoint, graph cache, and VRAM release
remain required before Phase 5 can be marked complete.

## Phase 6: resumable trajectory synthesis

Keep the healthy localhost teacher and trusted graph from Phase 5 available,
then start a disposable smoke run. The command prints the generated run ID;
preserve it for inspection or resume:

```bash
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --target 10 --workers 2
```

For the pilot, start a distinct run and resume only with its original target.
Worker count may be adjusted between sessions; semantic configuration, graph,
catalog, model identity, and target may not change:

```bash
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --target 100 --workers 4

.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --run-id <run-id> --target 100 --workers 4 --resume
```

`SIGINT` and `SIGTERM` stop new work, let in-flight atomic commits finish, and
leave the manifest in `interrupted` state. A completed file is immutable and is
the resume source of truth. Failures retain every attempt in
`trajectories/failed/<seed>.json`; only transient model/timeout/transport
failures below the configured attempt limit are retried. If a dead process
leaves `.synthesis.lock`, inspect it first and then explicitly add
`--recover-stale-lock`. Never use that flag while the recorded PID is live.
When compatibility or existing-run refusals block a deliberate restart, add
`--new-run` to start a fresh suffixed run directory (plan §9 rule 6) instead of
mutating or resuming the old one; without it, re-using an occupied run ID
fails with `run already exists; use --resume`.

Run the local fault-injection regressions with:

```bash
python3.12 -m pytest -q \
  tests/unit/test_mini_manifest.py \
  tests/integration/test_mini_resume.py
```

Verified locally on 2026-08-19 with Python 3.12.13: all eight focused tests,
49 unit tests, and 19 non-model integration tests passed; the full suite had
68 passes and one expected live-model skip. The focused resume
test interrupts a deterministic ten-seed run, resumes to exactly ten unique
immutable completions, verifies transient retry records, and proves another
completed resume does not initialize a generator. The real ten-trajectory
teacher smoke and killed 100-trajectory MoLab run remain acceptance gates.

## Phase 8: run-scoped LoRA training

Phase 7 dataset preparation now writes both
`datasets/dataset_info.json` and the full run-specific
`training/resolved_llamafactory.yaml`. The renderer verifies the dataset
manifest and file hashes before writing either file. It never starts training,
sets `overwrite_output_dir: false`, and keeps datasets, checkpoints,
TensorBoard logs, and adapter artifacts beneath the selected run directory.

Stop the teacher first. The MoLab launcher verifies that VRAM returns to its
recorded baseline before it reports a successful stop:

```bash
bash src/serve/vllm_molab.sh stop

.venv-mini-runtime/bin/python -m src.mini.prepare_dataset \
  --config configs/mini/pipeline.toml --run-id <run-id>
```

Render and run the bounded 4B smoke profile. It trains for 20 steps and saves
at steps 10 and 20:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile smoke --per-device-batch-size 2

.venv-mini-train/bin/llamafactory-cli train \
  artifacts/mini/runs/<run-id>/training/resolved_llamafactory_smoke.yaml

.venv-mini-train/bin/python -m src.mini.training verify \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --output-dir artifacts/mini/runs/<run-id>/training/smoke \
  --minimum-step 20
```

The verifier requires finite recorded loss and a complete resumable checkpoint:
trainer state, optimizer, scheduler, LoRA config, and LoRA weights. Render the
one-step continuation from the newest trusted smoke checkpoint, run it, and
prove that the trainer advanced:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile resume-check --per-device-batch-size 2

.venv-mini-train/bin/llamafactory-cli train \
  artifacts/mini/runs/<run-id>/training/resolved_llamafactory_resume_check.yaml

.venv-mini-train/bin/python -m src.mini.training verify \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --output-dir artifacts/mini/runs/<run-id>/training/smoke \
  --minimum-step 21 --previous-step 20
```

Probe the 8B student at batch size 2 for one step while recording peak allocated
GPU memory. If it OOMs or peak allocation exceeds 90 GB, rerender and rerun with
batch size 1. Gradient accumulation changes from 8 to 16, preserving effective
batch size 16:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile memory-probe --per-device-batch-size 2

.venv-mini-train/bin/llamafactory-cli train \
  artifacts/mini/runs/<run-id>/training/resolved_llamafactory_memory_probe_batch_2.yaml

# Only after an OOM or a measured peak above 90 GB:
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile memory-probe --per-device-batch-size 1
```

The full profile uses the 8B model, one epoch, validation evaluation, BF16,
gradient checkpointing, and local TensorBoard reporting. If the memory probe
selected batch size 1, rerender the full profile with that size before launch:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile full --per-device-batch-size 2

.venv-mini-train/bin/llamafactory-cli train \
  artifacts/mini/runs/<run-id>/training/resolved_llamafactory.yaml

.venv-mini-train/bin/python -m src.mini.training verify \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --output-dir artifacts/mini/runs/<run-id>/training/checkpoints \
  --minimum-step 1 --promote-adapter
```

If a session ends during the full job, render the same full profile with the
latest trusted checkpoint explicitly and relaunch it. Do not set
`overwrite_output_dir` or copy a checkpoint between runs:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> --profile full \
  --per-device-batch-size 2 \
  --resume-from-checkpoint \
  artifacts/mini/runs/<run-id>/training/checkpoints/checkpoint-<step>

.venv-mini-train/bin/llamafactory-cli train \
  artifacts/mini/runs/<run-id>/training/resolved_llamafactory.yaml
```

The last command validates that the output identifies a LoRA adapter, copies
only adapter files atomically to `training/adapter`, records tokenizer and
package metadata plus the dataset-manifest hash, updates the run manifest when
present, and scans text training outputs for values of currently defined
secret-like environment variables. It never merges the adapter into the base
model.

Run the CPU-only Phase 8 regressions with:

```bash
python3.12 -m pytest -q \
  tests/unit/test_mini_training.py \
  tests/unit/test_data_split.py
```

Verified locally on 2026-08-20: all 11 focused tests passed. LlamaFactory and
an NVIDIA runtime were absent locally, so the 20-step training, real checkpoint
reload, 8B memory probe, full epoch, and validation-loss gates remain for MoLab.

## Phase 9: executable teacher/student evaluation

Evaluation uses only the trajectory seeds listed in
`datasets/dataset_manifest.json` under `split.validation_seeds`. The first
candidate invocation freezes their task IDs, source hashes, prompt hashes,
tool-schema hashes, initial-state hashes, reference-state hashes, decoding
settings, and maximum turns in `evaluation/suite.json`. A later candidate must
match that contract exactly.

Start the teacher as described in Phase 5, then score it. The model name must
be the exact identifier exposed by `/v1/models`:

```bash
.venv-mini-runtime/bin/python -m src.mini.evaluate \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --candidate teacher --model Qwen/Qwen3-14B

bash src/serve/vllm_molab.sh stop
```

The stop command is a required boundary: it exits nonzero unless GPU memory
returns within the configured tolerance of its pre-launch baseline. Do not
start the student until it succeeds.

Serve the verified adapter through vLLM's named LoRA path. Use an absolute
adapter path; the named module becomes the student model identifier. The
maximum rank must be at least the `r` recorded in `adapter_config.json`:

```bash
export MOLAB_VLLM_RUN_DIR="$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/<run-id>"
export MOLAB_VLLM_MODEL=Qwen/Qwen3-8B
export MOLAB_VLLM_EXPECTED_MODEL=envfactory-mini-student
export MOLAB_VLLM_LORA_MODULE="envfactory-mini-student=$(realpath "$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/<run-id>/training/adapter")"
export MOLAB_VLLM_MAX_LORA_RANK=64

bash src/serve/vllm_molab.sh start
bash src/serve/vllm_molab.sh health

.venv-mini-runtime/bin/python -m src.mini.evaluate \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --candidate student --model envfactory-mini-student \
  --student-serving-mode lora \
  --adapter-path "$(realpath "$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/<run-id>/training/adapter")" \
  --max-lora-rank 64

bash src/serve/vllm_molab.sh stop
```

The evaluator refuses to score a student under the unmodified base-model name.
Its health probe requires both `/v1/models` and a chat completion to report the
named adapter, then records the adapter tree hash, base revision, adapter rank,
and maximum served rank. If the verified vLLM build cannot serve LoRA, merge
only as an explicit, separate export under the run directory and use
`--student-serving-mode merged` with `--merged-model-path`, `--adapter-path`,
`--base-revision`, and `--merge-dtype`; the evaluator records both artifact
hashes and refuses paths outside the run.

Each task receives a new stateful MCP session. The initial state is immediately
round-tripped before inference, tool calls are schema-checked before execution,
and every client closes in `finally`. Failed traces are retained under
`evaluation/traces/failed/` with current secret values and secret-shaped fields
redacted. Candidate metrics are written to `teacher_metrics.json` and
`student_metrics.json`; `evaluation/report.md` links those sources, compares
both candidates side by side, applies the provisional gates, and links up to
five representative failures per candidate. Pass `--overwrite` only when
intentionally replacing a candidate result without changing the frozen suite.

Run the CPU-only Phase 9 regressions with:

```bash
python3.12 -m pytest -q \
  tests/unit/test_mini_evaluation.py \
  tests/unit/test_mini_config.py
```

## Phase 10: marimo notebooks and clean-session workflow

The notebooks are UI orchestrators for the commands below. They never contain
an API key, silently install dependencies, mutate the shared LlamaFactory YAML,
or automatically upload artifacts. Named subprocess state and streaming logs
are stored beneath `$ENVFACTORY_MINI_ARTIFACT_ROOT/notebook/`; changing a
reactive control cannot replace a live named job. Every stop control sends a
graceful termination and waits before force-killing.

Set persistent paths and the secret environment outside the notebook, then
install both isolated environments once. `requirements-molab.txt` installs the
marimo command used to open the notebooks:

```bash
export ENVFACTORY_REPO_ROOT="$(pwd -P)"
export ENVFACTORY_MINI_ARTIFACT_ROOT=/persistent/envfactory/artifacts/mini
export HF_HOME=/persistent/envfactory/huggingface
read -rsp "Local vLLM API key: " VLLM_API_KEY
export VLLM_API_KEY
printf '\n'

uv venv .venv-mini-runtime --python 3.12
uv pip install --python .venv-mini-runtime/bin/python -e .
uv pip install --python .venv-mini-runtime/bin/python \
  --torch-backend=auto -r requirements-molab.txt

uv venv .venv-mini-train --python 3.12
uv pip install --python .venv-mini-train/bin/python -e .
uv pip install --python .venv-mini-train/bin/python \
  --torch-backend=auto -r requirements-molab-train.txt

.venv-mini-runtime/bin/marimo edit examples/molab_mini_generate.py
# In the later, GPU-exclusive training phase:
.venv-mini-runtime/bin/marimo edit examples/molab_mini_train.py
```

The session clock shown in each notebook warns after 10.5 hours. Installation
has no notebook run button by design, so a reactive rerun cannot invoke `uv`.

### Generation notebook action equivalents

The read-only doctor and redacted configuration preview correspond to the
doctor's JSON report. It records secret-variable presence, never values:

```bash
.venv-mini-runtime/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml --without-model --json
```

Start, check, and stop the localhost teacher with the existing launcher. The
notebook records both its short launcher PID/log and the server's own PID file
and log beneath the selected run directory:

```bash
export MOLAB_VLLM_RUN_DIR="$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/notebook-teacher"
export MOLAB_VLLM_CONFIG="$ENVFACTORY_REPO_ROOT/configs/mini/pipeline.toml"
export MOLAB_VLLM_PYTHON="$ENVFACTORY_REPO_ROOT/.venv-mini-runtime/bin/python"
export MOLAB_VLLM_EXECUTABLE="$ENVFACTORY_REPO_ROOT/.venv-mini-runtime/bin/vllm"

bash src/serve/vllm_molab.sh start
.venv-mini-runtime/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml --require-model
bash src/serve/vllm_molab.sh stop
```

Validate the catalog and build or reuse the graph. `--force` is exposed only
through a separate deliberate checkbox:

```bash
.venv-mini-runtime/bin/python -m src.mini.catalog \
  --config configs/mini/pipeline.toml --check
.venv-mini-runtime/bin/python -m src.mini.build_graph \
  --config configs/mini/pipeline.toml
# Deliberate rebuild only:
.venv-mini-runtime/bin/python -m src.mini.build_graph \
  --config configs/mini/pipeline.toml --force
```

Each generation size creates a separate job. Replace `<run-id>` only when
resuming; the notebook deliberately keeps the canonical pilot target on resume:

```bash
# Smoke
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml --target 10 --workers 2

# Pilot
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml --target 100 --workers 4

# Full first release
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml --target 2000 --workers 4

# Resume the pilot with its original semantic configuration
.venv-mini-runtime/bin/python -m src.mini.synthesize \
  --config configs/mini/pipeline.toml \
  --run-id <run-id> --target 100 --workers 4 --resume
```

Use `Ctrl-C` in a foreground terminal for the same graceful synthesis
shutdown. For a notebook-managed job, the equivalent explicit sequence reads
its visible state, sends `TERM`, waits 30 seconds, and uses `KILL` only if the
PID remains live:

```bash
JOB_STATE="$ENVFACTORY_MINI_ARTIFACT_ROOT/notebook/jobs/generation-pilot.json"
JOB_PID="$(.venv-mini-runtime/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$JOB_STATE")"
kill -TERM "$JOB_PID"
for _ in $(seq 1 30); do kill -0 "$JOB_PID" 2>/dev/null || break; sleep 1; done
kill -0 "$JOB_PID" 2>/dev/null && kill -KILL "$JOB_PID"
```

Refresh live metrics and bounded log tails without changing the run:

```bash
.venv-mini-runtime/bin/python -m json.tool \
  "$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/<run-id>/run_manifest.json"
tail -n 120 "$ENVFACTORY_MINI_ARTIFACT_ROOT/notebook/logs/generation-pilot.log"
```

Convert and validate the selected completed run:

```bash
.venv-mini-runtime/bin/python -m src.mini.prepare_dataset \
  --config configs/mini/pipeline.toml --run-id <run-id>
```

### Training/evaluation notebook action equivalents

Before training, the notebook runs the training-profile doctor and scans every
run-scoped server PID file. Any live server makes GPU exclusivity false:

```bash
.venv-mini-train/bin/python -m src.mini.doctor \
  --config configs/mini/pipeline.toml \
  --environment-profile train --without-model --json

find "$ENVFACTORY_MINI_ARTIFACT_ROOT/runs" -path '*/logs/model_server.pid' \
  -type f -print -exec sh -c 'kill -0 "$(cat "$1")" 2>/dev/null && echo LIVE' _ {} \;
```

Select and inspect a prepared run without printing full prompts or scenarios:

```bash
export RUN_DIR="$ENVFACTORY_MINI_ARTIFACT_ROOT/runs/<run-id>"
test -d "$RUN_DIR"
.venv-mini-runtime/bin/python -m json.tool "$RUN_DIR/datasets/dataset_manifest.json"
.venv-mini-runtime/bin/python - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["RUN_DIR"]) / "datasets" / "sft_train.json"
rows = json.loads(path.read_text())
sample = rows[0] if rows else {}
print({"fields": sorted(sample), "field_lengths": {k: len(str(v)) for k, v in sample.items()}})
PY
```

The configuration preview reads the shared template and lists run-specific
resolved YAML paths. Rendering always writes beneath `$RUN_DIR/training/` and
never edits `configs/mini/llamafactory_sft.yaml`:

```bash
sed -n '1,240p' configs/mini/llamafactory_sft.yaml
find "$RUN_DIR/training" -maxdepth 1 -name 'resolved_llamafactory*.yaml' -print
printf 'dataset_dir=%s\ntraining_dir=%s\nadapter_dir=%s\n' \
  "$RUN_DIR/datasets" "$RUN_DIR/training" "$RUN_DIR/training/adapter"
```

Render, run, stop if needed, and verify the smoke profile:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile smoke --per-device-batch-size 2
.venv-mini-train/bin/llamafactory-cli train \
  "$RUN_DIR/training/resolved_llamafactory_smoke.yaml"
.venv-mini-train/bin/python -m src.mini.training verify \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --output-dir "$RUN_DIR/training/smoke" --minimum-step 20
```

Render a new full profile, or explicitly resume from a trusted checkpoint
inside the same run, then launch and promote the verified adapter:

```bash
.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile full --per-device-batch-size 2

.venv-mini-train/bin/python -m src.mini.training render \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --profile full --per-device-batch-size 2 \
  --resume-from-checkpoint "$RUN_DIR/training/checkpoints/checkpoint-<step>"

.venv-mini-train/bin/llamafactory-cli train \
  "$RUN_DIR/training/resolved_llamafactory.yaml"
.venv-mini-train/bin/python -m src.mini.training verify \
  --config configs/mini/pipeline.toml --run-id <run-id> \
  --output-dir "$RUN_DIR/training/checkpoints" --minimum-step 1 --promote-adapter
```

For a notebook-managed `training-smoke` or `training-full` PID, use the same
documented `TERM`, 60-second wait, then `KILL` sequence shown above. Never
start an evaluation endpoint until that PID is gone.

Teacher/student server and evaluation commands are the Phase 9 commands above.
The notebook exposes the same named LoRA settings and then displays the report:

```bash
sed -n '1,260p' "$RUN_DIR/evaluation/report.md"
tail -n 120 "$ENVFACTORY_MINI_ARTIFACT_ROOT/notebook/logs/evaluation-teacher.log"
tail -n 120 "$ENVFACTORY_MINI_ARTIFACT_ROOT/notebook/logs/evaluation-student.log"
```

### Explicit artifact export

Both notebooks show the same checklist: run manifest, resolved config,
environment report, logs, completed trajectories, dataset manifest, adapter,
and evaluation report. They require an absolute destination separate from the
artifact tree. Preview the exact non-deleting command, then run it explicitly:

```bash
EXPORT_ROOT=/persistent/envfactory/exports
printf 'source=%s\ndestination=%s\n' "$RUN_DIR" "$EXPORT_ROOT/<run-id>"
rsync -a --protect-args "$RUN_DIR/" "$EXPORT_ROOT/<run-id>/"
```

Do not add `--delete`, and do not prune local artifacts until the exported copy
has been independently verified. The notebook never performs an automatic
upload or delete.

Run the Phase 10 CPU checks with:

```bash
python3.12 -m pytest -q tests/unit/test_mini_notebook.py
python3.12 -m compileall -q \
  src/mini/notebook.py \
  examples/molab_mini_generate.py \
  examples/molab_mini_train.py
```

Local verification on 2026-08-20 passed eight focused tests with one expected
POSIX-only skip, Python 3.12 compile checks, `marimo check` under marimo 0.24.0, and headless HTML execution
of both notebooks with no failed cells. The clean-session exit gate remains:
open each notebook in a fresh
GPU-backed MoLab allocation, follow this command sequence, verify streaming
logs/PIDs/graceful stops and the 10.5-hour warning, and persist the selected run
to an explicit destination.

The same local verification also launched both notebooks as live marimo apps
and inspected their rendered controls. All required sections appeared in order,
the browser console remained free of errors, traversal-shaped run IDs and
relative export destinations were rejected, and missing non-MoLab executables
were reported as sanitized UI errors. This does not satisfy the Linux/GPU gate.

A disposable Linux/Python 3.12 reproduction subsequently passed all nine
focused Phase 10 tests and both notebook checks. The live generation UI launched the
real read-only doctor from `.venv-mini-runtime/bin/python`, displayed its
completed state and bounded report tail, and ran the catalog check to eight
servers/55 tools. Changing the artifact root did not create another catalog
job. The live training UI launched `.venv-mini-train/bin/python` and preserved
`gpu_exclusive: true`, an empty live-server list, PID, return code, and report
tail after refresh. Hardware/model/training/evaluation/export actions remain
MoLab-only acceptance gates.

## Phase 11 performance measurements

Run performance comparisons only with identical inputs and seed sets. Catalog
registration has a dedicated safe benchmark that records clocks, CPU samples,
process-tree RSS, child-process counts, and GPU telemetry when NVML is
available. Reports contain hashes and server names, but no environment values,
prompts, scenarios, or secrets:

```bash
.venv-mini-runtime/bin/python -m src.mini.benchmark catalog-registration \
  --config configs/mini/pipeline.toml \
  --label bounded-2 --samples 5 \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/catalog-bounded-2.json"
```

Compare two retained JSON reports without rerunning either workload:

```bash
.venv-mini-runtime/bin/python -m src.mini.benchmark compare \
  --before "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/catalog-before.json" \
  --after "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/catalog-bounded-2.json" \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/catalog-comparison.md"
```

`MCP_REGISTRATION_CONCURRENCY` may select one through four registration slots;
the default is two and values above four fail at startup. Do not use this
deployment control to compare different runs without recording it in the
benchmark label and exported report. The remaining generation, serving-context,
vLLM, and student-model sweeps require the same held-out inputs on MoLab and
must retain raw reports before changing defaults.

With the teacher endpoint healthy and the trusted graph cache present, run the
ten-trajectory worker sweep. It creates a separate synthesis run for each row,
but every row uses the same deterministic seed set, graph, catalog, and model
identity:

```bash
.venv-mini-runtime/bin/python -m src.mini.benchmark generation-workers \
  --config configs/mini/pipeline.toml \
  --label molab-smoke --workers 1 2 4 --target 10 --samples 1 \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/generation-workers.json" \
  --markdown-output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/generation-workers.md"
```

The raw report records per-run duration, successful trajectories per hour,
trajectory latency p50/p95, retries, failures, CPU/RSS/GPU peaks, and child
process cleanup. A non-completed synthesis state stops the sweep and still
writes the partial report with `summary.complete = false`; an exception still
leaves the underlying run artifacts for diagnosis. Counts above four are
rejected because the current four-CPU mini contract caps synthesis at four
workers. Test 6 and 8 only after a doctor report justifies revising that
contract in configuration and code. Inspect quality/failures and p95 latency,
not throughput alone, before changing the default.

The serving-context benchmark measures one live server configuration per
invocation, because context length, `max_num_seqs`, and GPU utilization are
vLLM launch settings. It requires a healthy endpoint (`--require-model` doctor
check passes) and a completed source run whose real trajectory prompts it
replays at deterministic length percentiles. Requests use temperature 0, the
run seed, thinking disabled, and recorded `max_tokens`, so latency differences
come from the server configuration rather than decoding variance. Reports
record prompt lengths and hashes, never prompt text:

```bash
# Measure the currently running server configuration (for example 16K context)
.venv-mini-runtime/bin/python -m src.mini.benchmark serving-context \
  --config configs/mini/pipeline.toml \
  --label ctx16k-seq4 --run-id <completed-run-id> \
  --requests 20 --concurrency 1 --max-tokens 256 \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/serving-ctx16k-seq4.json" \
  --markdown-output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/serving-ctx16k-seq4.md"

# Change exactly one vLLM launch setting (context, max_num_seqs, or GPU
# utilization), restart the server, rerun with a new label, then compare:
.venv-mini-runtime/bin/python -m src.mini.benchmark compare \
  --before "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/serving-ctx16k-seq4.json" \
  --after "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/serving-ctx8k-seq4.json" \
  --output "$ENVFACTORY_MINI_ARTIFACT_ROOT/benchmarks/serving-comparison.md"
```

Latency starts before client-side queueing, so p50/p95 under
`--concurrency 2..8` reflects what server-side `max_num_seqs` tuning changes;
concurrency stays within a client bound of eight and is recorded in the report.
Over-context request failures at 8K (HTTP 400/500) are retained as measurement
data with sanitized messages, not hidden: they bound the usable real-prompt
distribution at each context length. Track OOMs through the same error records
plus NVML peaks in the raw JSON. Retain every raw JSON report before changing
any serving default.

### Student 4B versus 8B throughput and quality comparison

No dedicated launcher exists for this comparison and none is needed: the
training notebook/runbook already drives LlamaFactory, and
`python -m src.mini.evaluate` already scores both students identically. The
comparison is a protocol over those existing artifacts:

1. Render and run the 20-step smoke job for `Qwen/Qwen3-4B`, then for
   `Qwen/Qwen3-8B`, each in its own run directory, with identical dataset,
   seed, effective batch size, and LoRA configuration. Record
   `train_runtime`, `train_samples_per_second`, peak allocated/reserved VRAM,
   and any OOM from each run's trainer logs or trainer state.
2. Train one epoch on the full split for the student that passes the memory
   probe (plan §6.3: fall back from batch size 2 to 1 before quantizing).
3. Evaluate both adapters with the same held-out suite, decoding settings,
   and maximum turns via `python -m src.mini.evaluate --model-role student`.
4. Retain both `evaluation/report.md` files plus the raw training logs before
   selecting the default student. Decide on executable task success and
   parse/valid-tool rates first; use throughput only to break ties.

Do not compare runs that used different datasets, seeds, batch sizes, or
LoRA targets; record every deviation in the decision log.
