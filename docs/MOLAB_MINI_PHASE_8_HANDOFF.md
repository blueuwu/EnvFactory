# MoLab Mini Phase 8 handoff

Date: 2026-08-20  
Status: In progress; local implementation complete, MoLab GPU gates pending

## Files changed

- `configs/mini/llamafactory_sft.yaml`: shared BF16 Qwen3 LoRA template.
- `configs/mini/dataset_info.example.json`: documented local dataset registry.
- `src/mini/training.py`: trusted profile rendering, checkpoint/resume
  validation, secret scan, adapter promotion, and training metadata capture.
- `src/mini/prepare_dataset.py`: emits the full run-specific training profile
  after all dataset gates pass.
- `tests/unit/test_mini_training.py`: Phase 8 CPU regressions.
- `requirements.txt`: explicit PyYAML runtime dependency.
- `docs/MOLAB_MINI_RUNBOOK.md`: smoke, resume, memory probe, full training, and
  adapter verification commands.
- `docs/MOLAB_MINI_IMPLEMENTATION_PLAN.md`: phase status and decision evidence.

## Behavior changed

Successful dataset conversion creates a run-local LlamaFactory dataset registry
and full training YAML. Additional CLI profiles cover a 20-step Qwen3-4B smoke,
a one-step Qwen3-8B memory probe at batch size 1 or 2, and an explicit one-step
checkpoint continuation. All profiles use effective batch size 16, BF16 LoRA,
gradient checkpointing, three preprocessing workers, two dataloader workers,
an 8,192-token cutoff, and non-overwriting output semantics.

Training verification rejects incomplete checkpoints, non-finite or missing
loss, non-advancing resume state, non-LoRA adapter output, paths outside the
run, dataset tampering, and current secret values found in textual outputs.
Verified adapter files are copied atomically; base weights are never merged.

## Tests run

```text
python -m pytest -q -p no:cacheprovider \
  --basetemp .test-work/pytest-phase8-final-focused \
  tests/unit/test_mini_training.py tests/unit/test_data_split.py
11 passed in 0.56s

python -m compileall -q src/mini \
  tests/unit/test_mini_training.py tests/unit/test_data_split.py
passed

git diff --check
passed (line-ending warnings only)
```

## Tests not run

- LlamaFactory 0.9.4 parser/dataset dry load: LlamaFactory is not installed in
  the local interpreter.
- Twenty-step GPU smoke and checkpoint reload: no NVIDIA runtime is present.
- Qwen3-8B batch-size memory probe: `nvidia-smi` is unavailable locally.
- Full one-epoch training and validation loss: requires the prepared MoLab run
  and RTX PRO 6000.
- Repository-wide compile: blocked by the pre-existing unrelated syntax error
  in `src/gen/query_gen/query_gen_conv.py:105`.
- Complete unit suite: the local interpreter lacks existing project
  dependencies including FastMCP, NumPy, and NetworkX.

## Migrations and risks

No existing artifact is migrated. Rerunning dataset preparation adds
`dataset_info.json` and run-specific training metadata beneath that run.
LlamaFactory outputs remain backward-independent because no shared YAML is
mutated.

The exact installed LlamaFactory 0.9.4 build must parse the generated YAML and
load both dataset entries on MoLab. Peak VRAM and checkpoint reload remain real
GPU acceptance gates; local structural validation is not a substitute.

## Exact next task

On MoLab, stop vLLM, prepare a real pilot dataset, run the documented 20-step
4B smoke and one-step resume verification, then perform the 8B batch-2 memory
probe (falling back to batch 1 only on OOM or peak above 90 GB). Record the
installed versions, loss, steps, peak VRAM, wall time, and resulting artifact
hashes before starting the full epoch.
