"""Unit tests for Phase 11 benchmark records and comparisons."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mini.artifacts import append_jsonl
from src.mini.benchmark import (
    BenchmarkError,
    _safe_error_message,
    benchmark_generation_workers,
    benchmark_serving_context,
    measure_operation,
    render_comparison_markdown,
    render_generation_matrix_markdown,
    render_serving_comparison_markdown,
    render_serving_context_markdown,
)
from src.mini.synthesize import RunPaths


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "mini" / "pipeline.toml"


def test_measure_operation_records_both_clocks_and_process_resources():
    value, resources = measure_operation(lambda: (time.sleep(0.03), "done")[1])

    assert value == "done"
    assert resources.wall_seconds >= 0.02
    assert resources.monotonic_seconds >= 0.02
    assert resources.peak_rss_bytes > 0
    assert resources.sample_count >= 1
    assert resources.child_processes_peak >= resources.child_processes_before
    assert resources.child_processes_after >= 0


def test_benchmark_errors_redact_environment_secrets(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "super-secret-value")
    rendered = _safe_error_message(
        RuntimeError("request api_key=super-secret-value could not complete")
    )
    assert "super-secret-value" not in rendered
    assert "api_key=[REDACTED]" in rendered


def test_comparison_is_source_labelled_and_calculates_duration_change():
    before = {
        "stage": "catalog_registration",
        "label": "unbounded",
        "registration_concurrency": None,
        "summary": {
            "monotonic_seconds_median": 2.0,
            "monotonic_seconds_p95": 2.2,
            "peak_rss_bytes_max": 200,
            "child_processes_peak_max": 8,
            "child_processes_after_max": 0,
        },
    }
    after = {
        "stage": "catalog_registration",
        "label": "bounded-2",
        "registration_concurrency": 2,
        "summary": {
            "monotonic_seconds_median": 2.5,
            "monotonic_seconds_p95": 2.6,
            "peak_rss_bytes_max": 150,
            "child_processes_peak_max": 2,
            "child_processes_after_max": 0,
        },
    }

    rendered = render_comparison_markdown(before, after)

    assert "| Label | unbounded | bounded-2 |" in rendered
    assert "| Peak child processes | 8 | 2 |" in rendered
    assert "+0.5000 seconds (+25.0%)" in rendered


def test_comparison_rejects_different_stages():
    with pytest.raises(BenchmarkError, match="same stage"):
        render_comparison_markdown(
            {"stage": "catalog_registration"}, {"stage": "graph_build"}
        )


def test_generation_worker_matrix_uses_identical_inputs_and_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    calls = []
    preflight = SimpleNamespace(
        graph_result=SimpleNamespace(manifest={"output_sha256": "graph-sha"}),
        catalog=SimpleNamespace(digest="catalog-sha"),
        model_identity="teacher-model",
    )

    async def runner(config, *, target, workers, run_id, preflight):
        calls.append((target, workers, run_id, preflight))
        paths = RunPaths.for_run(config, run_id)
        for index in range(target):
            append_jsonl(
                paths.events,
                {
                    "operation": "trajectory_completed",
                    "duration": workers + index / 10,
                },
            )
        return SimpleNamespace(
            run_id=run_id,
            state="completed",
            target_trajectories=target,
            completed_seeds=list(range(target)),
            failed_seeds=[],
            attempted_count=target,
            retried_count=0,
            failure_summary={},
            trajectory_rate_per_hour=float(workers * 100),
        )

    report = benchmark_generation_workers(
        CONFIG_PATH,
        label="smoke",
        workers=(1, 2),
        target=3,
        _preflight=preflight,
        _runner=runner,
    )

    assert [(target, workers) for target, workers, _, _ in calls] == [(3, 1), (3, 2)]
    assert len({run_id for _, _, run_id, _ in calls}) == 2
    assert all(value is preflight for _, _, _, value in calls)
    assert report["inputs"]["worker_counts"] == [1, 2]
    assert report["inputs"]["seeds_sha256"]
    assert [item["workers"] for item in report["summary"]["by_worker_count"]] == [1, 2]
    assert report["iterations"][0]["trajectory_duration_seconds_p95"] == pytest.approx(1.19)
    rendered = render_generation_matrix_markdown(report)
    assert "| 1 | 1 |" in rendered
    assert "| 2 | 1 |" in rendered
    assert "same run seed, target, graph, catalog, and model identity" in rendered


def test_generation_worker_matrix_stops_and_returns_partial_failure_report(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    preflight = SimpleNamespace(
        graph_result=SimpleNamespace(manifest={"output_sha256": "graph-sha"}),
        catalog=SimpleNamespace(digest="catalog-sha"),
        model_identity="teacher-model",
    )
    calls = []

    async def runner(config, *, target, workers, run_id, preflight):
        calls.append(workers)
        return SimpleNamespace(
            run_id=run_id,
            state="failed",
            target_trajectories=target,
            completed_seeds=[],
            failed_seeds=list(range(target)),
            attempted_count=target,
            retried_count=0,
            failure_summary={"TimeoutError": target},
            trajectory_rate_per_hour=0.0,
        )

    report = benchmark_generation_workers(
        CONFIG_PATH,
        label="partial",
        workers=(1, 2, 4),
        target=2,
        _preflight=preflight,
        _runner=runner,
    )

    assert calls == [1]
    assert report["summary"]["complete"] is False
    assert report["summary"]["aborted_after_run_id"] == report["iterations"][0]["run_id"]
    assert report["iterations"][0]["failure_summary"] == {"TimeoutError": 2}
    assert "**Incomplete:**" in render_generation_matrix_markdown(report)


@pytest.mark.parametrize("workers", [(), (1, 1), (0,), (5,)])
def test_generation_worker_matrix_rejects_unsafe_worker_lists(workers):
    with pytest.raises(BenchmarkError, match="worker"):
        benchmark_generation_workers(
            CONFIG_PATH,
            label="invalid",
            workers=workers,
            _preflight=object(),
            _runner=lambda **_: None,
        )


def _write_completed_run(tmp_path: Path, queries_per_file: int = 2, files: int = 5) -> str:
    """Create a fake completed run whose node queries span real lengths."""
    completed = (
        tmp_path / "artifacts" / "runs" / "bench-src" / "trajectories" / "completed"
    )
    completed.mkdir(parents=True)
    for file_index in range(files):
        nodes = [
            {"query": "q" * (10 * (file_index * queries_per_file + node_index) + 1)}
            for node_index in range(queries_per_file)
        ]
        (completed / f"{file_index}.json").write_text(
            json.dumps({"nodes": nodes, "seed": file_index}), encoding="utf-8"
        )
    return "bench-src"


def test_serving_context_replays_real_prompts_and_records_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    run_id = _write_completed_run(tmp_path)
    seen_prompts = []

    def health(config):
        return {
            "returned_model": "served-teacher",
            "base_url": config.teacher.base_url,
            "reported_max_model_len": 16384,
        }

    def post(prompt):
        seen_prompts.append(prompt)
        return {
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 8},
            "model": "served-teacher",
        }

    report = benchmark_serving_context(
        CONFIG_PATH,
        label="ctx16k",
        run_id=run_id,
        requests=6,
        _health=health,
        _post_chat=post,
    )

    assert report["stage"] == "serving_context"
    assert report["server"]["served_model"] == "served-teacher"
    assert report["server"]["reported_max_model_len"] == 16384
    assert report["server"]["configured_max_num_seqs"] == 4
    assert report["inputs"]["source_run_id"] == run_id
    assert report["inputs"]["prompt_pool_size"] == 10
    assert report["inputs"]["temperature"] == 0
    assert report["inputs"]["thinking_enabled"] is False
    # Prompts are replayed in deterministic length-percentile order.
    assert report["inputs"]["selected_prompt_chars"] == sorted(
        report["inputs"]["selected_prompt_chars"]
    )
    assert [len(prompt) for prompt in seen_prompts] == sorted(
        len(prompt) for prompt in seen_prompts
    )
    assert len(seen_prompts) == 6
    # No prompt text may appear anywhere in the serialized report.
    serialized = json.dumps(report)
    for prompt in seen_prompts:
        assert prompt not in serialized
    assert all(record["error"] is None for record in report["requests"])
    assert report["summary"]["successful_requests"] == 6
    assert report["summary"]["failed_requests"] == 0
    assert report["summary"]["prompt_tokens_total"] == sum(
        len(prompt) for prompt in seen_prompts
    )
    assert report["summary"]["latency_seconds_p50"] is not None
    rendered = render_serving_context_markdown(report)
    assert "| Reported context | 16384 |" in rendered
    assert "| Latency p95 (s) |" in rendered
    for prompt in seen_prompts:
        assert prompt not in rendered


def test_serving_context_records_over_context_failures_as_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    run_id = _write_completed_run(tmp_path)

    def health(config):
        return {"returned_model": "m", "reported_max_model_len": 8192}

    def post(prompt):
        if len(prompt) > 50:
            raise OSError("simulated over-context HTTP 400")
        return {"usage": {"prompt_tokens": len(prompt), "completion_tokens": 4}}

    report = benchmark_serving_context(
        CONFIG_PATH,
        label="ctx8k",
        run_id=run_id,
        requests=10,
        _health=health,
        _post_chat=post,
    )

    assert report["summary"]["complete"] is True
    assert report["summary"]["failed_requests"] > 0
    assert report["summary"]["successful_requests"] > 0
    failed = [record for record in report["requests"] if record["error"] is not None]
    assert all(record["error"] == "OSError" for record in failed)
    assert all(record["latency_seconds"] is not None for record in failed)
    assert report["summary"]["error_kinds"] == {"OSError": len(failed)}
    assert render_serving_context_markdown(report)


def test_serving_context_concurrency_is_recorded_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    run_id = _write_completed_run(tmp_path)

    with pytest.raises(BenchmarkError, match="concurrency"):
        benchmark_serving_context(
            CONFIG_PATH,
            label="bad",
            run_id=run_id,
            concurrency=9,
            _health=lambda config: {},
            _post_chat=lambda prompt: {},
        )

    report = benchmark_serving_context(
        CONFIG_PATH,
        label="conc2",
        run_id=run_id,
        requests=4,
        concurrency=2,
        _health=lambda config: {},
        _post_chat=lambda prompt: {"usage": {}},
    )
    assert report["inputs"]["concurrency"] == 2
    assert all(record["prompt_tokens"] is None for record in report["requests"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requests": 0},
        {"max_tokens": 0},
        {"concurrency": 0},
    ],
)
def test_serving_context_rejects_invalid_measurement_inputs(tmp_path, kwargs):
    with pytest.raises(BenchmarkError):
        benchmark_serving_context(
            CONFIG_PATH,
            label="invalid",
            run_id="whatever",
            _health=lambda config: {},
            _post_chat=lambda prompt: {},
            **kwargs,
        )


def test_serving_context_requires_completed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    with pytest.raises(BenchmarkError, match="completed trajectory directory"):
        benchmark_serving_context(
            CONFIG_PATH,
            label="missing",
            run_id="no-such-run",
            _health=lambda config: {},
            _post_chat=lambda prompt: {},
        )


def test_serving_comparison_renders_deltas_and_rejects_stage_mismatch():
    def report(label, p50, p95, context):
        return {
            "stage": "serving_context",
            "label": label,
            "server": {
                "served_model": "Qwen/Qwen3-14B",
                "reported_max_model_len": context,
                "configured_max_model_len": context,
                "configured_max_num_seqs": 4,
                "configured_gpu_memory_utilization": 0.85,
            },
            "inputs": {"source_run_id": "r", "requests": 20, "concurrency": 1},
            "summary": {
                "latency_seconds_p50": p50,
                "latency_seconds_p95": p95,
                "successful_requests": 20,
                "failed_requests": 0,
                "prompt_tokens_total": 1000,
                "completion_tokens_total": 200,
            },
        }

    before = report("ctx8k", 1.0, 2.0, 8192)
    after = report("ctx16k", 0.8, 1.6, 16384)
    rendered = render_serving_comparison_markdown(before, after)
    assert "| Label | ctx8k | ctx16k |" in rendered
    assert "| Configured context | 8192 | 16384 |" in rendered
    assert "| Latency p50 (s) | 1.0000 | 0.8000 |" in rendered
    assert "-0.2000 seconds (-20.0%)" in rendered
    with pytest.raises(BenchmarkError, match="same stage"):
        render_serving_comparison_markdown(before, {"stage": "other"})
