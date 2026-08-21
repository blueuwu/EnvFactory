"""Smoke: offline mini pipeline spine (plan §13).

Runs config -> catalog -> graph build -> dataset conversion hermetically in
process with stubbed embeddings, classifier, and tokenizer. Fixture inputs
live under the pytest temp dir, which also exercises recording inputs that
reside outside the repository root.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.mini.build_graph import build_graph
from src.mini.catalog import CatalogReport, CatalogServer
from src.mini.config import load_config
from src.mini.manifest import RunManifest
from src.mini.prepare_dataset import prepare_dataset

pytestmark = pytest.mark.smoke

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "mini" / "pipeline.toml"

SERVER_TOOLS = (
    ("Calculator", "calculate"),
    ("Calendar", "list_events"),
    ("Weather", "get_weather"),
)


class StableEmbeddingBackend:
    identity = "smoke:stable-embedding"

    def encode(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) - 127.5
            rows.append(row)
        return np.stack(rows)


class AlwaysUserProvided:
    identity = "smoke:stable-classifier"
    settings = {
        "temperature": 0.0,
        "seed": 42,
        "thinking": False,
        "response_format": "json_schema",
    }

    def classify(self, parameters, param_to_tool_map=None):
        return [True] * len(parameters)


class FakeTokenizer:
    chat_template = "smoke-fake-chat-template"
    init_kwargs = {"_commit_hash": "smoke-tokenizer-commit"}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is False
        return list(range(128))


def _metadata(server: str, tool: str) -> dict:
    return {
        "class_name": server,
        "description": f"Smoke {server} server",
        "tools": [
            {
                "name": tool,
                "description": f"Smoke {tool}",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer", "description": "A value"}},
                    "required": ["value"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string", "description": "A result"}},
                },
            }
        ],
    }


def _catalog(fixture_dir: Path) -> CatalogReport:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    servers = []
    for server_name, tool_name in SERVER_TOOLS:
        metadata = _metadata(server_name, tool_name)
        metadata_path = fixture_dir / f"{server_name}_metadata.json"
        tool_path = fixture_dir / f"{server_name}.py"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        tool_path.write_text("# smoke fixture\n", encoding="utf-8")
        servers.append(
            CatalogServer(
                name=server_name,
                metadata_path=metadata_path,
                tool_path=tool_path,
                metadata=metadata,
                metadata_tool_names=(tool_name,),
                registered_tool_names=(tool_name, "load_scenario", "save_scenario"),
            )
        )
    return CatalogReport(servers=tuple(servers), digest="smoke-catalog")


def _trajectory(seed: int, server: str, tool: str) -> dict:
    full_name = f"{server}-{tool}"
    return {
        "seed": seed,
        "scenario": "smoke",
        "user_tools": None,
        "user_profile": None,
        "sampled_tool_names": [full_name],
        "mcp_servers": [server],
        "nodes": [
            {
                "raw_tool_call": [full_name],
                "initial_scenario": {server: {"value": 0}},
                "final_scenario": {server: {"value": 1}},
                "query": f"Run seed {seed}",
                "decision": True,
                "accuracy": 1.0,
                "pass_k_trace": {},
                "pass_k_scenario": {},
                "pass_k_decision": {},
                "mcp_servers": [server],
                "steps": [
                    {"role": "user", "content": f"Run seed {seed}"},
                    {
                        "role": "tool_call",
                        "content": [{"name": full_name, "arguments": {"value": seed}}],
                    },
                    {"role": "tool_response", "content": [f"result {seed}"]},
                    {"role": "assistant", "content": f"Finished {seed}"},
                ],
            }
        ],
    }


def test_offline_spine_config_catalog_graph_dataset(tmp_path) -> None:
    started = time.perf_counter()

    # Stage 1: configuration rooted entirely inside the temp dir.
    artifact_root = tmp_path / "artifacts"
    config = load_config(CONFIG_PATH)
    graph_config = config.graph.model_copy(
        update={
            "path": artifact_root / "graph" / "graph.pkl",
            "manifest_path": artifact_root / "graph" / "manifest.json",
            "embedding_cache": artifact_root / "cache" / "embeddings.sqlite3",
            "user_provided_cache": artifact_root / "cache" / "user.sqlite3",
        }
    )
    config = config.model_copy(update={"artifact_root": artifact_root, "graph": graph_config})

    # Stage 2: catalog over fixture servers outside the repository root.
    catalog = _catalog(tmp_path / "inputs")

    # Stage 3: deterministic graph build with stubbed live computation.
    result = build_graph(
        config,
        catalog=catalog,
        embedding_backend=StableEmbeddingBackend(),
        user_classifier=AlwaysUserProvided(),
    )
    assert result.cached is False
    assert result.manifest["output_sha256"] == hashlib.sha256(
        config.graph.path.read_bytes()
    ).hexdigest()

    # Stage 4: dataset conversion over a completed fixture run.
    run_id = "smoke-spine"
    run_root = artifact_root / "runs" / run_id
    completed = run_root / "trajectories" / "completed"
    completed.mkdir(parents=True)
    seeds = [30, 2, 11, 7, 19, 5]
    for seed, (server, tool) in zip(seeds, SERVER_TOOLS * 2, strict=True):
        (completed / f"{seed}.json").write_text(
            json.dumps(_trajectory(seed, server, tool)), encoding="utf-8"
        )
    manifest = RunManifest.create(
        run_id=run_id,
        config_sha256="smoke-config",
        graph_sha256=result.manifest["output_sha256"],
        catalog_sha256=catalog.digest,
        teacher_model=config.teacher.model,
        target_trajectories=len(seeds),
        seeds=seeds,
        git_commit=None,
        git_dirty=False,
    )
    manifest.state = "completed"
    manifest.completed_seeds = list(seeds)
    manifest.valid_count = len(seeds)
    manifest.checkpoint(run_root / "run_manifest.json")

    prepared = prepare_dataset(config, run_id, tokenizer=FakeTokenizer(), catalog=catalog)

    counts = prepared.manifest["counts"]
    assert counts["source_trajectories"] == len(seeds)
    assert counts["retained_trajectories"] == len(seeds)
    assert counts["train_trajectories"] + counts["validation_trajectories"] == len(seeds)
    assert counts["validation_trajectories"] >= 1
    assert prepared.dataset_info_path.is_file()
    assert prepared.training_profile_path.is_file()

    elapsed = time.perf_counter() - started
    assert elapsed < 60, f"offline spine took {elapsed:.1f}s (budget is 60s)"
