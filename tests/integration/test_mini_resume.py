from __future__ import annotations

import asyncio
import json
from pathlib import Path

import networkx as nx
import pytest

from src.graph.tool_chain import ToolQueryChain, ToolQueryNode
from src.graph.tool_node import Tool
from src.mini.build_graph import GraphBuildResult
from src.mini.catalog import CatalogReport, CatalogServer
from src.mini.config import load_config
from src.mini.synthesize import (
    DISK_STOP_BYTES,
    DISK_WARN_BYTES,
    PILOT_MIN_ATTEMPTS,
    SynthesisPreflight,
    SynthesisError,
    config_compatibility_hash,
    make_run_id,
    synthesize,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "mini" / "pipeline.toml"


class FakeGraph:
    def __init__(self, tool: Tool):
        self.tool = tool
        self.graph = nx.DiGraph()
        self.graph.add_node(tool)

    def sample(self, sampler, max_nodes: int, seed: int) -> ToolQueryChain:
        return ToolQueryChain([self.tool], seed=seed)


class FakeGenerator:
    def __init__(self, stop: asyncio.Event | None = None, counter: list[int] | None = None):
        self.stop = stop
        self.counter = counter
        self.reset_count = 0
        self.gen_count = 0

    def reset_run_state(self) -> None:
        self.reset_count += 1

    async def gen(self, chain: ToolQueryChain) -> ToolQueryChain:
        self.gen_count += 1
        await asyncio.sleep(0)
        if self.counter is not None:
            self.counter[0] += 1
            if self.counter[0] == 3 and self.stop is not None:
                self.stop.set()
        chain.scenario = "test scenario"
        chain.tool_chain = [
            ToolQueryNode(
                raw_tool_call=list(chain.init_tool_chain),
                initial_scenario={"Tiny": {"value": 0}},
                final_scenario={"Tiny": {"value": 1}},
                query="Ping once",
                decision=True,
                steps=[
                    {"role": "user", "content": "Ping once"},
                    {
                        "role": "tool_call",
                        "content": [{"name": "Tiny-ping", "arguments": {}}],
                    },
                    {"role": "tool_response", "content": ["pong"]},
                    {"role": "assistant", "content": "Done"},
                ],
            )
        ]
        return chain


class AlwaysFailingGenerator(FakeGenerator):
    """Every attempt fails transiently, so each seed burns all its attempts."""

    async def gen(self, chain: ToolQueryChain) -> ToolQueryChain:
        raise TimeoutError("transport timeout while serving")


class TransientOnceGenerator(FakeGenerator):
    def __init__(self, attempts: dict[int, int]):
        super().__init__()
        self.attempts = attempts

    async def gen(self, chain: ToolQueryChain) -> ToolQueryChain:
        count = self.attempts.get(chain.seed, 0) + 1
        self.attempts[chain.seed] = count
        if count == 1:
            raise TimeoutError("temporary transport timeout; api_key=must-not-leak")
        return await super().gen(chain)


def _fixture(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    generation = config.generation.model_copy(update={"max_attempts_per_seed": 2})
    config = config.model_copy(
        update={"artifact_root": tmp_path / "artifacts", "generation": generation}
    )
    tool = Tool(
        {
            "server": "Tiny",
            "name": "Tiny-ping",
            "description": "Ping",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {}},
        }
    )
    graph = FakeGraph(tool)
    server = CatalogServer(
        name="Tiny",
        metadata_path=tmp_path / "Tiny_metadata.json",
        tool_path=tmp_path / "Tiny.py",
        metadata={"class_name": "Tiny", "tools": [{"name": "ping"}]},
        metadata_tool_names=("ping",),
        registered_tool_names=("ping", "load_scenario", "save_scenario"),
    )
    catalog = CatalogReport(servers=(server,), digest="catalog-hash")
    preflight = SynthesisPreflight(
        graph_result=GraphBuildResult(
            graph=graph,
            manifest={"output_sha256": "graph-hash"},
            cached=True,
        ),
        catalog=catalog,
        environment={"packages": {}, "gpu": {}, "resources": {"cpu_count": 4}},
        model_identity=config.teacher.model,
        config_sha256=config_compatibility_hash(config),
    )
    return config, preflight


def test_interruption_resume_and_completed_noop_are_exact(tmp_path) -> None:
    config, preflight = _fixture(tmp_path)
    run_id = "resume-test"
    stop = asyncio.Event()
    counter = [0]

    interrupted = asyncio.run(
        synthesize(
            config,
            target=10,
            workers=2,
            run_id=run_id,
            preflight=preflight,
            generator_factory=lambda _: FakeGenerator(stop, counter),
            stop_event=stop,
            initialize_mcp=False,
            monitor_interval=0.01,
        )
    )
    assert interrupted.state == "interrupted"
    assert 0 < len(interrupted.completed_seeds) < 10

    completed = asyncio.run(
        synthesize(
            config,
            target=10,
            workers=2,
            run_id=run_id,
            resume=True,
            preflight=preflight,
            generator_factory=lambda _: FakeGenerator(),
            initialize_mcp=False,
            monitor_interval=0.01,
        )
    )
    assert completed.state == "completed"
    assert len(completed.completed_seeds) == 10
    assert len(set(completed.completed_seeds)) == 10

    completed_dir = config.artifact_root / "runs" / run_id / "trajectories" / "completed"
    assert len(list(completed_dir.glob("*.json"))) == 10
    for path in completed_dir.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["seed"] == int(path.stem)

    def forbidden_factory(_: int):
        raise AssertionError("completed resume must not initialize inference")

    noop = asyncio.run(
        synthesize(
            config,
            target=10,
            workers=2,
            run_id=run_id,
            resume=True,
            preflight=preflight,
            generator_factory=forbidden_factory,
            initialize_mcp=False,
        )
    )
    assert noop.state == "completed"


def test_transient_failures_are_preserved_then_retried(tmp_path) -> None:
    config, preflight = _fixture(tmp_path)
    attempts: dict[int, int] = {}
    manifest = asyncio.run(
        synthesize(
            config,
            target=3,
            workers=2,
            run_id="retry-test",
            preflight=preflight,
            generator_factory=lambda _: TransientOnceGenerator(attempts),
            initialize_mcp=False,
            monitor_interval=0.01,
        )
    )
    assert manifest.state == "completed"
    assert manifest.attempted_count == 6
    assert manifest.retried_count == 3
    assert manifest.valid_count == 3
    assert not manifest.failed_seeds
    failure_dir = config.artifact_root / "runs" / "retry-test" / "trajectories" / "failed"
    for path in failure_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))["failures"][0]
        assert record["retryable"] is True
        assert "must-not-leak" not in record["safe_message"]


def test_workers_reuse_one_reset_generator_and_close_clients(tmp_path, monkeypatch) -> None:
    from src.manager.mcp_client_manager import MCPManager

    config, preflight = _fixture(tmp_path)
    generators: list[FakeGenerator] = []
    close_calls = []

    def factory(_: int) -> FakeGenerator:
        generator = FakeGenerator()
        generators.append(generator)
        return generator

    monkeypatch.setattr(MCPManager, "init_config", lambda *args, **kwargs: None)

    async def close_all_clients() -> None:
        close_calls.append(True)

    monkeypatch.setattr(MCPManager, "aclose_all_clients", close_all_clients)
    manifest = asyncio.run(
        synthesize(
            config,
            target=6,
            workers=2,
            run_id="worker-reuse-test",
            preflight=preflight,
            generator_factory=factory,
            monitor_interval=0.01,
        )
    )

    assert manifest.state == "completed"
    assert len(generators) == 2
    assert sum(generator.gen_count for generator in generators) == 6
    assert any(generator.gen_count > 1 for generator in generators)
    assert sum(generator.reset_count for generator in generators) == 12
    assert close_calls == [True]


def test_resume_refuses_semantic_configuration_change(tmp_path) -> None:
    config, preflight = _fixture(tmp_path)
    asyncio.run(
        synthesize(
            config,
            target=1,
            workers=1,
            run_id="compatibility-test",
            preflight=preflight,
            generator_factory=lambda _: FakeGenerator(),
            initialize_mcp=False,
        )
    )
    changed = config.model_copy(
        update={"generation": config.generation.model_copy(update={"pass_k": 99})}
    )
    changed_preflight = SynthesisPreflight(
        graph_result=preflight.graph_result,
        catalog=preflight.catalog,
        environment=preflight.environment,
        model_identity=preflight.model_identity,
        config_sha256=config_compatibility_hash(changed),
    )
    with pytest.raises(SynthesisError, match="configuration"):
        asyncio.run(
            synthesize(
                changed,
                target=1,
                workers=1,
                run_id="compatibility-test",
                resume=True,
                preflight=changed_preflight,
                generator_factory=lambda _: FakeGenerator(),
                initialize_mcp=False,
            )
        )


def _events(config, run_id: str) -> list[dict]:
    path = config.artifact_root / "runs" / run_id / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_pilot_circuit_breaker_fails_run_above_server_failure_rate(tmp_path) -> None:
    config, preflight = _fixture(tmp_path)
    # One server, two attempts per seed: 20 failed attempts trip the 20%
    # breaker exactly at PILOT_MIN_ATTEMPTS with a 100% Tiny failure rate.
    with pytest.raises(SynthesisError, match="circuit breaker"):
        asyncio.run(
            synthesize(
                config,
                target=30,
                workers=1,
                run_id="breaker-test",
                preflight=preflight,
                generator_factory=lambda _: AlwaysFailingGenerator(),
                initialize_mcp=False,
                monitor_interval=0.01,
            )
        )

    manifest = json.loads(
        (config.artifact_root / "runs" / "breaker-test" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["state"] == "failed"
    assert manifest["completed_seeds"] == []
    # The breaker fires on the exact PILOT_MIN_ATTEMPTS-th failed attempt:
    # ten seeds burning two transient attempts each.
    assert sum(manifest["attempts_by_seed"].values()) == PILOT_MIN_ATTEMPTS
    completed_dir = (
        config.artifact_root / "runs" / "breaker-test" / "trajectories" / "completed"
    )
    assert list(completed_dir.glob("*.json")) == []
    failures = [
        record
        for record in _events(config, "breaker-test")
        if record["operation"] == "trajectory_failed" or record.get("outcome") == "failed_attempt"
    ]
    assert failures, "failure events must be recorded before the breaker fires"
    states = [record for record in _events(config, "breaker-test") if record["operation"] == "run_state"]
    assert states[-1]["outcome"] == "failed"
    assert states[-1]["exception_class"] == "SynthesisError"


def test_disk_pressure_stops_new_trajectories_and_warns_below_thresholds(
    tmp_path, monkeypatch
) -> None:
    import shutil as shutil_module

    config, preflight = _fixture(tmp_path)

    class FakeUsage:
        def __init__(self, free: int):
            self.free = free

    # Stop level: below DISK_STOP_BYTES the producer halts and the run ends
    # interrupted without reaching the target.
    monkeypatch.setattr(
        shutil_module, "disk_usage", lambda _root: FakeUsage(DISK_STOP_BYTES - 1024)
    )
    manifest = asyncio.run(
        synthesize(
            config,
            target=10,
            workers=1,
            run_id="disk-stop-test",
            preflight=preflight,
            generator_factory=lambda _: FakeGenerator(),
            initialize_mcp=False,
            monitor_interval=0.01,
        )
    )
    assert manifest.state == "interrupted"
    assert len(manifest.completed_seeds) < 10
    stop_events = [
        record
        for record in _events(config, "disk-stop-test")
        if record["operation"] == "disk_pressure"
        and record["outcome"] == "stopping_new_trajectories"
    ]
    assert stop_events and stop_events[0]["free_bytes"] < DISK_STOP_BYTES

    # Warn level: between thresholds generation continues but a warning is
    # recorded exactly once.
    monkeypatch.setattr(
        shutil_module,
        "disk_usage",
        lambda _root: FakeUsage((DISK_STOP_BYTES + DISK_WARN_BYTES) // 2),
    )
    manifest = asyncio.run(
        synthesize(
            config,
            target=2,
            workers=1,
            run_id="disk-warn-test",
            preflight=preflight,
            generator_factory=lambda _: FakeGenerator(),
            initialize_mcp=False,
            monitor_interval=0.01,
        )
    )
    assert manifest.state == "completed"
    warnings = [
        record
        for record in _events(config, "disk-warn-test")
        if record["operation"] == "disk_pressure" and record["outcome"] == "warning"
    ]
    assert len(warnings) == 1


def test_new_run_suffixes_past_existing_manifest_instead_of_refusing(
    tmp_path, monkeypatch
) -> None:
    config, preflight = _fixture(tmp_path)
    base = make_run_id(config, preflight.config_sha256)
    # Simulate an existing run occupying the natural next ID.
    occupied = config.artifact_root / "runs" / base / "run_manifest.json"
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_text("{}", encoding="utf-8")

    with pytest.raises(SynthesisError, match="run already exists; use --resume"):
        asyncio.run(
            synthesize(
                config,
                target=1,
                workers=1,
                run_id=base,
                preflight=preflight,
                generator_factory=lambda _: FakeGenerator(),
                initialize_mcp=False,
                monitor_interval=0.01,
            )
        )

    manifest = asyncio.run(
        synthesize(
            config,
            target=1,
            workers=1,
            new_run=True,
            preflight=preflight,
            generator_factory=lambda _: FakeGenerator(),
            initialize_mcp=False,
            monitor_interval=0.01,
        )
    )
    assert manifest.state == "completed"
    assert manifest.run_id == f"{base}-2"
    assert occupied.read_text(encoding="utf-8") == "{}", "existing run must stay untouched"
    assert (config.artifact_root / "runs" / manifest.run_id / "trajectories" / "completed").is_dir()
