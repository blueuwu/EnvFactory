"""Integration: serving-context benchmark over a real loopback HTTP endpoint.

Exercises the un-mocked health check and chat-completion poster path end to
end against an in-process server on an ephemeral port, including over-context
failures recorded as data (plan Phase 11 instrument contract).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src.mini.benchmark import (
    benchmark_serving_context,
    render_serving_context_markdown,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "mini" / "pipeline.toml"

MAX_PROMPT_CHARS = 25


class _StubHandler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            if self.headers.get("Authorization") != "Bearer loopback-key":
                self._send({"error": "unauthorized"}, 401)
                return
            self._send({"data": [{"id": "Qwen/Qwen3-14B", "max_model_len": 16384}]})
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = request["messages"][0]["content"]
        if len(prompt) > MAX_PROMPT_CHARS:
            self._send({"error": "prompt exceeds context"}, 400)
            return
        if (
            request.get("temperature") != 0
            or request.get("chat_template_kwargs", {}).get("enable_thinking") is not False
        ):
            self._send({"error": "decoding settings must be pinned"}, 400)
            return
        self._send(
            {
                "model": request["model"],
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": request["max_tokens"],
                },
            }
        )

    def log_message(self, *args) -> None:
        pass


def _write_completed_run(tmp_path: Path) -> str:
    completed = tmp_path / "artifacts" / "runs" / "loopback-src" / "trajectories" / "completed"
    completed.mkdir(parents=True)
    lengths = (5, 15, 30, 60)
    for index, length in enumerate(lengths):
        nodes = [{"query": "q" * length}]
        (completed / f"{index}.json").write_text(
            json.dumps({"nodes": nodes, "seed": index}), encoding="utf-8"
        )
    return "loopback-src"


@pytest.fixture()
def loopback_endpoint():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_serving_context_end_to_end_over_real_loopback(
    tmp_path, monkeypatch, loopback_endpoint
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "loopback-key")
    monkeypatch.setenv(
        "ENVFACTORY_MINI_TEACHER_BASE_URL", f"http://127.0.0.1:{loopback_endpoint}/v1"
    )
    monkeypatch.setenv("ENVFACTORY_MINI_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    run_id = _write_completed_run(tmp_path)

    report = benchmark_serving_context(
        CONFIG_PATH,
        label="loopback-e2e",
        run_id=run_id,
        requests=4,
        max_tokens=16,
        timeout=10.0,
    )

    assert report["server"]["served_model"] == "Qwen/Qwen3-14B"
    assert report["server"]["reported_max_model_len"] == 16384
    # Two shortest prompts succeed; the two longest hit the stub context limit.
    assert report["summary"]["successful_requests"] == 2
    assert report["summary"]["failed_requests"] == 2
    assert report["summary"]["error_kinds"] == {"HTTPError": 2}
    failed = [record for record in report["requests"] if record["error"]]
    assert all(record["http_status"] == 400 for record in failed)
    assert report["summary"]["complete"] is True
    serialized = json.dumps(report)
    assert "q" * 5 not in serialized, "prompt text must never reach reports"

    rendered = render_serving_context_markdown(report)
    assert "| Failed requests | 2 |" in rendered
