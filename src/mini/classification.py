"""Deterministic teacher classification and its content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from src.graph.tool_node import Parameter, Tool


class ClassificationError(RuntimeError):
    """Raised when user-provided classification cannot be completed safely."""


class UserProvidedClassifier(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def settings(self) -> dict[str, Any]: ...

    def classify(
        self,
        parameters: list[Parameter],
        param_to_tool_map: dict[Parameter, Tool] | None = None,
    ) -> list[bool]: ...


def _classification_payload(parameter: Parameter, tool: Tool | None) -> dict[str, Any]:
    return {
        "prompt_version": 1,
        "parameter": {
            "name": parameter.name.strip(),
            "data_type": parameter.data_type.strip(),
            "description": " ".join(parameter.description.split()),
        },
        "tool": None
        if tool is None
        else {
            "server": tool.server.strip(),
            "name": tool.name.strip(),
            "description": " ".join(tool.description.split()),
        },
    }


def classification_prompt_hash(parameter: Parameter, tool: Tool | None) -> str:
    """Hash the canonical parameter/tool content from which the prompt is made."""
    normalized = json.dumps(
        _classification_payload(parameter, tool),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _extract_classifications(parsed: object, batch: list[Parameter]) -> list[bool]:
    """Accept the requested schema and unambiguous single-item fallback shapes."""
    values: object | None = None
    if isinstance(parsed, dict):
        values = parsed.get("classifications")
        if values is None and len(batch) == 1:
            direct = parsed.get("can_directly_provide")
            if type(direct) is bool:
                values = [direct]
            else:
                nested = parsed.get(batch[0].name.strip())
                if type(nested) is bool:
                    values = [nested]
                elif isinstance(nested, dict):
                    nested_direct = nested.get("can_directly_provide")
                    if type(nested_direct) is bool:
                        values = [nested_direct]
    if (
        not isinstance(values, list)
        or len(values) != len(batch)
        or any(type(value) is not bool for value in values)
    ):
        raise ValueError("teacher returned invalid classifications")
    return values


class TeacherUserProvidedClassifier:
    """Classify parameters through an OpenAI-compatible local teacher endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        seed: int,
        batch_size: int = 1,
        max_attempts: int = 3,
        client: object | None = None,
    ) -> None:
        if not base_url or not api_key or not model:
            raise ClassificationError("teacher classification requires base URL, API key, and model")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.__api_key = api_key
        self._client = client

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "max_attempts": self.max_attempts,
            "thinking": False,
            "response_format": "json_schema",
            "max_tokens": 512,
        }

    @property
    def identity(self) -> str:
        settings_hash = hashlib.sha256(
            json.dumps(self.settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"teacher:{self.model}@{self.base_url}#{settings_hash}"

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ClassificationError("the OpenAI client is required for teacher classification") from exc
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    @property
    def _api_key(self) -> str:
        # Kept off identity/settings/manifests and never exposed in exceptions.
        return self.__api_key

    def classify(
        self,
        parameters: list[Parameter],
        param_to_tool_map: dict[Parameter, Tool] | None = None,
    ) -> list[bool]:
        if not parameters:
            return []
        results: list[bool] = []
        for start in range(0, len(parameters), self.batch_size):
            batch = parameters[start : start + self.batch_size]
            items = [
                _classification_payload(
                    parameter,
                    param_to_tool_map.get(parameter) if param_to_tool_map else None,
                )
                for parameter in batch
            ]
            prompt = (
                "Decide whether a normal user can directly provide each parameter from their "
                "request. Return true only when the value can be stated directly, not when it "
                "must be obtained from a tool. Return JSON matching the supplied schema.\n\n"
                + json.dumps(items, ensure_ascii=False, sort_keys=True)
            )
            schema = {
                "name": "user_provided_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "classifications": {
                            "type": "array",
                            "items": {"type": "boolean"},
                            "minItems": len(batch),
                            "maxItems": len(batch),
                        }
                    },
                    "required": ["classifications"],
                    "additionalProperties": False,
                },
            }
            last_error: Exception | None = None
            for _attempt in range(self.max_attempts):
                try:
                    request_options: dict[str, Any] = {}
                    if _attempt == 0:
                        request_options["response_format"] = {
                            "type": "json_schema",
                            "json_schema": schema,
                        }
                    elif _attempt == 1:
                        request_options["response_format"] = {"type": "json_object"}
                    response = self._get_client().chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        top_p=1.0,
                        seed=self.seed + _attempt,
                        max_tokens=512,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        **request_options,
                    )
                    content = response.choices[0].message.content
                    parsed = json.loads(content)
                    values = _extract_classifications(parsed, batch)
                    break
                except Exception as exc:
                    last_error = exc
            else:
                assert last_error is not None
                raise ClassificationError(
                    "teacher classification failed after "
                    f"{self.max_attempts} attempts with {type(last_error).__name__}"
                ) from last_error
            results.extend(values)
        return results


class CachedUserProvidedClassifier:
    """SQLite cache keyed by classifier identity and normalized prompt hash."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS user_provided (
            classifier_identity TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            value INTEGER NOT NULL CHECK (value IN (0, 1)),
            PRIMARY KEY (classifier_identity, prompt_sha256)
        )
    """

    def __init__(self, classifier: UserProvidedClassifier, path: str | Path) -> None:
        self.classifier = classifier
        self.path = Path(path)
        self.hits = 0
        self.misses = 0

    @property
    def identity(self) -> str:
        return self.classifier.identity

    @property
    def settings(self) -> dict[str, Any]:
        return self.classifier.settings

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute(self._SCHEMA)
        return connection

    def classify(
        self,
        parameters: list[Parameter],
        param_to_tool_map: dict[Parameter, Tool] | None = None,
    ) -> list[bool]:
        if not parameters:
            return []
        values_by_hash: dict[str, bool] = {}
        missing_by_hash: dict[str, Parameter] = {}
        tool_by_hash: dict[str, Tool] = {}
        hashes: list[str] = []

        with self._connect() as connection:
            for parameter in parameters:
                tool = param_to_tool_map.get(parameter) if param_to_tool_map else None
                prompt_hash = classification_prompt_hash(parameter, tool)
                hashes.append(prompt_hash)
                if prompt_hash in values_by_hash or prompt_hash in missing_by_hash:
                    continue
                row = connection.execute(
                    "SELECT value FROM user_provided "
                    "WHERE classifier_identity = ? AND prompt_sha256 = ?",
                    (self.identity, prompt_hash),
                ).fetchone()
                if row is None:
                    missing_by_hash[prompt_hash] = parameter
                    if tool is not None:
                        tool_by_hash[prompt_hash] = tool
                else:
                    values_by_hash[prompt_hash] = bool(row[0])

            self.hits += len(set(hashes)) - len(missing_by_hash)
            self.misses += len(missing_by_hash)
            if missing_by_hash:
                missing_hashes = list(missing_by_hash)
                missing_parameters = [missing_by_hash[key] for key in missing_hashes]
                missing_map = {
                    missing_by_hash[key]: tool_by_hash[key]
                    for key in missing_hashes
                    if key in tool_by_hash
                }
                classified = self.classifier.classify(missing_parameters, missing_map)
                if len(classified) != len(missing_parameters) or any(
                    type(value) is not bool for value in classified
                ):
                    raise ClassificationError("classifier returned invalid result count or values")
                for prompt_hash, value in zip(missing_hashes, classified, strict=True):
                    connection.execute(
                        "INSERT OR REPLACE INTO user_provided "
                        "(classifier_identity, prompt_sha256, value) VALUES (?, ?, ?)",
                        (self.identity, prompt_hash, int(value)),
                    )
                    values_by_hash[prompt_hash] = value

        return [values_by_hash[prompt_hash] for prompt_hash in hashes]


__all__ = [
    "CachedUserProvidedClassifier",
    "ClassificationError",
    "TeacherUserProvidedClassifier",
    "UserProvidedClassifier",
    "classification_prompt_hash",
]
