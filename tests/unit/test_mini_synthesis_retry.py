from src.mini.synthesize import (
    TrajectoryValidationError,
    deterministic_replacement_seeds,
    _failure_record_retryable,
    _is_transient,
    validate_trajectory_payload,
)


def test_replacement_seeds_are_stable_unique_and_appendable() -> None:
    existing = [11, 22, 33]

    first = deterministic_replacement_seeds(42, existing, 2)
    assert first == deterministic_replacement_seeds(42, existing, 2)
    assert len(first) == len(set(first)) == 2
    assert set(first).isdisjoint(existing)

    following = deterministic_replacement_seeds(42, [*existing, *first], 1)
    assert following[0] not in {*existing, *first}


def test_no_accepted_node_is_retryable_during_validation() -> None:
    error = TrajectoryValidationError("trajectory has no accepted node")

    assert _is_transient(error, "validation") is True
    assert _failure_record_retryable(
        {
            "stage": "validation",
            "exception_class": "TrajectoryValidationError",
            "safe_message": "trajectory has no accepted node",
            "retryable": False,
        }
    ) is True


def test_other_validation_failures_remain_non_retryable() -> None:
    error = TrajectoryValidationError("tool call is outside mini catalog")

    assert _is_transient(error, "validation") is False


def test_resume_validation_ignores_incomplete_rejected_nodes() -> None:
    payload = {
        "seed": 7,
        "scenario": {},
        "user_tools": [],
        "user_profile": {},
        "nodes": [
            {
                "decision": True,
                "mcp_servers": ["Tiny"],
                "initial_scenario": {"Tiny": {}},
                "final_scenario": {"Tiny": {}},
                "steps": [
                    {"role": "user", "content": "Ping the service."},
                    {
                        "role": "tool_call",
                        "content": [{"name": "Tiny-ping", "arguments": {}}],
                    },
                    {"role": "tool_response", "content": [{"ok": True}]},
                    {"role": "assistant", "content": "Done."},
                ],
            },
            {
                "decision": False,
                "mcp_servers": ["Tiny"],
                "initial_scenario": {"Tiny": {}},
                "final_scenario": None,
                "steps": [],
            },
        ],
    }

    validate_trajectory_payload(payload, {"Tiny-ping"})
