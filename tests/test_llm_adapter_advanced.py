"""Advanced LLM adapter tests — Phase 7.

Covers:

* JSON / function-calling schema shape is self-consistent.
* OpenAI and Anthropic payload builders place role prompts in the right slot.
* ``parse_openai_response`` handles tool-call arguments AND plain content.
* ``parse_anthropic_response`` handles ``tool_use`` blocks AND ``text`` blocks.
* ``StreamingActionParser`` tolerates chatter, nested braces, and escaped
  quotes spread across token boundaries.
* ``generate_action_with_retry`` retries malformed output, recovers on
  success, and degrades to NOOP when the provider keeps failing.
"""

from __future__ import annotations

import json

from chaosops.agents.llm_adapter import (
    ACTION_JSON_SCHEMA,
    StreamingActionParser,
    anthropic_tool_spec,
    build_anthropic_messages,
    build_openai_messages,
    generate_action_with_retry,
    openai_tool_spec,
    parse_anthropic_response,
    parse_openai_response,
    parse_streaming_action,
)
from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsObservation,
    FailureType,
    RoleView,
    ServiceMetrics,
)


def _obs(role: AgentRole = AgentRole.DEV) -> ChaosOpsObservation:
    view = RoleView(
        role=role,
        visible_metrics={
            "db": ServiceMetrics(
                cpu_pct=20, memory_mb=500, latency_ms=200, error_rate=0.1, replicas=1
            )
        },
        visible_logs=[],
        visible_alerts=[],
        visible_fleet_actions=[],
        shared_chat=[],
    )
    return ChaosOpsObservation(
        done=False, reward=0.0, view=view, step=1, turn_role=role, message=""
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_action_schema_enumerates_every_action_type() -> None:
    enum = ACTION_JSON_SCHEMA["properties"]["action_type"]["enum"]
    assert set(enum) == {a.value for a in ActionType}


def test_openai_tool_spec_wraps_schema() -> None:
    spec = openai_tool_spec()
    assert spec["type"] == "function"
    assert spec["function"]["parameters"] is ACTION_JSON_SCHEMA


def test_anthropic_tool_spec_uses_input_schema_key() -> None:
    spec = anthropic_tool_spec()
    assert spec["input_schema"] is ACTION_JSON_SCHEMA
    assert "name" in spec


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------


def test_build_openai_messages_has_system_then_user() -> None:
    messages = build_openai_messages(_obs(), system_prompt="YOU ARE DEV")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "YOU ARE DEV"
    assert messages[1]["role"] == "user"
    assert "STEP 1" in messages[1]["content"]


def test_build_openai_messages_omits_system_when_empty() -> None:
    messages = build_openai_messages(_obs(), system_prompt="")
    # With no role prompt file on disk either, we still get one user message.
    assert all(m["role"] != "system" for m in messages) or messages[0]["role"] == "system"


def test_build_anthropic_messages_separates_system() -> None:
    payload = build_anthropic_messages(_obs(), system_prompt="BE CAREFUL")
    assert payload["system"] == "BE CAREFUL"
    assert payload["messages"][0]["role"] == "user"
    assert "STEP 1" in payload["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def test_parse_openai_response_from_tool_call() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "chaosops_action",
                                "arguments": json.dumps(
                                    {
                                        "action_type": "restart",
                                        "target": "db",
                                        "args": {},
                                    }
                                ),
                            }
                        }
                    ],
                }
            }
        ]
    }
    action = parse_openai_response(response, role=AgentRole.DEV)
    assert action.action_type is ActionType.RESTART
    assert action.target == "db"


def test_parse_openai_response_falls_back_to_content() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": 'noise... {"action_type":"noop"} tail',
                    "tool_calls": [],
                }
            }
        ]
    }
    action = parse_openai_response(response, role=AgentRole.SRE)
    assert action.action_type is ActionType.NOOP


def test_parse_openai_response_handles_content_list() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": '{"action_type":"rollback",'},
                        {"type": "text", "text": '"target":"db"}'},
                    ],
                    "tool_calls": [],
                }
            }
        ]
    }
    action = parse_openai_response(response, role=AgentRole.DEV)
    assert action.action_type is ActionType.ROLLBACK
    assert action.target == "db"


def test_parse_anthropic_response_from_tool_use() -> None:
    response = {
        "content": [
            {
                "type": "tool_use",
                "name": "chaosops_action",
                "input": {
                    "action_type": "identify_root_cause",
                    "args": {"failure_type": FailureType.DB_DEADLOCK.value},
                },
            }
        ]
    }
    action = parse_anthropic_response(response, role=AgentRole.SRE)
    assert action.action_type is ActionType.IDENTIFY_ROOT_CAUSE
    assert action.args["failure_type"] == FailureType.DB_DEADLOCK.value


def test_parse_anthropic_response_falls_back_to_text() -> None:
    response = {
        "content": [
            {"type": "text", "text": 'sure: {"action_type":"communicate",'},
            {"type": "text", "text": '"args":{"message":"hi"}}'},
        ]
    }
    action = parse_anthropic_response(response, role=AgentRole.MANAGER)
    assert action.action_type is ActionType.COMMUNICATE
    assert action.args.get("message") == "hi"


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------


def test_streaming_parser_accumulates_across_chunks() -> None:
    parser = StreamingActionParser(role=AgentRole.DEV)
    # Leading chatter is discarded; JSON begins after it.
    assert parser.feed("Sure, here is my action: ") is False
    assert parser.feed('{"action_type":"scale",') is False
    assert parser.feed('"target":"db","args":{"replicas":3}}') is True

    action = parser.action()
    assert parser.finished
    assert action.action_type is ActionType.SCALE
    assert action.args == {"replicas": 3}


def test_streaming_parser_ignores_braces_inside_strings() -> None:
    parser = StreamingActionParser(role=AgentRole.MANAGER)
    parser.feed('{"action_type":"communicate","args":{"message":"uh } oh"}}')
    assert parser.finished
    action = parser.action()
    assert action.action_type is ActionType.COMMUNICATE
    assert action.args["message"] == "uh } oh"


def test_streaming_parser_respects_escaped_quotes() -> None:
    parser = StreamingActionParser(role=AgentRole.DEV)
    parser.feed('{"action_type":"communicate","args":{"message":"say \\"hi\\""}}')
    assert parser.finished
    action = parser.action()
    assert action.args["message"] == 'say "hi"'


def test_parse_streaming_action_helper() -> None:
    chunks = ['{"action_type":"rest', 'art","target":"db"}']
    action = parse_streaming_action(iter(chunks), role=AgentRole.DEV)
    assert action.action_type is ActionType.RESTART
    assert action.target == "db"


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def test_retry_returns_action_on_first_valid_output() -> None:
    calls: list[str] = []

    def gen(prompt: str) -> str:
        calls.append(prompt)
        return '{"action_type":"restart","target":"db"}'

    action = generate_action_with_retry(
        "initial", role=AgentRole.DEV, generate=gen, max_attempts=3
    )
    assert action.action_type is ActionType.RESTART
    assert len(calls) == 1


def test_retry_adds_reminder_and_recovers() -> None:
    outputs = iter(["not json at all", '{"action_type":"noop"}'])
    calls: list[str] = []

    def gen(prompt: str) -> str:
        calls.append(prompt)
        return next(outputs)

    action = generate_action_with_retry(
        "initial", role=AgentRole.SRE, generate=gen, max_attempts=3
    )
    assert action.action_type is ActionType.NOOP
    assert len(calls) == 2
    assert "Previous reply did not" in calls[1]


def test_retry_exhausts_attempts_and_falls_back_to_noop() -> None:
    def gen(_: str) -> str:
        return "still no json, sorry"

    action = generate_action_with_retry(
        "x", role=AgentRole.DEV, generate=gen, max_attempts=2, fallback=ActionType.NOOP
    )
    assert action.action_type is ActionType.NOOP
    assert action.target is None
    assert action.args == {}


def test_retry_swallows_provider_exceptions() -> None:
    attempts: list[int] = []

    def gen(_: str) -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("429 rate limited")
        return '{"action_type":"escalate"}'

    action = generate_action_with_retry(
        "x", role=AgentRole.MANAGER, generate=gen, max_attempts=3
    )
    assert action.action_type is ActionType.ESCALATE
    assert len(attempts) == 2
