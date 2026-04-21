"""Tests for ``chaosops.agents.llm_adapter``.

The adapter is the only place observation-to-string and string-to-action
conversion happens, so it gets a focused parser suite: well-formed JSON,
noisy chat-wrapped JSON, code-fenced output, malformed input, and
canonical serialization.
"""

from __future__ import annotations

import json

from chaosops.agents.llm_adapter import (
    action_to_training_target,
    build_prompt,
    parse_action,
    render_observation,
)
from chaosops.env.models import (
    ActionType,
    AgentRole,
    Alert,
    ChaosOpsObservation,
    FailureType,
    FleetAgentLog,
    LogLine,
    RoleView,
    ServiceHealth,
    ServiceMetrics,
    ServiceName,
)


# ---------------------------------------------------------------------------
# Parser — happy path + tolerance
# ---------------------------------------------------------------------------


def test_parse_strict_json_action() -> None:
    raw = '{"action_type": "restart", "target": "db", "args": {}}'
    action = parse_action(raw, role=AgentRole.DEV)
    assert action.action_type == ActionType.RESTART
    assert action.target == "db"
    assert action.role == AgentRole.DEV


def test_parse_chat_wrapped_json() -> None:
    raw = (
        "Looking at the logs, I should restart the DB.\n"
        'Action: {"action_type":"restart","target":"db"} — confident.'
    )
    action = parse_action(raw, role=AgentRole.DEV)
    assert action.action_type == ActionType.RESTART
    assert action.target == "db"


def test_parse_code_fenced_json() -> None:
    raw = "```json\n" '{"action_type": "scale", "target": "payments", "args": {"replicas": 4}}' "\n```"
    action = parse_action(raw, role=AgentRole.DEV)
    assert action.action_type == ActionType.SCALE
    assert action.args["replicas"] == 4  # coerced to int


def test_parse_coerces_string_replicas_to_int() -> None:
    raw = '{"action_type": "scale", "target": "payments", "args": {"replicas": "3"}}'
    action = parse_action(raw, role=AgentRole.DEV)
    assert action.args["replicas"] == 3
    assert isinstance(action.args["replicas"], int)


def test_parse_drops_invalid_replicas() -> None:
    raw = '{"action_type": "scale", "target": "payments", "args": {"replicas": "lots"}}'
    action = parse_action(raw, role=AgentRole.DEV)
    assert "replicas" not in action.args


def test_parse_validates_failure_type_arg() -> None:
    raw = '{"action_type": "identify_root_cause", "args": {"failure_type": "db_deadlock"}}'
    action = parse_action(raw, role=AgentRole.SRE)
    assert action.args["failure_type"] == FailureType.DB_DEADLOCK.value


def test_parse_drops_unknown_failure_type() -> None:
    raw = '{"action_type": "identify_root_cause", "args": {"failure_type": "solar_flare"}}'
    action = parse_action(raw, role=AgentRole.SRE)
    assert "failure_type" not in action.args


# ---------------------------------------------------------------------------
# Parser — failure modes
# ---------------------------------------------------------------------------


def test_parse_unknown_action_falls_back_to_noop() -> None:
    raw = '{"action_type": "launch_missiles"}'
    action = parse_action(raw, role=AgentRole.SRE)
    assert action.action_type == ActionType.NOOP


def test_parse_empty_input_returns_noop() -> None:
    action = parse_action("", role=AgentRole.SRE)
    assert action.action_type == ActionType.NOOP


def test_parse_broken_json_returns_noop() -> None:
    action = parse_action("I think we should {restart the db??", role=AgentRole.DEV)
    assert action.action_type == ActionType.NOOP


def test_parse_respects_custom_fallback() -> None:
    action = parse_action("nope", role=AgentRole.MANAGER, fallback=ActionType.COMMUNICATE)
    assert action.action_type == ActionType.COMMUNICATE


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


def test_action_to_training_target_round_trips() -> None:
    raw = '{"action_type": "restart", "target": "db", "args": {}}'
    action = parse_action(raw, role=AgentRole.DEV)
    serialized = action_to_training_target(action)
    decoded = json.loads(serialized)
    assert decoded["action_type"] == "restart"
    assert decoded["target"] == "db"


# ---------------------------------------------------------------------------
# Observation rendering
# ---------------------------------------------------------------------------


def _sre_view() -> RoleView:
    return RoleView(
        role=AgentRole.SRE,
        visible_metrics={
            ServiceName.DB.value: ServiceMetrics(
                cpu_pct=71.0,
                memory_mb=500.0,
                latency_ms=1800.0,
                error_rate=0.45,
                replicas=1,
                health=ServiceHealth.CRITICAL,
            )
        },
        visible_logs=[
            LogLine(
                timestamp=1.0,
                service=ServiceName.DB,
                level="ERROR",
                message="deadlock on txn",
            )
        ],
        visible_alerts=[
            Alert(
                service=ServiceName.DB,
                severity="page",
                message="payments error rate > 30%",
                triggered_at_step=1,
            )
        ],
        visible_fleet_actions=[],
        shared_chat=[],
    )


def test_render_observation_contains_core_sections() -> None:
    text = render_observation(_sre_view(), step=1)
    assert "ROLE SRE" in text
    assert "METRICS" in text
    assert "ALERTS" in text
    assert "LOGS" in text
    assert "JSON only" in text  # response format instruction present


def test_render_observation_omits_empty_sections() -> None:
    empty_view = RoleView(
        role=AgentRole.MANAGER,
        visible_metrics={},
        visible_logs=[],
        visible_alerts=[],
        visible_fleet_actions=[],
        shared_chat=[],
    )
    text = render_observation(empty_view, step=1)
    assert "METRICS" not in text
    assert "ALERTS" not in text


def test_oversight_private_note_is_rendered() -> None:
    view = RoleView(
        role=AgentRole.OVERSIGHT,
        visible_metrics={},
        visible_logs=[],
        visible_alerts=[],
        visible_fleet_actions=[
            FleetAgentLog(
                agent_name="autoscaler",
                action="scale_down",
                target="payments",
                step=1,
            )
        ],
        shared_chat=[],
        private_note="cross-reference fleet actions with alerts",
    )
    text = render_observation(view, step=1)
    assert "NOTE cross-reference" in text


def test_build_prompt_prepends_role_system_prompt() -> None:
    obs = ChaosOpsObservation(
        view=_sre_view(),
        step=1,
        turn_role=AgentRole.SRE,
    )
    prompt = build_prompt(obs, system_prompt="You are the SRE.")
    assert prompt.startswith("You are the SRE.")
    assert "ROLE SRE" in prompt
