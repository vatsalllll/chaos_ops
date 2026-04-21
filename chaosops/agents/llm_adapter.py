"""Prompt & parser layer between LLM outputs and ``ChaosOpsAction``.

This module is the ONLY place where observation-to-string and
string-to-action conversion happens. Keeping it isolated means:

* The same adapter works for Unsloth + TRL GRPO training and for inference
  via the OpenEnv HTTP client.
* Prompt iteration (add/remove fields, tweak phrasing) never touches the
  simulator, reward function, or environment code.
* Tests can exercise the parser against fixed action strings without ever
  loading a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
    FailureType,
    RoleView,
)


PROMPT_DIR = Path(__file__).parent / "prompts"

ROLE_PROMPT_FILES: dict[AgentRole, str] = {
    AgentRole.SRE: "sre.md",
    AgentRole.DEV: "dev.md",
    AgentRole.MANAGER: "manager.md",
    AgentRole.OVERSIGHT: "oversight.md",
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def load_role_prompt(role: AgentRole) -> str:
    path = PROMPT_DIR / ROLE_PROMPT_FILES[role]
    if not path.exists():
        return ""
    return path.read_text().strip()


def render_observation(view: RoleView, *, step: int) -> str:
    """Render a ``RoleView`` as a compact text prompt body.

    Compact because each role turn eats into the 8B model's context budget
    five times per episode; we want the entire trajectory to fit in <1,200
    tokens. Numbers are rounded and logs are bulleted.
    """
    lines: list[str] = [f"STEP {step}", f"ROLE {view.role.value.upper()}"]

    if view.visible_metrics:
        lines.append("METRICS")
        for name, m in view.visible_metrics.items():
            lines.append(
                f"  {name}: cpu={m.cpu_pct:.0f}% mem={m.memory_mb:.0f}MB "
                f"lat={m.latency_ms:.0f}ms err={m.error_rate:.0%} "
                f"repl={m.replicas} health={m.health.value}"
            )

    if view.visible_alerts:
        lines.append("ALERTS")
        for a in view.visible_alerts[-4:]:
            lines.append(f"  [{a.severity}] {a.service.value}: {a.message}")

    if view.visible_logs:
        lines.append("LOGS")
        for log in view.visible_logs[-4:]:
            lines.append(f"  ({log.level}) {log.service.value}: {log.message}")

    if view.visible_fleet_actions:
        lines.append("FLEET_ACTIONS")
        for fa in view.visible_fleet_actions[-6:]:
            lines.append(
                f"  step {fa.step}: {fa.agent_name} {fa.action} -> {fa.target}"
            )

    if view.shared_chat:
        lines.append("CHAT")
        for msg in view.shared_chat[-6:]:
            lines.append(f"  {msg}")

    if view.private_note:
        lines.append(f"NOTE {view.private_note}")

    lines.append(
        "RESPOND with JSON only: "
        '{"action_type": "<type>", "target": "<service|agent>", "args": {...}}'
    )
    return "\n".join(lines)


def build_prompt(
    obs: ChaosOpsObservation, *, system_prompt: str | None = None
) -> str:
    role_prompt = system_prompt or load_role_prompt(obs.turn_role)
    body = render_observation(obs.view, step=obs.step)
    return f"{role_prompt}\n\n{body}" if role_prompt else body


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_JSON_BLOCK = re.compile(r"\{[^{}]*\}")

_VALID_ACTION_STRINGS: dict[str, ActionType] = {a.value: a for a in ActionType}


def parse_action(
    raw: str, *, role: AgentRole, fallback: ActionType = ActionType.NOOP
) -> ChaosOpsAction:
    """Extract a ``ChaosOpsAction`` from an LLM output string.

    The parser tolerates:
      * leading/trailing chatter around a JSON block
      * unknown action types (falls back to ``NOOP``)
      * missing ``target`` or ``args``
      * integer-valued replicas encoded as strings

    What it won't do: silently turn a malformed output into a confident
    action. If nothing parses, it returns a NOOP (and the reward function
    logs a miscommunication penalty upstream).
    """
    payload = _extract_json(raw)
    if payload is None:
        return ChaosOpsAction(role=role, action_type=fallback)

    action_type_raw = str(payload.get("action_type", "")).strip().lower()
    action_type = _VALID_ACTION_STRINGS.get(action_type_raw, fallback)

    target = payload.get("target")
    if target is not None:
        target = str(target)

    args = payload.get("args")
    if not isinstance(args, Mapping):
        args = {}
    coerced_args: dict[str, object] = {}
    for key, value in args.items():
        if key == "replicas":
            try:
                coerced_args["replicas"] = int(value)
            except (TypeError, ValueError):
                continue
        elif key == "failure_type":
            try:
                coerced_args["failure_type"] = FailureType(str(value)).value
            except ValueError:
                continue
        else:
            coerced_args[str(key)] = value

    return ChaosOpsAction(
        role=role, action_type=action_type, target=target, args=coerced_args
    )


def _extract_json(raw: str) -> dict[str, object] | None:
    raw = raw.strip()
    # Fast path — entire output is JSON.
    for candidate in _iter_json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _iter_json_candidates(raw: str):
    if raw.startswith("{") and raw.endswith("}"):
        yield raw
    # Code fences.
    for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw):
        yield match.group(1)
    # Bare braces (non-nested is enough for our action schema).
    for match in _JSON_BLOCK.finditer(raw):
        yield match.group(0)


def action_to_training_target(action: ChaosOpsAction) -> str:
    """Canonical serialization of an action — used as the supervised target
    when running SFT warm-up before GRPO, and as the greedy decoding target
    for the oracle-trajectory teacher."""
    return json.dumps(
        {
            "action_type": action.action_type.value,
            "target": action.target,
            "args": action.args,
        },
        sort_keys=True,
    )
