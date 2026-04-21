"""Prompt & parser layer between LLM outputs and ``ChaosOpsAction``.

This module is the ONLY place where observation-to-string and
string-to-action conversion happens. Keeping it isolated means:

* The same adapter works for Unsloth + TRL GRPO training and for inference
  via the OpenEnv HTTP client.
* Prompt iteration (add/remove fields, tweak phrasing) never touches the
  simulator, reward function, or environment code.
* Tests can exercise the parser against fixed action strings without ever
  loading a model.

Public API, in layers:

* Rendering: ``render_observation``, ``build_prompt``
* Parsing: ``parse_action`` (single string), ``StreamingActionParser`` (token stream)
* Structured-output schemas: ``ACTION_JSON_SCHEMA``, ``openai_tool_spec``,
  ``anthropic_tool_spec``
* Provider builders: ``build_openai_messages``, ``build_anthropic_messages``
* Robustness: ``generate_action_with_retry`` for retry + fallback parsing
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
    FailureType,
    RoleView,
)


_LOG = logging.getLogger(__name__)


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

    if view.private_inbox:
        lines.append("PRIVATE_INBOX")
        for msg in view.private_inbox[-4:]:
            lines.append(f"  {msg}")

    if view.private_note:
        lines.append(f"NOTE {view.private_note}")

    lines.append(
        "RESPOND with JSON only: "
        '{"action_type": "<type>", "target": "<service|agent>", "args": {...}}'
    )
    lines.append(
        "For private messages use args.to=\"sre|dev|manager|oversight\"; "
        "default communicate broadcasts to all roles."
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
    # Brace-matched candidates (handles nested ``args`` objects).
    yield from _iter_balanced_braces(raw)
    # Last-ditch: bare non-nested braces (kept for backward compat with
    # older test fixtures that relied on this path).
    for match in _JSON_BLOCK.finditer(raw):
        yield match.group(0)


def _iter_balanced_braces(raw: str):
    """Yield every top-level balanced ``{...}`` block, string-aware."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield raw[start : i + 1]
                    start = -1


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


# ---------------------------------------------------------------------------
# Function-calling / structured-output schema
# ---------------------------------------------------------------------------


#: JSON Schema describing a ``ChaosOpsAction`` payload. Shared between OpenAI
#: function-calling, Anthropic tool use, and local validators so the action
#: vocabulary has a single source of truth.
ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action_type"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [a.value for a in ActionType],
            "description": "Which action to take this turn.",
        },
        "target": {
            "type": ["string", "null"],
            "description": (
                "Target service or fleet-agent name (e.g., 'db', "
                "'autoscaler'). Omit or null for NOOP / COMMUNICATE."
            ),
        },
        "args": {
            "type": "object",
            "description": (
                "Action-specific parameters. "
                "communicate: {to?: role, message: str}. "
                "scale: {replicas: int}. "
                "identify_root_cause: {failure_type: FailureType}."
            ),
            "additionalProperties": True,
            "properties": {
                "to": {
                    "type": "string",
                    "enum": [r.value for r in AgentRole],
                    "description": "Private recipient role for communicate.",
                },
                "message": {"type": "string"},
                "replicas": {"type": "integer", "minimum": 0},
                "failure_type": {
                    "type": "string",
                    "enum": [f.value for f in FailureType],
                },
            },
        },
    },
}


def openai_tool_spec(name: str = "chaosops_action") -> dict[str, Any]:
    """Return an OpenAI ``tools[]`` entry for structured action output."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Emit exactly one ChaosOps incident-response action for the "
                "current turn."
            ),
            "parameters": ACTION_JSON_SCHEMA,
        },
    }


def anthropic_tool_spec(name: str = "chaosops_action") -> dict[str, Any]:
    """Return an Anthropic ``tools[]`` entry for structured action output."""
    return {
        "name": name,
        "description": (
            "Emit exactly one ChaosOps incident-response action for the "
            "current turn."
        ),
        "input_schema": ACTION_JSON_SCHEMA,
    }


# ---------------------------------------------------------------------------
# Provider-specific prompt builders
# ---------------------------------------------------------------------------


def build_openai_messages(
    obs: ChaosOpsObservation, *, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """Build an OpenAI chat-completions ``messages`` array.

    The system message carries the role-specific prompt; the user message
    carries the rendered observation. Pair with :func:`openai_tool_spec` and
    ``tool_choice={"type": "function", "function": {"name": ...}}`` to force
    structured output.
    """
    system = system_prompt or load_role_prompt(obs.turn_role)
    body = render_observation(obs.view, step=obs.step)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": body})
    return messages


def build_anthropic_messages(
    obs: ChaosOpsObservation, *, system_prompt: str | None = None
) -> dict[str, Any]:
    """Build an Anthropic ``messages.create`` payload fragment.

    Anthropic separates ``system`` from ``messages``; returning both in one
    dict keeps call-sites ergonomic: ``client.messages.create(**payload, ...)``.
    """
    system = system_prompt or load_role_prompt(obs.turn_role)
    body = render_observation(obs.view, step=obs.step)
    return {
        "system": system or "",
        "messages": [{"role": "user", "content": body}],
    }


def parse_openai_response(
    response: Any, *, role: AgentRole, fallback: ActionType = ActionType.NOOP
) -> ChaosOpsAction:
    """Extract an action from an OpenAI chat-completions response.

    Handles both paths: tool/function call arguments (preferred) and plain
    ``content`` text. Accepts either the SDK object or a plain dict.
    """
    message = _openai_first_message(response)
    if message is None:
        return ChaosOpsAction(role=role, action_type=fallback)

    tool_calls = _get(message, "tool_calls") or []
    for call in tool_calls:
        fn = _get(call, "function") or {}
        args_raw = _get(fn, "arguments")
        if args_raw:
            action = _action_from_json_str(args_raw, role=role, fallback=fallback)
            if action.action_type is not fallback or args_raw.strip():
                return action

    content = _get(message, "content") or ""
    if isinstance(content, list):
        content = "".join(
            str(_get(c, "text") or "") for c in content if _get(c, "type") == "text"
        )
    return parse_action(str(content), role=role, fallback=fallback)


def parse_anthropic_response(
    response: Any, *, role: AgentRole, fallback: ActionType = ActionType.NOOP
) -> ChaosOpsAction:
    """Extract an action from an Anthropic ``messages.create`` response."""
    content = _get(response, "content") or []
    text_chunks: list[str] = []
    for block in content:
        block_type = _get(block, "type")
        if block_type == "tool_use":
            payload = _get(block, "input") or {}
            if isinstance(payload, Mapping):
                return _action_from_mapping(payload, role=role, fallback=fallback)
        elif block_type == "text":
            text_chunks.append(str(_get(block, "text") or ""))
    return parse_action("\n".join(text_chunks), role=role, fallback=fallback)


def _openai_first_message(response: Any) -> Any:
    choices = _get(response, "choices") or []
    if not choices:
        return None
    first = choices[0]
    return _get(first, "message") or first


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _action_from_json_str(
    raw: str, *, role: AgentRole, fallback: ActionType
) -> ChaosOpsAction:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return parse_action(raw, role=role, fallback=fallback)
    if not isinstance(payload, Mapping):
        return ChaosOpsAction(role=role, action_type=fallback)
    return _action_from_mapping(payload, role=role, fallback=fallback)


def _action_from_mapping(
    payload: Mapping[str, Any],
    *,
    role: AgentRole,
    fallback: ActionType,
) -> ChaosOpsAction:
    return parse_action(json.dumps(dict(payload)), role=role, fallback=fallback)


# ---------------------------------------------------------------------------
# Streaming parser — consumes tokens, yields an action once JSON closes
# ---------------------------------------------------------------------------


@dataclass
class StreamingActionParser:
    """Incremental parser for token streams.

    Feed tokens one-by-one via :meth:`feed`; once a balanced top-level JSON
    object closes, ``finished`` flips to True and :meth:`action` returns the
    parsed :class:`ChaosOpsAction`. Safe to call :meth:`action` before the
    stream finishes — it falls back through :func:`parse_action`.

    Handles:
      * leading chatter before the first ``{``
      * nested braces (``args`` is an object)
      * strings containing braces (``"}"`` inside quoted message bodies)
      * escaped characters inside strings (``\\"``)
    """

    role: AgentRole
    fallback: ActionType = ActionType.NOOP
    _buf: list[str] = field(default_factory=list)
    _depth: int = 0
    _started: bool = False
    _in_string: bool = False
    _escape: bool = False
    _closed: bool = False

    def feed(self, chunk: str) -> bool:
        """Append ``chunk`` to the stream. Returns True once JSON closes."""
        if self._closed or not chunk:
            return self._closed
        for ch in chunk:
            if not self._started:
                if ch == "{":
                    self._started = True
                    self._depth = 1
                    self._buf.append(ch)
                continue
            self._buf.append(ch)
            if self._escape:
                self._escape = False
                continue
            if ch == "\\" and self._in_string:
                self._escape = True
                continue
            if ch == '"':
                self._in_string = not self._in_string
                continue
            if self._in_string:
                continue
            if ch == "{":
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    self._closed = True
                    return True
        return False

    @property
    def finished(self) -> bool:
        return self._closed

    @property
    def raw(self) -> str:
        return "".join(self._buf)

    def action(self) -> ChaosOpsAction:
        """Return the parsed action, falling back to NOOP on failure."""
        return parse_action(self.raw, role=self.role, fallback=self.fallback)


def parse_streaming_action(
    chunks: Iterable[str],
    *,
    role: AgentRole,
    fallback: ActionType = ActionType.NOOP,
) -> ChaosOpsAction:
    """Consume a token iterator and return the parsed action."""
    parser = StreamingActionParser(role=role, fallback=fallback)
    for chunk in chunks:
        if parser.feed(chunk):
            break
    return parser.action()


# ---------------------------------------------------------------------------
# Retry + fallback wrapper
# ---------------------------------------------------------------------------


GenerateFn = Callable[[str], str]
"""A caller-supplied function: ``prompt -> raw model output``.

Kept provider-agnostic so the same retry wrapper can front OpenAI,
Anthropic, a local vLLM endpoint, or a deterministic stub in tests.
"""


def generate_action_with_retry(
    prompt: str,
    *,
    role: AgentRole,
    generate: GenerateFn,
    max_attempts: int = 3,
    fallback: ActionType = ActionType.NOOP,
    reminder: str = (
        "Previous reply did not contain a valid JSON action. "
        "Respond with ONLY a JSON object matching the schema."
    ),
) -> ChaosOpsAction:
    """Call ``generate`` up to ``max_attempts`` times; return a valid action.

    On each failed attempt (parser returns the raw fallback action with no
    target/args) we append a reminder to the prompt and retry. If every
    attempt fails — including provider exceptions — we return a NOOP rather
    than crash the episode; downstream reward logic will charge the penalty.
    """
    if max_attempts < 1:
        max_attempts = 1

    current_prompt = prompt
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        try:
            last_raw = generate(current_prompt) or ""
        except Exception as exc:  # noqa: BLE001 — provider errors vary widely
            _LOG.warning("LLM generation attempt %d failed: %s", attempt, exc)
            current_prompt = f"{prompt}\n\n{reminder}"
            continue

        action = parse_action(last_raw, role=role, fallback=fallback)
        if _is_usable_action(action, last_raw):
            return action
        current_prompt = f"{prompt}\n\n{reminder}\n\nPrevious output:\n{last_raw[:500]}"

    _LOG.info(
        "LLM retry budget exhausted for role=%s; falling back to %s",
        role.value,
        fallback.value,
    )
    return ChaosOpsAction(role=role, action_type=fallback)


def _is_usable_action(action: ChaosOpsAction, raw: str) -> bool:
    """Heuristic: did the parser actually find structured content?

    A bare fallback (``NOOP`` with no target, no args, and no JSON in the
    raw output) means the model produced nothing parseable — retry-worthy.
    A deliberate NOOP with an explicit JSON block is fine.
    """
    if action.target or action.args:
        return True
    if action.action_type is not ActionType.NOOP:
        return True
    return "{" in raw and "}" in raw
