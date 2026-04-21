"""Per-action-type handlers for :class:`WorldSim`.

Each handler mutates ``sim.state`` in response to one agent action and
sets the outcome flags that the reward function consumes. Split out of
``world_sim.py`` so new action types can be added without scrolling past
600 lines of simulator infrastructure.

Public surface:

* :data:`ACTION_HANDLERS` — dispatch table keyed by :class:`ActionType`.
* :func:`handle_action` — convenience wrapper used by ``WorldSim``.

Every handler has the same signature ``(sim, action, flags) -> None``
and is expected to mutate ``flags`` in-place using the keys defined in
``WorldSim.apply_action``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from chaosops.env.injectors import is_beneficial_action
from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    FailureType,
    ServiceHealth,
    ServiceName,
)

if TYPE_CHECKING:  # pragma: no cover
    from chaosops.env.world_sim import WorldSim


Handler = Callable[["WorldSim", ChaosOpsAction, dict[str, bool]], None]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _act_noop(_sim: "WorldSim", _action: ChaosOpsAction, _flags: dict[str, bool]) -> None:
    return


def _act_communicate(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    body = str(action.args.get("message", "")).strip()
    if not body:
        flags["miscommunication"] = True
        return
    # Private channels: if ``to`` is a recipient role, store it in the
    # role-scoped private chat buffer instead of the broadcast history.
    recipient = str(action.args.get("to", "")).strip().lower()
    if recipient and recipient != "all":
        sim._emit_private_message(action.role, recipient, body)
        return
    sim.state.chat_history.append(f"[{action.role.value}] {body}")


def _act_restart(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    target = action.target
    if not target or target not in sim.state.services:
        flags["wrong_fix"] = True
        sim.state.wrong_fixes += 1
        return
    svc = sim.state.services[target]
    if is_beneficial_action(action, sim.state):
        svc.health = ServiceHealth.HEALTHY
        svc.memory_mb = 340.0
        svc.cpu_pct = 22.0
        svc.latency_ms = 45.0
        svc.error_rate = 0.01
        # DB restart also clears downstream knock-on effects.
        if target == ServiceName.DB.value:
            for ds in (ServiceName.PAYMENTS.value, ServiceName.AUTH.value):
                sim.state.services[ds].latency_ms = 55.0
                sim.state.services[ds].error_rate = 0.01
                sim.state.services[ds].health = ServiceHealth.HEALTHY
    else:
        flags["wrong_fix"] = True
        sim.state.wrong_fixes += 1
        _maybe_trigger_cascade(sim, flags)


def _act_rollback(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    target = action.target
    if not target or target not in sim.state.services:
        flags["wrong_fix"] = True
        sim.state.wrong_fixes += 1
        return
    svc = sim.state.services[target]
    if is_beneficial_action(action, sim.state):
        svc.health = ServiceHealth.HEALTHY
        svc.error_rate = 0.01
        svc.latency_ms = 45.0
    else:
        flags["wrong_fix"] = True
        sim.state.wrong_fixes += 1
        _maybe_trigger_cascade(sim, flags)


def _act_scale(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    target = action.target
    replicas = int(action.args.get("replicas", 0))
    if not target or target not in sim.state.services or replicas <= 0:
        flags["wrong_fix"] = True
        sim.state.wrong_fixes += 1
        return
    svc = sim.state.services[target]
    svc.replicas = replicas
    if is_beneficial_action(action, sim.state):
        svc.cpu_pct = 30.0
        svc.latency_ms = 55.0
        svc.error_rate = 0.01
        svc.health = ServiceHealth.HEALTHY
    else:
        flags["wrong_fix"] = True
        sim.state.wrong_fixes += 1


def _act_flag_rogue(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    if action.role != AgentRole.OVERSIGHT:
        # Only Oversight may legally flag — from any other role this counts
        # as a miscommunication, not a rogue flag.
        flags["miscommunication"] = True
        sim.state.miscommunications += 1
        return
    flagged = action.target or str(action.args.get("agent_name", "")).strip()
    if not flagged:
        return
    sim.state.oversight_flags.append(flagged)
    truth = sim.state.rogue_fleet_agent
    if truth is not None and flagged == truth:
        flags["rogue_flagged_correctly"] = True
    else:
        flags["rogue_flagged_incorrectly"] = True


def _act_identify_root_cause(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    claimed = str(action.args.get("failure_type", "")).strip()
    try:
        claimed_ft = FailureType(claimed)
    except ValueError:
        flags["miscommunication"] = True
        return
    sim.state.declared_root_cause = claimed_ft
    if sim.state.declared_root_cause_step is None:
        sim.state.declared_root_cause_step = sim.state.step_count
    if claimed_ft == sim.state.failure_type:
        flags["root_cause_correct"] = True


def _act_escalate(
    _sim: "WorldSim", _action: ChaosOpsAction, _flags: dict[str, bool]
) -> None:
    # Useful primarily for misrouted-traffic resolution; silent otherwise.
    return


def _act_resolve(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    if action.role != AgentRole.MANAGER:
        flags["miscommunication"] = True
        sim.state.miscommunications += 1
        return
    # Actual resolution is decided by ``CORRECT_FIX`` in ``apply_action``.


# ---------------------------------------------------------------------------
# Cascade helper — shared by RESTART + ROLLBACK wrong-fix paths
# ---------------------------------------------------------------------------


def _maybe_trigger_cascade(sim: "WorldSim", flags: dict[str, bool]) -> None:
    if sim.state.failure_type != FailureType.CASCADE:
        return
    notif = sim.state.services[ServiceName.NOTIFICATIONS.value]
    if notif.health != ServiceHealth.HEALTHY:
        return
    notif.memory_mb = 900.0
    notif.health = ServiceHealth.DEGRADED
    flags["cascade_triggered"] = True
    sim._emit_log(
        ServiceName.NOTIFICATIONS,
        "WARN",
        "secondary failure: heap growing after failed remediation",
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


ACTION_HANDLERS: dict[ActionType, Handler] = {
    ActionType.NOOP: _act_noop,
    ActionType.COMMUNICATE: _act_communicate,
    ActionType.RESTART: _act_restart,
    ActionType.ROLLBACK: _act_rollback,
    ActionType.SCALE: _act_scale,
    ActionType.FLAG_ROGUE: _act_flag_rogue,
    ActionType.IDENTIFY_ROOT_CAUSE: _act_identify_root_cause,
    ActionType.ESCALATE: _act_escalate,
    ActionType.RESOLVE: _act_resolve,
}


def handle_action(
    sim: "WorldSim", action: ChaosOpsAction, flags: dict[str, bool]
) -> None:
    """Route ``action`` to its handler, mutating ``sim`` + ``flags`` in place."""
    handler = ACTION_HANDLERS.get(action.action_type, _act_noop)
    handler(sim, action, flags)


__all__ = ["ACTION_HANDLERS", "handle_action", "Handler"]
