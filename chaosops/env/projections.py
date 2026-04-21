"""Role-aware observation projection.

Each :class:`AgentRole` sees a *subset* of the ground-truth state — this
is what makes ChaosOps AI partially observable and gives the Oversight
agent meaningful information asymmetry.

Pulled out of ``world_sim.py`` so per-role visibility rules live in one
place. :func:`project_view` is the single entry point; role-specific
helpers are kept private.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chaosops.env.models import (
    AgentRole,
    FleetAgentLog,
    LogLine,
    RoleView,
)

if TYPE_CHECKING:  # pragma: no cover
    from chaosops.env.world_sim import WorldSim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_logs(sim: "WorldSim") -> list[LogLine]:
    """Strip the ``is_red_herring`` flag before sending logs to agents."""
    tail = list(sim.state.all_logs)[-sim.MAX_LOGS_PER_VIEW :]
    return [
        LogLine(
            timestamp=line.timestamp,
            service=line.service,
            level=line.level,
            message=line.message,
            is_red_herring=False,
        )
        for line in tail
    ]


def _public_fleet_actions(sim: "WorldSim") -> list[FleetAgentLog]:
    """Strip the ground-truth ``was_malicious`` flag from fleet logs."""
    return [
        FleetAgentLog(
            agent_name=fa.agent_name,
            action=fa.action,
            target=fa.target,
            step=fa.step,
            was_malicious=False,
        )
        for fa in sim.state.fleet_actions
    ]


def _chat_tail(sim: "WorldSim") -> list[str]:
    return sim.state.chat_history[-sim.MAX_MESSAGES_PER_VIEW :]


def _private_inbox(sim: "WorldSim", role: AgentRole) -> list[str]:
    """Messages addressed *to* ``role`` via a private channel."""
    return list(sim.state.private_chat.get(role.value, []))[
        -sim.MAX_MESSAGES_PER_VIEW :
    ]


# ---------------------------------------------------------------------------
# Role-specific projections
# ---------------------------------------------------------------------------


def _sre_view(sim: "WorldSim") -> RoleView:
    return RoleView(
        role=AgentRole.SRE,
        visible_metrics=sim.state.services,
        visible_logs=_public_logs(sim),
        visible_alerts=list(sim.state.all_alerts),
        visible_fleet_actions=[],
        shared_chat=_chat_tail(sim),
        private_inbox=_private_inbox(sim, AgentRole.SRE),
    )


def _dev_view(sim: "WorldSim") -> RoleView:
    return RoleView(
        role=AgentRole.DEV,
        visible_metrics=sim.state.services,
        visible_logs=_public_logs(sim)[-3:],
        visible_alerts=list(sim.state.all_alerts),
        visible_fleet_actions=[],
        shared_chat=_chat_tail(sim),
        private_inbox=_private_inbox(sim, AgentRole.DEV),
    )


def _manager_view(sim: "WorldSim") -> RoleView:
    return RoleView(
        role=AgentRole.MANAGER,
        visible_metrics={},
        visible_logs=[],
        visible_alerts=list(sim.state.all_alerts),
        visible_fleet_actions=[],
        shared_chat=_chat_tail(sim),
        private_inbox=_private_inbox(sim, AgentRole.MANAGER),
    )


def _oversight_view(sim: "WorldSim") -> RoleView:
    fleet_actions = _public_fleet_actions(sim)
    private_note = (
        "fleet actions visible; cross-reference with alerts."
        if fleet_actions
        else None
    )
    return RoleView(
        role=AgentRole.OVERSIGHT,
        visible_metrics=sim.state.services,
        visible_logs=_public_logs(sim),
        visible_alerts=list(sim.state.all_alerts),
        visible_fleet_actions=fleet_actions,
        shared_chat=_chat_tail(sim),
        private_note=private_note,
        private_inbox=_private_inbox(sim, AgentRole.OVERSIGHT),
    )


_PROJECTORS = {
    AgentRole.SRE: _sre_view,
    AgentRole.DEV: _dev_view,
    AgentRole.MANAGER: _manager_view,
    AgentRole.OVERSIGHT: _oversight_view,
}


def project_view(sim: "WorldSim", role: AgentRole) -> RoleView:
    """Return the observation that ``role`` is allowed to see."""
    return _PROJECTORS[role](sim)


__all__ = ["project_view"]
