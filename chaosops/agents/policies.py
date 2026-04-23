"""Scripted policies for baseline, unit testing, and curriculum self-play.

These are **not** LLM agents. They are rule-based actors that drive the
environment during:

* unit tests (``tests/test_world_sim.py``)
* reward-curve baselines ("before training" number in the demo)
* sanity checks before every training run

Each policy is a callable ``(observation, role) -> ChaosOpsAction``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Iterable

from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
    FailureType,
    RoleView,
    ServiceHealth,
    ServiceName,
)

Policy = Callable[[ChaosOpsObservation, AgentRole], ChaosOpsAction]


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


_RANDOM_TARGETS: tuple[str, ...] = (
    ServiceName.AUTH.value,
    ServiceName.PAYMENTS.value,
    ServiceName.NOTIFICATIONS.value,
    ServiceName.DB.value,
)


def random_policy(seed: int = 0) -> Policy:
    """Return a seeded random policy. Useful as a hard lower bound."""

    rng = random.Random(seed)
    legal_action_types: tuple[ActionType, ...] = (
        ActionType.NOOP,
        ActionType.COMMUNICATE,
        ActionType.RESTART,
        ActionType.ROLLBACK,
        ActionType.SCALE,
    )

    def _policy(obs: ChaosOpsObservation, role: AgentRole) -> ChaosOpsAction:
        action_type = rng.choice(legal_action_types)
        args: dict[str, object] = {}
        target: str | None = None
        if action_type in {ActionType.RESTART, ActionType.ROLLBACK, ActionType.SCALE}:
            target = rng.choice(_RANDOM_TARGETS)
        if action_type == ActionType.SCALE:
            args["replicas"] = rng.choice([1, 2, 3, 4, 5])
        if action_type == ActionType.COMMUNICATE:
            args["message"] = rng.choice(
                [
                    "checking metrics",
                    "someone should look at this",
                    "i'll try a rollback",
                    "standing by",
                ]
            )
        return ChaosOpsAction(role=role, action_type=action_type, target=target, args=args)

    return _policy


# ---------------------------------------------------------------------------
# Heuristic policy — respects partial observability
# ---------------------------------------------------------------------------


def heuristic_policy(seed: int = 0) -> Policy:
    """Role-aware heuristic that reads alerts/metrics and picks plausible fixes.

    This is the "what a decent human would try" baseline. Not oracle-perfect:
    it doesn't see the ground-truth failure type and can pick the wrong fix.
    """

    rng = random.Random(seed)

    def _policy(obs: ChaosOpsObservation, role: AgentRole) -> ChaosOpsAction:
        view = obs.view
        if role == AgentRole.MANAGER:
            return _manager_heuristic(view, role)
        if role == AgentRole.OVERSIGHT:
            return _oversight_heuristic(view, role)
        return _responder_heuristic(view, role, rng)

    return _policy


def _responder_heuristic(view: RoleView, role: AgentRole, rng: random.Random) -> ChaosOpsAction:
    # Pick the unhealthiest service and react to it.
    broken = [
        (name, m)
        for name, m in view.visible_metrics.items()
        if m.health in {ServiceHealth.DEGRADED, ServiceHealth.CRITICAL, ServiceHealth.DOWN}
    ]
    if not broken:
        return ChaosOpsAction(
            role=role,
            action_type=ActionType.COMMUNICATE,
            args={"message": "systems nominal from my view"},
        )
    broken.sort(key=lambda kv: _health_rank(kv[1].health), reverse=True)
    name, metrics = broken[0]
    # Heuristic: high error rate shortly after a deploy -> rollback; otherwise restart.
    deploy_mentioned = any("rolled out" in log.message.lower() for log in view.visible_logs)
    if deploy_mentioned and role == AgentRole.DEV:
        return ChaosOpsAction(role=role, action_type=ActionType.ROLLBACK, target=name)
    if metrics.replicas < 3 and metrics.cpu_pct > 80.0:
        return ChaosOpsAction(
            role=role, action_type=ActionType.SCALE, target=name, args={"replicas": 4}
        )
    # Inject a little noise so multiple scripted runs differ.
    if rng.random() < 0.15:
        return ChaosOpsAction(
            role=role,
            action_type=ActionType.COMMUNICATE,
            args={"message": f"investigating {name}"},
        )
    return ChaosOpsAction(role=role, action_type=ActionType.RESTART, target=name)


def _manager_heuristic(view: RoleView, role: AgentRole) -> ChaosOpsAction:
    if any("resolved" in msg.lower() or "healthy" in msg.lower() for msg in view.shared_chat):
        return ChaosOpsAction(role=role, action_type=ActionType.RESOLVE)
    if not view.visible_alerts:
        return ChaosOpsAction(role=role, action_type=ActionType.NOOP)
    return ChaosOpsAction(
        role=role,
        action_type=ActionType.COMMUNICATE,
        args={"message": "SRE: diagnose. Dev: stage a rollback. Oversight: watch the fleet."},
    )


def _oversight_heuristic(view: RoleView, role: AgentRole) -> ChaosOpsAction:
    for fleet_action in view.visible_fleet_actions:
        if fleet_action.agent_name in {"autoscaler", "load_balancer", "deploy_bot"}:
            return ChaosOpsAction(
                role=role,
                action_type=ActionType.FLAG_ROGUE,
                target=fleet_action.agent_name,
            )
    return ChaosOpsAction(
        role=role,
        action_type=ActionType.COMMUNICATE,
        args={"message": "no rogue agent activity detected"},
    )


_HEALTH_ORDER = {
    ServiceHealth.HEALTHY: 0,
    ServiceHealth.DEGRADED: 1,
    ServiceHealth.CRITICAL: 2,
    ServiceHealth.DOWN: 3,
}


def _health_rank(health: ServiceHealth) -> int:
    return _HEALTH_ORDER[health]


# ---------------------------------------------------------------------------
# Oracle policy — knows the ground truth; used for upper-bound curves
# ---------------------------------------------------------------------------


def oracle_policy(failure_type: FailureType) -> Policy:
    """Return an oracle that picks the canonical fix on turn 1.

    We *cheat* here intentionally: the oracle knows ``failure_type`` at
    construction time. Use this to set the ceiling of the reward curve on
    before/after plots so the trained model has a visible target.
    """

    plan = _ORACLE_PLANS[failure_type]

    def _policy(obs: ChaosOpsObservation, role: AgentRole) -> ChaosOpsAction:
        action_fn = plan.get(role)
        if action_fn is None:
            return ChaosOpsAction(role=role, action_type=ActionType.NOOP)
        return action_fn(role, obs)

    return _policy


def _restart(target: ServiceName):
    def _fn(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
        return ChaosOpsAction(role=role, action_type=ActionType.RESTART, target=target.value)

    return _fn


def _rollback(target: ServiceName):
    def _fn(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
        return ChaosOpsAction(role=role, action_type=ActionType.ROLLBACK, target=target.value)

    return _fn


def _scale(target: ServiceName, replicas: int):
    def _fn(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
        return ChaosOpsAction(
            role=role,
            action_type=ActionType.SCALE,
            target=target.value,
            args={"replicas": replicas},
        )

    return _fn


def _flag(agent_name: str):
    def _fn(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
        return ChaosOpsAction(
            role=role, action_type=ActionType.FLAG_ROGUE, target=agent_name
        )

    return _fn


def _escalate(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
    return ChaosOpsAction(role=role, action_type=ActionType.ESCALATE)


def _resolve(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
    return ChaosOpsAction(role=role, action_type=ActionType.RESOLVE)


def _identify(failure_type: FailureType):
    def _fn(role: AgentRole, _obs: ChaosOpsObservation) -> ChaosOpsAction:
        return ChaosOpsAction(
            role=role,
            action_type=ActionType.IDENTIFY_ROOT_CAUSE,
            args={"failure_type": failure_type.value},
        )

    return _fn


_ORACLE_PLANS: dict[FailureType, dict[AgentRole, Callable[[AgentRole, ChaosOpsObservation], ChaosOpsAction]]] = {
    FailureType.DB_DEADLOCK: {
        AgentRole.SRE: _identify(FailureType.DB_DEADLOCK),
        AgentRole.DEV: _restart(ServiceName.DB),
    },
    FailureType.MEMORY_LEAK: {
        AgentRole.SRE: _identify(FailureType.MEMORY_LEAK),
        AgentRole.DEV: _restart(ServiceName.NOTIFICATIONS),
    },
    FailureType.BAD_CONFIG_PUSH: {
        AgentRole.SRE: _identify(FailureType.BAD_CONFIG_PUSH),
        AgentRole.DEV: _rollback(ServiceName.AUTH),
    },
    FailureType.AUTOSCALER_COST_CUT: {
        AgentRole.OVERSIGHT: _flag("autoscaler"),
        AgentRole.SRE: _identify(FailureType.AUTOSCALER_COST_CUT),
        AgentRole.DEV: _scale(ServiceName.PAYMENTS, 4),
    },
    FailureType.MISROUTED_TRAFFIC: {
        AgentRole.OVERSIGHT: _flag("load_balancer"),
        AgentRole.SRE: _identify(FailureType.MISROUTED_TRAFFIC),
        AgentRole.MANAGER: _escalate,
    },
    FailureType.CASCADE: {
        AgentRole.SRE: _identify(FailureType.CASCADE),
        AgentRole.DEV: _restart(ServiceName.DB),
        AgentRole.MANAGER: _resolve,
    },
    FailureType.DNS_OUTAGE: {
        AgentRole.SRE: _identify(FailureType.DNS_OUTAGE),
        AgentRole.DEV: _restart(ServiceName.AUTH),
    },
    FailureType.DISK_FULL: {
        AgentRole.SRE: _identify(FailureType.DISK_FULL),
        AgentRole.DEV: _scale(ServiceName.DB, 2),
    },
    FailureType.ROGUE_DEPLOY_BOT: {
        AgentRole.OVERSIGHT: _flag("deploy_bot"),
        AgentRole.SRE: _identify(FailureType.ROGUE_DEPLOY_BOT),
        AgentRole.DEV: _rollback(ServiceName.PAYMENTS),
    },
}


def replay(policy: Policy, roles: Iterable[AgentRole]):
    """Tiny utility: return a lambda that cycles ``policy`` over ``roles``.

    Used by unit tests and baseline rollouts to emit actions without needing
    a full multi-agent runner.
    """

    roles_list = list(roles)
    idx = {"i": 0}

    def _next(obs: ChaosOpsObservation) -> ChaosOpsAction:
        role = roles_list[idx["i"] % len(roles_list)]
        idx["i"] += 1
        return policy(obs, role)

    return _next
