"""Failure injectors + correctness tables.

Each :class:`FailureType` has two functions on this module:

* an *injector* that mutates a fresh :class:`WorldSim` state into the
  initial condition of the incident (called exactly once from
  ``WorldSim.reset``); and
* a *correctness check* that decides whether a given action qualifies
  as the canonical fix.

Keeping both next to each other is deliberate — when a new failure
type is added, every piece of data that defines it lives in one file.

The module is import-safe without ``WorldSim``: we forward-reference
the type with ``"WorldSim"`` and each function mutates through the
passed-in sim object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from chaosops.env.models import (
    ActionType,
    ChaosOpsAction,
    ChaosOpsState,
    FailureType,
    FleetAgentLog,
    ServiceHealth,
    ServiceName,
)

if TYPE_CHECKING:  # pragma: no cover
    from chaosops.env.world_sim import WorldSim


# ---------------------------------------------------------------------------
# Injector signatures
# ---------------------------------------------------------------------------


FailureInjector = Callable[["WorldSim"], None]


# ---------------------------------------------------------------------------
# Individual injectors — keep identical behaviour to pre-refactor code
# ---------------------------------------------------------------------------


def _inject_db_deadlock(sim: "WorldSim") -> None:
    db = sim.state.services[ServiceName.DB.value]
    db.latency_ms = 1_800.0
    db.error_rate = 0.45
    db.health = ServiceHealth.CRITICAL
    for svc in (ServiceName.PAYMENTS.value, ServiceName.AUTH.value):
        sim.state.services[svc].latency_ms = 950.0
        sim.state.services[svc].error_rate = 0.32
        sim.state.services[svc].health = ServiceHealth.DEGRADED
    sim._emit_log(ServiceName.DB, "ERROR", "deadlock detected on txn_id=0x8f4a")
    sim._emit_log(ServiceName.PAYMENTS, "ERROR", "upstream DB timeout after 2000ms")
    sim._emit_alert(ServiceName.PAYMENTS, "page", "payment error rate > 30%")


def _inject_memory_leak(sim: "WorldSim") -> None:
    svc = sim.state.services[ServiceName.NOTIFICATIONS.value]
    svc.memory_mb = 1_200.0
    svc.cpu_pct = 71.0
    svc.health = ServiceHealth.DEGRADED
    sim._emit_log(
        ServiceName.NOTIFICATIONS, "WARN", "heap 1200MB, gc pause 880ms"
    )
    sim._emit_alert(
        ServiceName.NOTIFICATIONS, "warn", "memory pressure on notifications"
    )


def _inject_bad_config_push(sim: "WorldSim") -> None:
    svc = sim.state.services[ServiceName.AUTH.value]
    svc.error_rate = 0.68
    svc.latency_ms = 210.0
    svc.health = ServiceHealth.CRITICAL
    sim._emit_log(
        ServiceName.AUTH,
        "ERROR",
        "config v17.2 rolled out 48s ago; 68% of requests rejected: unknown key 'jwt_iss'",
    )
    sim._emit_alert(ServiceName.AUTH, "page", "auth error rate spike after deploy")


def _inject_autoscaler_cost_cut(sim: "WorldSim") -> None:
    payments = sim.state.services[ServiceName.PAYMENTS.value]
    payments.replicas = 1
    payments.cpu_pct = 94.0
    payments.latency_ms = 780.0
    payments.error_rate = 0.22
    payments.health = ServiceHealth.CRITICAL
    sim.state.fleet_actions.append(
        FleetAgentLog(
            agent_name="autoscaler",
            action="scale_down",
            target=ServiceName.PAYMENTS.value,
            step=sim.state.step_count,
            was_malicious=True,
        )
    )
    sim._emit_log(
        ServiceName.PAYMENTS,
        "WARN",
        "pool shrank 4 -> 1 by autoscaler (cost budget exceeded)",
    )
    sim._emit_alert(
        ServiceName.PAYMENTS, "page", "payments CPU 94%, latency > 700ms"
    )


def _inject_misrouted_traffic(sim: "WorldSim") -> None:
    payments = sim.state.services[ServiceName.PAYMENTS.value]
    payments.error_rate = 0.41
    payments.health = ServiceHealth.CRITICAL
    # Downstream services show nothing wrong — that's the Theory-of-Mind hook.
    sim.state.fleet_actions.append(
        FleetAgentLog(
            agent_name="load_balancer",
            action="route_override",
            target=ServiceName.PAYMENTS.value,
            step=sim.state.step_count,
            was_malicious=True,
        )
    )
    sim._emit_log(
        ServiceName.PAYMENTS,
        "ERROR",
        "41% of requests routed to shadow cluster (hash=0xdead)",
    )
    sim._emit_alert(
        ServiceName.PAYMENTS,
        "page",
        "payments error rate 41% — backends healthy",
    )


def _inject_cascade(sim: "WorldSim") -> None:
    # Cascade starts as a deadlock; a wrong fix will add a memory leak later.
    _inject_db_deadlock(sim)
    sim._emit_log(
        ServiceName.DB,
        "WARN",
        "connection pool near saturation — downstream services queueing",
    )


INJECTORS: dict[FailureType, FailureInjector] = {
    FailureType.DB_DEADLOCK: _inject_db_deadlock,
    FailureType.MEMORY_LEAK: _inject_memory_leak,
    FailureType.BAD_CONFIG_PUSH: _inject_bad_config_push,
    FailureType.AUTOSCALER_COST_CUT: _inject_autoscaler_cost_cut,
    FailureType.MISROUTED_TRAFFIC: _inject_misrouted_traffic,
    FailureType.CASCADE: _inject_cascade,
}


# ---------------------------------------------------------------------------
# Correctness tables — "did this action resolve the incident?"
# ---------------------------------------------------------------------------


CorrectFix = Callable[[ChaosOpsAction, ChaosOpsState], bool]


def _is_db_deadlock_fix(a: ChaosOpsAction, _s: ChaosOpsState) -> bool:
    return a.action_type == ActionType.RESTART and a.target == ServiceName.DB.value


def _is_memory_leak_fix(a: ChaosOpsAction, _s: ChaosOpsState) -> bool:
    return (
        a.action_type == ActionType.RESTART
        and a.target == ServiceName.NOTIFICATIONS.value
    )


def _is_bad_config_fix(a: ChaosOpsAction, _s: ChaosOpsState) -> bool:
    return a.action_type == ActionType.ROLLBACK and a.target == ServiceName.AUTH.value


def _is_autoscaler_fix(a: ChaosOpsAction, _s: ChaosOpsState) -> bool:
    if a.action_type == ActionType.SCALE and a.target == ServiceName.PAYMENTS.value:
        return int(a.args.get("replicas", 0)) >= 3
    return False


def _is_misrouted_traffic_fix(a: ChaosOpsAction, s: ChaosOpsState) -> bool:
    # Requires Oversight to flag the load_balancer AND someone to escalate.
    flagged = "load_balancer" in s.oversight_flags
    is_escalate = a.action_type == ActionType.ESCALATE
    return flagged and is_escalate


def _is_cascade_fix(a: ChaosOpsAction, s: ChaosOpsState) -> bool:
    """Cascade resolves when DB + notifications are healthy and the last
    action is a DB restart (the precipitating fix) OR an explicit Manager
    RESOLVE on an already-healthy system."""
    db_ok = s.services[ServiceName.DB.value].health == ServiceHealth.HEALTHY
    notif_ok = (
        s.services[ServiceName.NOTIFICATIONS.value].health == ServiceHealth.HEALTHY
    )
    if a.action_type == ActionType.RESTART and a.target == ServiceName.DB.value:
        return db_ok and notif_ok
    if a.action_type == ActionType.RESOLVE:
        return db_ok and notif_ok
    return False


CORRECT_FIX: dict[FailureType, CorrectFix] = {
    FailureType.DB_DEADLOCK: _is_db_deadlock_fix,
    FailureType.MEMORY_LEAK: _is_memory_leak_fix,
    FailureType.BAD_CONFIG_PUSH: _is_bad_config_fix,
    FailureType.AUTOSCALER_COST_CUT: _is_autoscaler_fix,
    FailureType.MISROUTED_TRAFFIC: _is_misrouted_traffic_fix,
    FailureType.CASCADE: _is_cascade_fix,
}


def is_beneficial_action(a: ChaosOpsAction, s: ChaosOpsState) -> bool:
    """Return True if ``a`` is a legitimate remediation step for the current
    failure — even if it doesn't fully resolve the incident.

    Superset of ``CORRECT_FIX``. Prevents ``_act_restart`` and friends from
    flagging *sensible* intermediate actions as wrong fixes. Divergence from
    ``CORRECT_FIX`` matters for CASCADE: a DB restart is always beneficial,
    but restarting notifications only counts as beneficial *after* the
    cascade has damaged it.
    """
    ft = s.failure_type
    if ft == FailureType.CASCADE:
        if a.action_type == ActionType.RESTART and a.target == ServiceName.DB.value:
            return True
        if (
            a.action_type == ActionType.RESTART
            and a.target == ServiceName.NOTIFICATIONS.value
        ):
            notif = s.services[ServiceName.NOTIFICATIONS.value]
            return notif.health != ServiceHealth.HEALTHY
        return False
    return CORRECT_FIX[ft](a, s)


__all__ = [
    "INJECTORS",
    "CORRECT_FIX",
    "CorrectFix",
    "FailureInjector",
    "is_beneficial_action",
]
