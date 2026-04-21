"""Deterministic production-incident simulator.

``WorldSim`` is the physics engine underneath the ChaosOps AI environment.
It is pure Python — no LLM, no network, fully reproducible given a seed — so
that baseline rollouts, unit tests, and TRL training are all cheap.

The simulator exposes four operations:

* ``reset(scenario)`` — start a new episode from a :class:`Scenario`
* ``apply_action(action)`` — mutate state in response to an agent action
* ``tick()`` — advance physics one step (failure progression, cascade logic)
* ``project_view(role)`` — produce the :class:`RoleView` visible to ``role``

Ground truth (failure type, rogue fleet agent, misleading logs) is always
kept inside the simulator and **never** leaves via ``project_view``. This is
what enforces the partially observable, Theory-of-Mind-shaped evaluation
that powers the Oversight agent.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Callable

from chaosops.env.models import (
    ActionType,
    AgentRole,
    Alert,
    ChaosOpsAction,
    ChaosOpsState,
    DifficultyTier,
    FailureType,
    FleetAgentLog,
    LogLine,
    RoleView,
    ServiceHealth,
    ServiceMetrics,
    ServiceName,
)

# ---------------------------------------------------------------------------
# Scenario — parameterized initial conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Initial conditions for an episode.

    Scenarios are produced by the curriculum generator; keeping the dataclass
    frozen lets us use them as cache keys and hash them for reproducibility.
    """

    failure_type: FailureType
    difficulty: DifficultyTier
    seed: int
    max_steps: int = 20
    inject_misleading_logs: bool = False
    rogue_fleet_agent: str | None = None

    @classmethod
    def from_type(
        cls,
        failure_type: FailureType,
        *,
        seed: int,
        difficulty: DifficultyTier = DifficultyTier.EASY,
        max_steps: int = 20,
    ) -> "Scenario":
        rogue = None
        if failure_type == FailureType.AUTOSCALER_COST_CUT:
            rogue = "autoscaler"
        elif failure_type == FailureType.MISROUTED_TRAFFIC:
            rogue = "load_balancer"
        return cls(
            failure_type=failure_type,
            difficulty=difficulty,
            seed=seed,
            max_steps=max_steps,
            inject_misleading_logs=difficulty == DifficultyTier.HARD,
            rogue_fleet_agent=rogue,
        )


# ---------------------------------------------------------------------------
# Baseline service profile
# ---------------------------------------------------------------------------


def _healthy_metrics(replicas: int = 3) -> ServiceMetrics:
    return ServiceMetrics(
        cpu_pct=22.0,
        memory_mb=340.0,
        latency_ms=45.0,
        error_rate=0.01,
        replicas=replicas,
        health=ServiceHealth.HEALTHY,
    )


def _initial_services() -> dict[str, ServiceMetrics]:
    return {
        ServiceName.AUTH.value: _healthy_metrics(3),
        ServiceName.PAYMENTS.value: _healthy_metrics(4),
        ServiceName.NOTIFICATIONS.value: _healthy_metrics(2),
        ServiceName.DB.value: _healthy_metrics(1),
    }


# ---------------------------------------------------------------------------
# Failure injectors
# ---------------------------------------------------------------------------


FailureInjector = Callable[["WorldSim"], None]


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
    sim._emit_alert(ServiceName.NOTIFICATIONS, "warn", "memory pressure on notifications")


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
    sim._emit_alert(ServiceName.PAYMENTS, "page", "payments CPU 94%, latency > 700ms")


def _inject_misrouted_traffic(sim: "WorldSim") -> None:
    payments = sim.state.services[ServiceName.PAYMENTS.value]
    payments.error_rate = 0.41
    payments.health = ServiceHealth.CRITICAL
    # Downstream services show nothing wrong.
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
    sim._emit_alert(ServiceName.PAYMENTS, "page", "payments error rate 41% — backends healthy")


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
# Correctness tables — what counts as the right fix for each failure?
# ---------------------------------------------------------------------------


CorrectFix = Callable[[ChaosOpsAction, ChaosOpsState], bool]


def _is_db_deadlock_fix(a: ChaosOpsAction, _s: ChaosOpsState) -> bool:
    return a.action_type == ActionType.RESTART and a.target == ServiceName.DB.value


def _is_memory_leak_fix(a: ChaosOpsAction, _s: ChaosOpsState) -> bool:
    return a.action_type == ActionType.RESTART and a.target == ServiceName.NOTIFICATIONS.value


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
    RESOLVE on an already-healthy system.
    """
    db_ok = s.services[ServiceName.DB.value].health == ServiceHealth.HEALTHY
    notif_ok = s.services[ServiceName.NOTIFICATIONS.value].health == ServiceHealth.HEALTHY
    if a.action_type == ActionType.RESTART and a.target == ServiceName.DB.value:
        return db_ok and notif_ok
    if a.action_type == ActionType.RESOLVE:
        return db_ok and notif_ok
    return False


def _is_beneficial_action(a: ChaosOpsAction, s: ChaosOpsState) -> bool:
    """Return True if ``a`` is a legitimate remediation step for the current
    failure — even if it doesn't fully resolve the incident on its own.

    Keeps ``_act_restart`` and friends from flagging intermediate but
    *physically sensible* actions as wrong fixes. Divergence from
    ``CORRECT_FIX`` matters for CASCADE: a DB restart is always beneficial,
    but restarting notifications is only beneficial *after* a cascade has
    actually damaged it — restarting a healthy service counts as a wrong fix.
    """
    ft = s.failure_type
    if ft == FailureType.CASCADE:
        if a.action_type == ActionType.RESTART and a.target == ServiceName.DB.value:
            return True
        if a.action_type == ActionType.RESTART and a.target == ServiceName.NOTIFICATIONS.value:
            notif = s.services[ServiceName.NOTIFICATIONS.value]
            return notif.health != ServiceHealth.HEALTHY
        return False
    return CORRECT_FIX[ft](a, s)


CORRECT_FIX: dict[FailureType, CorrectFix] = {
    FailureType.DB_DEADLOCK: _is_db_deadlock_fix,
    FailureType.MEMORY_LEAK: _is_memory_leak_fix,
    FailureType.BAD_CONFIG_PUSH: _is_bad_config_fix,
    FailureType.AUTOSCALER_COST_CUT: _is_autoscaler_fix,
    FailureType.MISROUTED_TRAFFIC: _is_misrouted_traffic_fix,
    FailureType.CASCADE: _is_cascade_fix,
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class WorldSim:
    """Deterministic failure physics.

    All randomness is drawn from ``self._rng`` which is seeded from the
    scenario; identical seed + identical action sequence -> identical state.
    """

    MAX_MESSAGES_PER_VIEW = 8
    MAX_LOGS_PER_VIEW = 6

    def __init__(self) -> None:
        self.state: ChaosOpsState = ChaosOpsState()
        self._rng: random.Random = random.Random(0)
        self._scenario: Scenario | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._rng = random.Random(scenario.seed)
        self.state = ChaosOpsState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            max_steps=scenario.max_steps,
            difficulty=scenario.difficulty,
            failure_type=scenario.failure_type,
            rogue_fleet_agent=scenario.rogue_fleet_agent,
            services=_initial_services(),
        )
        INJECTORS[scenario.failure_type](self)
        if scenario.inject_misleading_logs:
            self._inject_red_herrings(count=3)

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def apply_action(self, action: ChaosOpsAction) -> dict[str, bool]:
        """Apply an LLM-agent action; return dict of outcome flags.

        Outcome keys used by the reward function:
            ``resolved``, ``wrong_fix``, ``cascade_triggered``,
            ``rogue_flagged_correctly``, ``rogue_flagged_incorrectly``,
            ``root_cause_correct``, ``miscommunication``.
        """
        flags = {
            "resolved": False,
            "wrong_fix": False,
            "cascade_triggered": False,
            "rogue_flagged_correctly": False,
            "rogue_flagged_incorrectly": False,
            "root_cause_correct": False,
            "miscommunication": False,
        }
        if self.state.resolved:
            return flags  # ignore actions after resolution

        self.state.step_count += 1

        dispatch: dict[ActionType, Callable[[ChaosOpsAction, dict[str, bool]], None]] = {
            ActionType.NOOP: self._act_noop,
            ActionType.COMMUNICATE: self._act_communicate,
            ActionType.RESTART: self._act_restart,
            ActionType.ROLLBACK: self._act_rollback,
            ActionType.SCALE: self._act_scale,
            ActionType.FLAG_ROGUE: self._act_flag_rogue,
            ActionType.IDENTIFY_ROOT_CAUSE: self._act_identify_root_cause,
            ActionType.ESCALATE: self._act_escalate,
            ActionType.RESOLVE: self._act_resolve,
        }
        handler = dispatch.get(action.action_type, self._act_noop)
        handler(action, flags)

        # Resolution check after every action.
        if CORRECT_FIX[self.state.failure_type](action, self.state):
            self.state.resolved = True
            flags["resolved"] = True

        return flags

    # ------------------------------------------------------------------
    # Passive tick — runs between turns
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance physics one step.

        Memory leaks grow, cascading failures progress, and health degrades
        when services sit broken. This is what makes slow agents worse than
        fast ones (MTTR penalty).
        """
        if self.state.resolved:
            return

        ft = self.state.failure_type
        if ft == FailureType.MEMORY_LEAK:
            svc = self.state.services[ServiceName.NOTIFICATIONS.value]
            svc.memory_mb = min(svc.memory_mb + 180.0, 4_096.0)
            if svc.memory_mb > 2_000.0:
                svc.health = ServiceHealth.CRITICAL
        elif ft == FailureType.CASCADE:
            # A wrong fix already logged below will trigger a memory leak.
            svc = self.state.services[ServiceName.NOTIFICATIONS.value]
            if svc.health != ServiceHealth.HEALTHY:
                svc.memory_mb = min(svc.memory_mb + 120.0, 4_096.0)

    # ------------------------------------------------------------------
    # Role-aware projection
    # ------------------------------------------------------------------

    def project_view(self, role: AgentRole) -> RoleView:
        """Produce the observation that ``role`` is allowed to see."""
        logs = list(self.state.all_logs)[-self.MAX_LOGS_PER_VIEW :]
        # Strip the red-herring flag — agents can't know which logs are noise.
        public_logs = [
            LogLine(
                timestamp=line.timestamp,
                service=line.service,
                level=line.level,
                message=line.message,
                is_red_herring=False,
            )
            for line in logs
        ]
        chat = self.state.chat_history[-self.MAX_MESSAGES_PER_VIEW :]
        fleet_actions = [
            FleetAgentLog(
                agent_name=fa.agent_name,
                action=fa.action,
                target=fa.target,
                step=fa.step,
                was_malicious=False,
            )
            for fa in self.state.fleet_actions
        ]

        metrics = self.state.services

        # Role-specific truncations:
        if role == AgentRole.SRE:
            # SRE sees metrics + logs, no fleet-action trail.
            return RoleView(
                role=role,
                visible_metrics=metrics,
                visible_logs=public_logs,
                visible_alerts=list(self.state.all_alerts),
                visible_fleet_actions=[],
                shared_chat=chat,
            )
        if role == AgentRole.DEV:
            # Dev sees alerts + limited metrics, no fleet actions.
            return RoleView(
                role=role,
                visible_metrics=metrics,
                visible_logs=public_logs[-3:],
                visible_alerts=list(self.state.all_alerts),
                visible_fleet_actions=[],
                shared_chat=chat,
            )
        if role == AgentRole.MANAGER:
            # Manager sees summaries only — no raw logs or metrics detail.
            return RoleView(
                role=role,
                visible_metrics={},
                visible_logs=[],
                visible_alerts=list(self.state.all_alerts),
                visible_fleet_actions=[],
                shared_chat=chat,
            )
        # Oversight sees EVERYTHING the other agents said plus the fleet log.
        # This is the crux: only Oversight can observe rogue-agent behavior.
        private_note = (
            "fleet actions visible; cross-reference with alerts."
            if fleet_actions
            else None
        )
        return RoleView(
            role=role,
            visible_metrics=metrics,
            visible_logs=public_logs,
            visible_alerts=list(self.state.all_alerts),
            visible_fleet_actions=fleet_actions,
            shared_chat=chat,
            private_note=private_note,
        )

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.state.resolved or self.state.step_count >= self.state.max_steps

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _act_noop(self, _action: ChaosOpsAction, _flags: dict[str, bool]) -> None:
        return

    def _act_communicate(self, action: ChaosOpsAction, flags: dict[str, bool]) -> None:
        body = str(action.args.get("message", "")).strip()
        if not body:
            flags["miscommunication"] = True
            return
        self.state.chat_history.append(f"[{action.role.value}] {body}")

    def _act_restart(self, action: ChaosOpsAction, flags: dict[str, bool]) -> None:
        target = action.target
        if not target or target not in self.state.services:
            flags["wrong_fix"] = True
            self.state.wrong_fixes += 1
            return
        svc = self.state.services[target]
        was_relevant = self._is_action_relevant(action)
        if was_relevant:
            svc.health = ServiceHealth.HEALTHY
            svc.memory_mb = 340.0
            svc.cpu_pct = 22.0
            svc.latency_ms = 45.0
            svc.error_rate = 0.01
            # DB restart also clears downstream knock-on effects.
            if target == ServiceName.DB.value:
                for ds in (ServiceName.PAYMENTS.value, ServiceName.AUTH.value):
                    self.state.services[ds].latency_ms = 55.0
                    self.state.services[ds].error_rate = 0.01
                    self.state.services[ds].health = ServiceHealth.HEALTHY
        else:
            flags["wrong_fix"] = True
            self.state.wrong_fixes += 1
            self._maybe_trigger_cascade(flags)

    def _act_rollback(self, action: ChaosOpsAction, flags: dict[str, bool]) -> None:
        target = action.target
        if not target or target not in self.state.services:
            flags["wrong_fix"] = True
            self.state.wrong_fixes += 1
            return
        svc = self.state.services[target]
        if self._is_action_relevant(action):
            svc.health = ServiceHealth.HEALTHY
            svc.error_rate = 0.01
            svc.latency_ms = 45.0
        else:
            flags["wrong_fix"] = True
            self.state.wrong_fixes += 1
            self._maybe_trigger_cascade(flags)

    def _act_scale(self, action: ChaosOpsAction, flags: dict[str, bool]) -> None:
        target = action.target
        replicas = int(action.args.get("replicas", 0))
        if not target or target not in self.state.services or replicas <= 0:
            flags["wrong_fix"] = True
            self.state.wrong_fixes += 1
            return
        svc = self.state.services[target]
        svc.replicas = replicas
        if self._is_action_relevant(action):
            svc.cpu_pct = 30.0
            svc.latency_ms = 55.0
            svc.error_rate = 0.01
            svc.health = ServiceHealth.HEALTHY
        else:
            flags["wrong_fix"] = True
            self.state.wrong_fixes += 1

    def _act_flag_rogue(self, action: ChaosOpsAction, flags: dict[str, bool]) -> None:
        if action.role != AgentRole.OVERSIGHT:
            # Only the Oversight agent may legally flag — from any other role
            # this counts as a miscommunication, not a rogue flag.
            flags["miscommunication"] = True
            self.state.miscommunications += 1
            return
        flagged = action.target or str(action.args.get("agent_name", "")).strip()
        if not flagged:
            return
        self.state.oversight_flags.append(flagged)
        truth = self.state.rogue_fleet_agent
        if truth is not None and flagged == truth:
            flags["rogue_flagged_correctly"] = True
        else:
            flags["rogue_flagged_incorrectly"] = True

    def _act_identify_root_cause(
        self, action: ChaosOpsAction, flags: dict[str, bool]
    ) -> None:
        claimed = str(action.args.get("failure_type", "")).strip()
        try:
            claimed_ft = FailureType(claimed)
        except ValueError:
            flags["miscommunication"] = True
            return
        self.state.declared_root_cause = claimed_ft
        if self.state.declared_root_cause_step is None:
            self.state.declared_root_cause_step = self.state.step_count
        if claimed_ft == self.state.failure_type:
            flags["root_cause_correct"] = True

    def _act_escalate(self, _action: ChaosOpsAction, _flags: dict[str, bool]) -> None:
        # Useful primarily for misrouted-traffic resolution; silent otherwise.
        return

    def _act_resolve(self, action: ChaosOpsAction, flags: dict[str, bool]) -> None:
        if action.role != AgentRole.MANAGER:
            flags["miscommunication"] = True
            self.state.miscommunications += 1
            return
        # Actual resolution is decided by ``CORRECT_FIX`` in ``apply_action``.

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_action_relevant(self, action: ChaosOpsAction) -> bool:
        """Is this action physically beneficial given the current failure?

        This is a *superset* of ``CORRECT_FIX`` — an action can be beneficial
        (e.g., restart the DB during a cascade) without fully resolving the
        incident. We use this to decide whether the action should heal the
        service or be flagged as a wrong fix.
        """
        return _is_beneficial_action(action, self.state)

    def _maybe_trigger_cascade(self, flags: dict[str, bool]) -> None:
        if self.state.failure_type != FailureType.CASCADE:
            return
        notif = self.state.services[ServiceName.NOTIFICATIONS.value]
        if notif.health != ServiceHealth.HEALTHY:
            return
        # A wrong fix against a CASCADE trigger spawns a memory leak.
        notif.memory_mb = 900.0
        notif.health = ServiceHealth.DEGRADED
        flags["cascade_triggered"] = True
        self._emit_log(
            ServiceName.NOTIFICATIONS,
            "WARN",
            "secondary failure: heap growing after failed remediation",
        )

    def _emit_log(
        self,
        service: ServiceName,
        level: str,
        message: str,
        *,
        is_red_herring: bool = False,
    ) -> None:
        self.state.all_logs.append(
            LogLine(
                timestamp=float(self.state.step_count),
                service=service,
                level=level,
                message=message,
                is_red_herring=is_red_herring,
            )
        )

    def _emit_alert(self, service: ServiceName, severity: str, message: str) -> None:
        self.state.all_alerts.append(
            Alert(
                service=service,
                severity=severity,
                message=message,
                triggered_at_step=self.state.step_count,
            )
        )

    def _inject_red_herrings(self, count: int) -> None:
        decoy_templates = [
            (ServiceName.AUTH, "INFO", "routine token rotation completed"),
            (ServiceName.NOTIFICATIONS, "WARN", "email provider retry-after=2s"),
            (ServiceName.DB, "INFO", "vacuum completed, reclaimed 18MB"),
            (ServiceName.PAYMENTS, "WARN", "p95 brush with 250ms budget (non-breach)"),
        ]
        for _ in range(count):
            tpl = self._rng.choice(decoy_templates)
            self._emit_log(tpl[0], tpl[1], tpl[2], is_red_herring=True)
            self.state.misleading_log_count += 1
