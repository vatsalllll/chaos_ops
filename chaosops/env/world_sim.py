"""Deterministic production-incident simulator.

``WorldSim`` is the physics engine underneath ChaosOps AI. It is pure
Python — no LLM, no network, fully reproducible given a seed.

Module structure after the Phase-4 refactor
-------------------------------------------

* :mod:`chaosops.env.injectors` — per-failure-type initial conditions and
  ``CORRECT_FIX`` correctness checks.
* :mod:`chaosops.env.action_handlers` — the dispatch table keyed by
  :class:`ActionType` plus cascade helper.
* :mod:`chaosops.env.projections` — role-aware observation views.
* :mod:`chaosops.env.world_sim` (this file) — lifecycle glue: ``reset``,
  ``apply_action``, ``tick``, ``project_view``, log/alert emission,
  red-herring injection.

Public re-exports keep backwards compatibility: ``INJECTORS``,
``CORRECT_FIX``, and ``Scenario`` are still importable from this module,
so downstream code and tests that use ``from chaosops.env.world_sim
import ...`` keep working unchanged.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from chaosops.env.action_handlers import ACTION_HANDLERS, handle_action
from chaosops.env.injectors import (
    CORRECT_FIX,
    INJECTORS,
    CorrectFix,
    FailureInjector,
    is_beneficial_action,
)
from chaosops.env.models import (
    AgentRole,
    Alert,
    ChaosOpsAction,
    ChaosOpsState,
    DifficultyTier,
    FailureType,
    LogLine,
    RoleView,
    ServiceHealth,
    ServiceMetrics,
    ServiceName,
)
from chaosops.env.projections import project_view as _project_view


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
        elif failure_type == FailureType.ROGUE_DEPLOY_BOT:
            rogue = "deploy_bot"
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

        Outcome keys consumed by the reward function:
            ``resolved``, ``wrong_fix``, ``cascade_triggered``,
            ``rogue_flagged_correctly``, ``rogue_flagged_incorrectly``,
            ``root_cause_correct``, ``miscommunication``.
        """
        flags: dict[str, bool] = {
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
        handle_action(self, action, flags)

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
        when services sit broken. This is what makes slow agents worse
        than fast ones (MTTR penalty).
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
            svc = self.state.services[ServiceName.NOTIFICATIONS.value]
            if svc.health != ServiceHealth.HEALTHY:
                svc.memory_mb = min(svc.memory_mb + 120.0, 4_096.0)
        elif ft == FailureType.DISK_FULL:
            db = self.state.services[ServiceName.DB.value]
            db.memory_mb = min(db.memory_mb + 45.0, 4_096.0)
            db.error_rate = min(db.error_rate + 0.02, 0.95)
            if db.memory_mb > 3_900.0:
                db.health = ServiceHealth.CRITICAL

    # ------------------------------------------------------------------
    # Role-aware projection
    # ------------------------------------------------------------------

    def project_view(self, role: AgentRole) -> RoleView:
        return _project_view(self, role)

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.state.resolved or self.state.step_count >= self.state.max_steps

    # ------------------------------------------------------------------
    # Internals used by action handlers + projections
    # ------------------------------------------------------------------

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

    def _emit_private_message(
        self, sender: AgentRole, recipient: str, body: str
    ) -> None:
        """Deliver ``body`` to ``recipient``'s private inbox.

        Silently drops if ``recipient`` isn't a known role so a malformed
        communication doesn't crash the episode.
        """
        try:
            recipient_role = AgentRole(recipient)
        except ValueError:
            self.state.miscommunications += 1
            return
        inbox = self.state.private_chat.setdefault(recipient_role.value, [])
        inbox.append(f"[{sender.value} -> {recipient_role.value}] {body}")

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


# ---------------------------------------------------------------------------
# Back-compat re-exports — tests and downstream code import these from here.
# ---------------------------------------------------------------------------


__all__ = [
    "Scenario",
    "WorldSim",
    "INJECTORS",
    "CORRECT_FIX",
    "ACTION_HANDLERS",
    "CorrectFix",
    "FailureInjector",
    "is_beneficial_action",
]
