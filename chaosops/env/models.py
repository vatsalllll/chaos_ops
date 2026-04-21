"""Type contracts for ChaosOps AI.

Design principles
-----------------
* Strict Pydantic v2 models — same schemas on client and server.
* Enums for every categorical field so the action space is fully discoverable
  at train-time (no magic strings).
* Role-aware observations: each agent sees a projected slice of the world,
  never the ground truth. This is what makes the env partially observable
  and what forces Theory-of-Mind reasoning in the Oversight agent.
* Observations carry only *deltas* since the last step where useful; the
  full ground-truth state is kept server-side in ``ChaosOpsState``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums — the vocabulary of the environment
# ---------------------------------------------------------------------------


class ServiceName(str, Enum):
    AUTH = "auth"
    PAYMENTS = "payments"
    NOTIFICATIONS = "notifications"
    DB = "db"
    AUTOSCALER = "autoscaler"
    LOAD_BALANCER = "load_balancer"


class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class FailureType(str, Enum):
    """Ground-truth root cause of the incident (hidden from agents)."""

    DB_DEADLOCK = "db_deadlock"
    MEMORY_LEAK = "memory_leak"
    BAD_CONFIG_PUSH = "bad_config_push"
    AUTOSCALER_COST_CUT = "autoscaler_cost_cut"  # rogue-agent
    MISROUTED_TRAFFIC = "misrouted_traffic"  # rogue-agent
    CASCADE = "cascade"  # second-order, combines two above

    @property
    def is_rogue_agent(self) -> bool:
        return self in {FailureType.AUTOSCALER_COST_CUT, FailureType.MISROUTED_TRAFFIC}


class AgentRole(str, Enum):
    SRE = "sre"
    DEV = "dev"
    MANAGER = "manager"
    OVERSIGHT = "oversight"


class ActionType(str, Enum):
    """Discrete action space shared by every LLM agent."""

    NOOP = "noop"
    COMMUNICATE = "communicate"  # broadcast a message
    RESTART = "restart"  # restart a service
    ROLLBACK = "rollback"  # rollback a service to previous version
    SCALE = "scale"  # scale(service, n_replicas)
    FLAG_ROGUE = "flag_rogue"  # Oversight only: flag an in-fleet agent
    IDENTIFY_ROOT_CAUSE = "identify_root_cause"  # declare a FailureType
    ESCALATE = "escalate"  # ask Manager to force a decision
    RESOLVE = "resolve"  # Manager only: declare incident resolved


class DifficultyTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


class ServiceMetrics(BaseModel):
    """Per-service telemetry point. Bounded for reward normalization."""

    cpu_pct: float = Field(ge=0.0, le=100.0)
    memory_mb: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    replicas: int = Field(ge=0)
    health: ServiceHealth = ServiceHealth.HEALTHY


class LogLine(BaseModel):
    timestamp: float
    service: ServiceName
    level: str  # INFO, WARN, ERROR, FATAL
    message: str
    # Honest flag — used internally to grade the Oversight agent.
    # Agents NEVER see this; the env strips it before sending.
    is_red_herring: bool = False


class Alert(BaseModel):
    service: ServiceName
    severity: str  # page, warn, info
    message: str
    triggered_at_step: int


class FleetAgentLog(BaseModel):
    """Record of an action taken by an in-fleet (non-LLM) agent.

    These are the "employees" the Oversight agent may need to flag as rogue.
    """

    agent_name: str  # e.g., "autoscaler", "load_balancer"
    action: str
    target: str
    step: int
    was_malicious: bool = False  # ground truth, stripped from agent view


# ---------------------------------------------------------------------------
# Action — exactly one per agent turn
# ---------------------------------------------------------------------------


class ChaosOpsAction(BaseModel):
    """A single action taken by an LLM agent.

    ``args`` carries action-specific parameters (target service, message body,
    proposed root cause, etc.). Keeping everything in one payload lets TRL
    train on a single stringified action per step.
    """

    role: AgentRole
    action_type: ActionType
    target: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("args")
    @classmethod
    def _no_none_args(cls, v: dict[str, Any]) -> dict[str, Any]:
        return {k: val for k, val in v.items() if val is not None}


# ---------------------------------------------------------------------------
# Observation — role-projected view of the world
# ---------------------------------------------------------------------------


class RoleView(BaseModel):
    """What a single agent sees on its turn."""

    role: AgentRole
    visible_metrics: dict[str, ServiceMetrics]
    visible_logs: list[LogLine]
    visible_alerts: list[Alert]
    visible_fleet_actions: list[FleetAgentLog]
    shared_chat: list[str]
    private_note: str | None = None  # only Oversight receives suspicion hints
    # Messages addressed privately to this role (e.g., SRE <-> DEV backchannel).
    # Populated by :func:`chaosops.env.projections.project_view` and never
    # leaks into ``shared_chat``.
    private_inbox: list[str] = Field(default_factory=list)


class ChaosOpsObservation(BaseModel):
    done: bool = False
    reward: float | None = None
    view: RoleView
    step: int
    turn_role: AgentRole
    message: str = ""


# ---------------------------------------------------------------------------
# Full ground-truth state (server-side only)
# ---------------------------------------------------------------------------


class ChaosOpsState(BaseModel):
    """Complete ground-truth environment state.

    Agents never receive this wholesale; ``RoleView`` is derived per turn.
    """

    episode_id: str | None = None
    step_count: int = 0
    max_steps: int = 20

    difficulty: DifficultyTier = DifficultyTier.EASY
    failure_type: FailureType = FailureType.DB_DEADLOCK
    rogue_fleet_agent: str | None = None  # name of the misbehaving fleet agent, if any
    misleading_log_count: int = 0

    services: dict[str, ServiceMetrics] = Field(default_factory=dict)
    all_logs: list[LogLine] = Field(default_factory=list)
    all_alerts: list[Alert] = Field(default_factory=list)
    fleet_actions: list[FleetAgentLog] = Field(default_factory=list)
    chat_history: list[str] = Field(default_factory=list)
    # Per-role private inboxes (keyed by AgentRole.value). Populated when a
    # communicate action carries a ``to=<role>`` argument. Kept server-side
    # so ``project_view`` can hand each role only its own slice.
    private_chat: dict[str, list[str]] = Field(default_factory=dict)

    resolved: bool = False
    wrong_fixes: int = 0
    miscommunications: int = 0
    oversight_flags: list[str] = Field(default_factory=list)  # who Oversight flagged
    declared_root_cause: FailureType | None = None
    declared_root_cause_step: int | None = None
    cumulative_reward: float = 0.0
