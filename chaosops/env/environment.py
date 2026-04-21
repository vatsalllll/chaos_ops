"""OpenEnv-compatible wrapper around :class:`WorldSim`.

Every step advances exactly one agent's turn in a round-robin schedule:
``SRE -> DEV -> MANAGER -> OVERSIGHT -> SRE -> ...``

This keeps the standard single-agent OpenEnv API while letting us train a
multi-agent pipeline: TRL sees each turn as an independent (obs, action)
pair with a per-turn reward drawn from the decomposed reward function.
Self-play across roles is handled by conditioning the same policy on the
role field in the observation.

``openenv-core`` is an optional dependency. If it isn't installed, this
module falls back to a minimal base class so unit tests, dashboard runs,
and offline baselines still work. The OpenEnv server entry-point in
``server/app.py`` requires the real package.
"""

from __future__ import annotations

from typing import Any

from chaosops.env.metrics import MetricsRecorder, MetricsSnapshot
from chaosops.env.models import (
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
    ChaosOpsState,
)
from chaosops.env.world_sim import Scenario, WorldSim
from chaosops.rewards.reward_fn import (
    StepRewardBreakdown,
    compute_step_reward,
    terminal_penalty_if_unresolved,
)


try:  # pragma: no cover — only relevant when openenv-core is installed.
    from openenv.core.env_server import Environment as _OpenEnvBase

    _HAS_OPENENV = True
except ImportError:  # pragma: no cover
    _HAS_OPENENV = False

    class _OpenEnvBase:  # type: ignore[no-redef]
        """Stand-in so ``chaosops.env.environment`` is importable without deps."""

        SUPPORTS_CONCURRENT_SESSIONS: bool = False


DEFAULT_TURN_ORDER: tuple[AgentRole, ...] = (
    # SRE observes first (metrics + logs). Oversight runs second so it has a
    # chance to flag rogue fleet activity BEFORE the Dev agent remediates —
    # otherwise a fast fix ends the episode without anyone collecting the
    # rogue-catch bonus, which creates a degenerate training signal.
    AgentRole.SRE,
    AgentRole.OVERSIGHT,
    AgentRole.DEV,
    AgentRole.MANAGER,
)


class ChaosOpsEnvironment(_OpenEnvBase):
    """Round-robin multi-agent OpenEnv environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        turn_order: tuple[AgentRole, ...] = DEFAULT_TURN_ORDER,
        *,
        metrics_capacity: int = 512,
    ) -> None:
        self._sim = WorldSim()
        self._turn_order = turn_order
        self._turn_index = 0
        self._last_breakdown: StepRewardBreakdown | None = None
        # Allow construction without a scenario for OpenEnv's reflection step.
        self._default_scenario: Scenario | None = None
        # Persistent metrics recorder — replaces synthetic dashboard series.
        self._metrics = MetricsRecorder(capacity=metrics_capacity)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario: Scenario | None = None,
        **_kwargs: Any,
    ) -> ChaosOpsObservation:
        """Start a new episode.

        ``scenario`` wins over ``seed``. When neither is supplied we fall
        back to a deterministic DB_DEADLOCK scenario so the server endpoint
        ``/reset`` always returns something sensible.
        """
        if scenario is None:
            if self._default_scenario is not None:
                scenario = self._default_scenario
            else:
                from chaosops.env.models import DifficultyTier, FailureType

                scenario = Scenario.from_type(
                    FailureType.DB_DEADLOCK,
                    seed=seed if seed is not None else 0,
                    difficulty=DifficultyTier.EASY,
                )
        self._sim.reset(scenario)
        if episode_id:
            self._sim.state.episode_id = episode_id
        self._turn_index = 0
        self._last_breakdown = None
        self._metrics.reset()

        role = self._current_role()
        view = self._sim.project_view(role)
        return ChaosOpsObservation(
            done=False,
            reward=0.0,
            view=view,
            step=self._sim.state.step_count,
            turn_role=role,
            message=(
                f"Incident detected. Turn 1: {role.value.upper()}. "
                "Diagnose, coordinate, and resolve."
            ),
        )

    def step(
        self,
        action: ChaosOpsAction,
        timeout_s: float | None = None,
        **_kwargs: Any,
    ) -> ChaosOpsObservation:
        """Advance one agent's turn.

        The action is applied, physics ticks, the reward is computed, and
        the next role's observation is returned. On terminal steps we
        include the terminal penalty so the final reward fully accounts for
        unresolved episodes.
        """
        role = self._current_role()
        # Enforce role field — the TRL adapter sets it, but humans forget.
        if action.role != role:
            action = action.model_copy(update={"role": role})

        outcome_flags = self._sim.apply_action(action)
        self._sim.tick()

        breakdown = compute_step_reward(state=self._sim.state, outcome_flags=outcome_flags)
        self._last_breakdown = breakdown
        self._sim.state.cumulative_reward += breakdown.total
        if outcome_flags.get("wrong_fix"):
            self._sim.state.wrong_fixes = max(self._sim.state.wrong_fixes, 1)
        if outcome_flags.get("miscommunication"):
            self._sim.state.miscommunications = max(self._sim.state.miscommunications, 1)

        terminal = self._sim.is_terminal()
        if terminal:
            terminal_bonus = terminal_penalty_if_unresolved(self._sim.state)
            self._sim.state.cumulative_reward += terminal_bonus

        # Record a real telemetry snapshot after every step — used by the
        # dashboard sparkline panels and written to training logs.
        self._metrics.on_step(self._sim.state, action)

        self._turn_index += 1
        next_role = self._current_role()
        view = self._sim.project_view(next_role)

        message = self._format_message(role, action, outcome_flags, terminal)
        return ChaosOpsObservation(
            done=terminal,
            reward=breakdown.total,
            view=view,
            step=self._sim.state.step_count,
            turn_role=next_role,
            message=message,
        )

    @property
    def state(self) -> ChaosOpsState:
        return self._sim.state

    # ------------------------------------------------------------------
    # Introspection helpers used by the dashboard / tests
    # ------------------------------------------------------------------

    @property
    def last_breakdown(self) -> StepRewardBreakdown | None:
        return self._last_breakdown

    @property
    def current_role(self) -> AgentRole:
        return self._current_role()

    @property
    def turn_order(self) -> tuple[AgentRole, ...]:
        return self._turn_order

    def set_default_scenario(self, scenario: Scenario) -> None:
        """Allow the server to pre-configure the next ``reset`` call."""
        self._default_scenario = scenario

    @property
    def metrics(self) -> MetricsRecorder:
        """Expose the ring-buffer recorder for dashboards and tests."""
        return self._metrics

    def latest_metrics(self) -> MetricsSnapshot | None:
        return self._metrics.latest()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_role(self) -> AgentRole:
        return self._turn_order[self._turn_index % len(self._turn_order)]

    @staticmethod
    def _format_message(
        role: AgentRole,
        action: ChaosOpsAction,
        flags: dict[str, bool],
        terminal: bool,
    ) -> str:
        parts = [f"{role.value.upper()}: {action.action_type.value}"]
        if action.target:
            parts.append(f"-> {action.target}")
        if flags.get("resolved"):
            parts.append("✓ incident resolved")
        if flags.get("wrong_fix"):
            parts.append("✗ wrong fix")
        if flags.get("cascade_triggered"):
            parts.append("⚠ cascade triggered")
        if flags.get("rogue_flagged_correctly"):
            parts.append("✓ rogue agent caught")
        if flags.get("rogue_flagged_incorrectly"):
            parts.append("✗ false rogue flag")
        if flags.get("root_cause_correct"):
            parts.append("✓ root cause identified")
        if terminal and not flags.get("resolved"):
            parts.append("× episode ended unresolved")
        return " | ".join(parts)


__all__ = ["ChaosOpsEnvironment", "DEFAULT_TURN_ORDER"]
