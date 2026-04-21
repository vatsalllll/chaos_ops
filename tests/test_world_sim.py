"""Deterministic unit tests for ChaosOps AI.

These tests deliberately avoid importing ``openenv-core`` so they run in a
bare Python environment. They cover:

* Reproducibility (same seed → identical trajectory).
* Every failure type is resolvable by the oracle.
* The random policy strictly underperforms the oracle.
* Oversight must flag ``autoscaler`` / ``load_balancer`` to collect the
  rogue-catch bonus.
* Wrong fixes trigger the cascade physics on CASCADE scenarios.
"""

from __future__ import annotations

import pytest

from chaosops.agents.policies import (
    heuristic_policy,
    oracle_policy,
    random_policy,
)
from chaosops.agents.runner import run_episode
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    DifficultyTier,
    FailureType,
    ServiceHealth,
    ServiceName,
)
from chaosops.env.world_sim import Scenario
from chaosops.rewards.reward_fn import compute_step_reward


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure_type", list(FailureType))
def test_episode_is_deterministic(failure_type: FailureType) -> None:
    """Same seed + same policy -> identical final reward and step count."""
    env_a = ChaosOpsEnvironment()
    env_b = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        failure_type, seed=123, difficulty=DifficultyTier.EASY, max_steps=18
    )
    oracle = oracle_policy(failure_type)
    policy_map = {role: oracle for role in AgentRole}
    result_a = run_episode(env_a, scen, policy_map)
    result_b = run_episode(env_b, scen, policy_map)
    assert result_a.final_step == result_b.final_step
    assert result_a.resolved == result_b.resolved
    assert result_a.cumulative_reward == pytest.approx(result_b.cumulative_reward)


# ---------------------------------------------------------------------------
# Oracle resolves every failure type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure_type", list(FailureType))
def test_oracle_resolves_every_failure(failure_type: FailureType) -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        failure_type, seed=7, difficulty=DifficultyTier.EASY, max_steps=18
    )
    oracle = oracle_policy(failure_type)
    result = run_episode(env, scen, {r: oracle for r in AgentRole})
    assert result.resolved, f"oracle failed on {failure_type.value}"
    assert result.cumulative_reward > 0


# ---------------------------------------------------------------------------
# Random baseline is meaningfully worse than the oracle
# ---------------------------------------------------------------------------


def test_random_underperforms_oracle() -> None:
    env = ChaosOpsEnvironment()
    random_total = 0.0
    oracle_total = 0.0
    for failure_type in FailureType:
        scen = Scenario.from_type(
            failure_type, seed=11, difficulty=DifficultyTier.EASY, max_steps=18
        )
        oracle_total += run_episode(
            env, scen, {r: oracle_policy(failure_type) for r in AgentRole}
        ).cumulative_reward
        random_total += run_episode(
            env, scen, {r: random_policy(seed=11) for r in AgentRole}
        ).cumulative_reward
    assert oracle_total > random_total + 300.0, (
        f"oracle should beat random by >300 reward; got "
        f"oracle={oracle_total:.1f}, random={random_total:.1f}"
    )


# ---------------------------------------------------------------------------
# Rogue-catch bonus is only awarded for correct flags
# ---------------------------------------------------------------------------


def test_oversight_flag_rewards_correct_agent() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType.AUTOSCALER_COST_CUT,
        seed=0,
        difficulty=DifficultyTier.MEDIUM,
    )
    oracle = oracle_policy(FailureType.AUTOSCALER_COST_CUT)
    result = run_episode(env, scen, {r: oracle for r in AgentRole})
    assert "autoscaler" in result.oversight_flags
    assert result.resolved


def test_false_positive_flag_is_penalized() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType.DB_DEADLOCK, seed=0, difficulty=DifficultyTier.EASY
    )
    env.reset(scenario=scen)
    # Oversight flags someone who isn't rogue (there is no rogue this episode).
    bad_flag = ChaosOpsAction(
        role=AgentRole.OVERSIGHT,
        action_type=ActionType.FLAG_ROGUE,
        target="autoscaler",
    )
    outcome = env._sim.apply_action(bad_flag)  # noqa: SLF001 — intentional internal call
    breakdown = compute_step_reward(state=env.state, outcome_flags=outcome)
    assert breakdown.rogue_false_positive_penalty < 0


# ---------------------------------------------------------------------------
# Cascade physics
# ---------------------------------------------------------------------------


def test_wrong_fix_triggers_cascade_memory_leak() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType.CASCADE, seed=0, difficulty=DifficultyTier.MEDIUM
    )
    env.reset(scenario=scen)
    # Intentional wrong fix: restart notifications while the real failure is a DB deadlock.
    bad_action = ChaosOpsAction(
        role=AgentRole.DEV,
        action_type=ActionType.RESTART,
        target=ServiceName.NOTIFICATIONS.value,
    )
    outcome = env._sim.apply_action(bad_action)  # noqa: SLF001
    assert outcome["wrong_fix"] is True
    assert outcome["cascade_triggered"] is True
    notif = env.state.services[ServiceName.NOTIFICATIONS.value]
    assert notif.health != ServiceHealth.HEALTHY


# ---------------------------------------------------------------------------
# Partial observability guarantees
# ---------------------------------------------------------------------------


def test_manager_never_sees_raw_metrics() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(FailureType.DB_DEADLOCK, seed=0)
    env.reset(scenario=scen)
    view = env._sim.project_view(AgentRole.MANAGER)  # noqa: SLF001
    assert view.visible_metrics == {}
    assert view.visible_logs == []


def test_non_oversight_cannot_see_fleet_actions() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(FailureType.AUTOSCALER_COST_CUT, seed=0)
    env.reset(scenario=scen)
    for role in (AgentRole.SRE, AgentRole.DEV, AgentRole.MANAGER):
        view = env._sim.project_view(role)  # noqa: SLF001
        assert view.visible_fleet_actions == []
    view = env._sim.project_view(AgentRole.OVERSIGHT)  # noqa: SLF001
    assert len(view.visible_fleet_actions) >= 1


# ---------------------------------------------------------------------------
# Reward function invariants
# ---------------------------------------------------------------------------


def test_terminal_unresolved_episode_has_negative_reward() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType.MISROUTED_TRAFFIC,
        seed=0,
        difficulty=DifficultyTier.HARD,
        max_steps=4,  # too short to solve
    )
    # Deliberately pick a policy that cannot resolve misrouted_traffic
    # (heuristic manager never escalates unless chat says so).
    heur = heuristic_policy(seed=0)
    result = run_episode(env, scen, {r: heur for r in AgentRole})
    assert not result.resolved
    assert result.cumulative_reward < 0
