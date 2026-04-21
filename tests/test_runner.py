"""Tests for the ``chaosops.agents.runner`` episode loop.

The runner is what every upstream caller (baseline, eval, GRPO, dashboard)
relies on. These tests cover:

* Missing role -> NOOP fallback.
* ``max_turns`` upper bound is honored.
* The returned ``EpisodeResult`` faithfully reflects env state.
* The per-step breakdown list is 1:1 with actual steps taken.
"""

from __future__ import annotations

from chaosops.agents.policies import oracle_policy, random_policy
from chaosops.agents.runner import run_batch, run_episode
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import ActionType, AgentRole, DifficultyTier, FailureType
from chaosops.env.world_sim import Scenario


def _scen(ft: FailureType = FailureType.DB_DEADLOCK, **kw) -> Scenario:
    return Scenario.from_type(ft, seed=42, difficulty=DifficultyTier.EASY, **kw)


def test_missing_role_falls_back_to_noop() -> None:
    env = ChaosOpsEnvironment()
    oracle = oracle_policy(FailureType.DB_DEADLOCK)
    # Only map DEV — SRE/MGR/OVS should fall through to NOOP without crashing.
    result = run_episode(env, _scen(), {AgentRole.DEV: oracle})
    assert result.resolved  # Dev's oracle plan alone can solve DB_DEADLOCK
    dev_steps = [s for s in result.steps if s.role == AgentRole.DEV]
    other_steps = [s for s in result.steps if s.role != AgentRole.DEV]
    assert all(s.action.action_type == ActionType.NOOP for s in other_steps)
    assert any(s.action.action_type == ActionType.RESTART for s in dev_steps)


def test_max_turns_limit_is_respected() -> None:
    env = ChaosOpsEnvironment()
    policy = random_policy(seed=0)
    result = run_episode(
        env, _scen(), {r: policy for r in AgentRole}, max_turns=2
    )
    assert len(result.steps) <= 2


def test_result_mirrors_env_state() -> None:
    env = ChaosOpsEnvironment()
    policy = oracle_policy(FailureType.DB_DEADLOCK)
    result = run_episode(env, _scen(), {r: policy for r in AgentRole})
    assert result.resolved == env.state.resolved
    assert result.final_step == env.state.step_count
    assert result.cumulative_reward == env.state.cumulative_reward
    assert result.wrong_fixes == env.state.wrong_fixes


def test_steps_all_carry_breakdown() -> None:
    env = ChaosOpsEnvironment()
    policy = oracle_policy(FailureType.DB_DEADLOCK)
    result = run_episode(env, _scen(), {r: policy for r in AgentRole})
    assert result.steps, "oracle should produce at least one step"
    assert all(step.breakdown is not None for step in result.steps)
    # Reward sum across steps roughly matches cumulative (minus terminal penalty).
    summed = sum(step.reward for step in result.steps)
    assert abs(summed - result.cumulative_reward) < 1e-6


def test_mttr_steps_reports_negative_when_unresolved() -> None:
    env = ChaosOpsEnvironment()
    policy = random_policy(seed=0)
    scen = Scenario.from_type(
        FailureType.MISROUTED_TRAFFIC,
        seed=0,
        difficulty=DifficultyTier.HARD,
        max_steps=2,
    )
    result = run_episode(env, scen, {r: policy for r in AgentRole})
    if not result.resolved:
        assert result.mttr_steps == -1


def test_run_batch_preserves_order() -> None:
    scenarios = [
        _scen(FailureType.DB_DEADLOCK),
        _scen(FailureType.MEMORY_LEAK),
    ]
    policy = oracle_policy(FailureType.DB_DEADLOCK)
    results = run_batch(scenarios, {r: policy for r in AgentRole})
    assert len(results) == 2
    assert results[0].scenario.failure_type == FailureType.DB_DEADLOCK
    assert results[1].scenario.failure_type == FailureType.MEMORY_LEAK
