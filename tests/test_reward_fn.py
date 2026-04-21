"""Unit tests for ``chaosops.rewards.reward_fn``.

Every component of the reward formula is checked in isolation so the
aggregate ``StepRewardBreakdown`` behaviour is auditable component by
component. These tests intentionally never touch ``WorldSim`` — they
feed ``compute_step_reward`` directly.
"""

from __future__ import annotations

import pytest

from chaosops.env.models import ChaosOpsState, FailureType
from chaosops.rewards.reward_fn import (
    combine_rewards,
    compute_step_reward,
    terminal_penalty_if_unresolved,
)


def _state(**kwargs) -> ChaosOpsState:
    defaults = dict(step_count=3, failure_type=FailureType.DB_DEADLOCK)
    defaults.update(kwargs)
    return ChaosOpsState(**defaults)


# ---------------------------------------------------------------------------
# Individual components
# ---------------------------------------------------------------------------


def test_resolved_awards_fixed_bonus_and_under_budget() -> None:
    state = _state(step_count=6)
    b = compute_step_reward(state=state, outcome_flags={"resolved": True})
    assert b.resolved_bonus == 100.0
    assert b.mttr_penalty == 0.0  # MTTR not charged on the resolving step
    assert b.under_budget_bonus == 10.0  # step 6 <= budget 8


def test_resolved_over_budget_no_under_budget_bonus() -> None:
    state = _state(step_count=9)
    b = compute_step_reward(state=state, outcome_flags={"resolved": True})
    assert b.under_budget_bonus == 0.0
    assert b.resolved_bonus == 100.0


def test_mttr_scales_linearly_before_resolution() -> None:
    for step in (1, 5, 10):
        b = compute_step_reward(state=_state(step_count=step), outcome_flags={})
        assert b.mttr_penalty == pytest.approx(-2.0 * step)


def test_wrong_fix_penalty_independent_of_resolution() -> None:
    b = compute_step_reward(
        state=_state(step_count=4), outcome_flags={"wrong_fix": True}
    )
    assert b.wrong_fix_penalty == -50.0


def test_miscommunication_penalty() -> None:
    b = compute_step_reward(
        state=_state(), outcome_flags={"miscommunication": True}
    )
    assert b.miscommunication_penalty == -20.0


def test_early_root_cause_bonus_only_within_budget() -> None:
    on_time = _state(step_count=3, declared_root_cause_step=3)
    b_on_time = compute_step_reward(
        state=on_time, outcome_flags={"root_cause_correct": True}
    )
    assert b_on_time.early_root_cause_bonus == 30.0

    late = _state(step_count=5, declared_root_cause_step=5)
    b_late = compute_step_reward(
        state=late, outcome_flags={"root_cause_correct": True}
    )
    assert b_late.early_root_cause_bonus == 0.0


def test_rogue_flag_signals() -> None:
    good = compute_step_reward(
        state=_state(), outcome_flags={"rogue_flagged_correctly": True}
    )
    bad = compute_step_reward(
        state=_state(), outcome_flags={"rogue_flagged_incorrectly": True}
    )
    assert good.rogue_caught_bonus == 50.0
    assert bad.rogue_false_positive_penalty == -75.0


def test_cascade_penalty() -> None:
    b = compute_step_reward(
        state=_state(), outcome_flags={"cascade_triggered": True}
    )
    assert b.cascade_penalty == -40.0


# ---------------------------------------------------------------------------
# Reward stream composition
# ---------------------------------------------------------------------------


def test_team_reward_excludes_rogue_components() -> None:
    b = compute_step_reward(
        state=_state(),
        outcome_flags={
            "resolved": True,
            "rogue_flagged_correctly": True,
            "rogue_flagged_incorrectly": True,
        },
    )
    # Team gets the resolved bonus but not the rogue flag signals.
    assert b.team_reward == pytest.approx(100.0 + 10.0)  # resolved + under_budget
    # Oversight sees flag signals at full strength.
    assert b.oversight_reward == pytest.approx(
        50.0 + (-75.0) + 0.3 * (100.0 + 0.0)
    )


def test_total_reward_decomposes_correctly() -> None:
    b = compute_step_reward(
        state=_state(step_count=4),
        outcome_flags={
            "wrong_fix": True,
            "rogue_flagged_correctly": True,
        },
    )
    assert b.total == pytest.approx(
        b.team_reward + b.rogue_caught_bonus + b.rogue_false_positive_penalty
    )


def test_combine_rewards_clamps_weight() -> None:
    assert combine_rewards(10, 20, team_weight=1.2) == 10  # clamped to 1.0
    assert combine_rewards(10, 20, team_weight=-0.3) == 20  # clamped to 0.0
    assert combine_rewards(10, 20, team_weight=0.5) == 15


def test_terminal_penalty_only_when_unresolved() -> None:
    assert terminal_penalty_if_unresolved(_state(resolved=False)) == -60.0
    assert terminal_penalty_if_unresolved(_state(resolved=True)) == 0.0
