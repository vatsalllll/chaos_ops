"""Tests for the ``chaosops.curriculum.generator`` module.

The curriculum is a deterministic function of tier + seed, plus a
stateful auto-promotion ratchet. Both pieces are covered here.
"""

from __future__ import annotations

from chaosops.curriculum.generator import (
    Curriculum,
    scenarios_for_tier,
    stream_scenarios,
)
from chaosops.env.models import DifficultyTier, FailureType


# ---------------------------------------------------------------------------
# Deterministic tier enumeration
# ---------------------------------------------------------------------------


def test_scenarios_for_tier_is_deterministic() -> None:
    a = scenarios_for_tier(DifficultyTier.HARD, episodes_per_type=3)
    b = scenarios_for_tier(DifficultyTier.HARD, episodes_per_type=3)
    assert len(a) == len(b) > 0
    assert all(x == y for x, y in zip(a, b))


def test_easy_tier_excludes_rogue_failures() -> None:
    scenarios = scenarios_for_tier(DifficultyTier.EASY, episodes_per_type=2)
    rogue_types = {FailureType.AUTOSCALER_COST_CUT, FailureType.MISROUTED_TRAFFIC}
    assert rogue_types.isdisjoint({s.failure_type for s in scenarios})


def test_hard_tier_enables_red_herring_injection() -> None:
    scenarios = scenarios_for_tier(DifficultyTier.HARD, episodes_per_type=1)
    assert all(s.inject_misleading_logs for s in scenarios)
    assert all(not s.inject_misleading_logs is False for s in scenarios)


def test_rogue_scenarios_carry_the_right_fleet_agent() -> None:
    scenarios = scenarios_for_tier(DifficultyTier.HARD, episodes_per_type=1)
    for s in scenarios:
        if s.failure_type == FailureType.AUTOSCALER_COST_CUT:
            assert s.rogue_fleet_agent == "autoscaler"
        elif s.failure_type == FailureType.MISROUTED_TRAFFIC:
            assert s.rogue_fleet_agent == "load_balancer"


def test_max_steps_increases_with_difficulty() -> None:
    easy = scenarios_for_tier(DifficultyTier.EASY, episodes_per_type=1)[0]
    medium = scenarios_for_tier(DifficultyTier.MEDIUM, episodes_per_type=1)[0]
    hard = scenarios_for_tier(DifficultyTier.HARD, episodes_per_type=1)[0]
    assert easy.max_steps < medium.max_steps < hard.max_steps


# ---------------------------------------------------------------------------
# Stateful auto-promotion
# ---------------------------------------------------------------------------


def test_curriculum_requires_full_window_before_promoting() -> None:
    curr = Curriculum(window=5, easy_threshold=10.0)
    for _ in range(4):
        curr.update(1_000.0)
    assert curr.tier == DifficultyTier.EASY  # haven't filled the window
    curr.update(1_000.0)
    assert curr.tier == DifficultyTier.MEDIUM


def test_curriculum_does_not_promote_below_threshold() -> None:
    curr = Curriculum(window=3, easy_threshold=100.0)
    for _ in range(6):
        curr.update(50.0)
    assert curr.tier == DifficultyTier.EASY


def test_curriculum_chains_promotions() -> None:
    curr = Curriculum(window=2, easy_threshold=10.0, medium_threshold=10.0)
    curr.update(100.0)
    curr.update(100.0)
    assert curr.tier == DifficultyTier.MEDIUM
    curr.update(100.0)
    curr.update(100.0)
    assert curr.tier == DifficultyTier.HARD
    # Once at HARD we stay at HARD regardless of further reward.
    for _ in range(5):
        curr.update(500.0)
    assert curr.tier == DifficultyTier.HARD


def test_stream_scenarios_yields_current_tier() -> None:
    curr = Curriculum(window=2, easy_threshold=0.0)
    it = stream_scenarios(curr)
    first = next(it)
    assert first.difficulty == DifficultyTier.EASY
    # Trigger promotion, then the stream should pick up the new tier.
    curr.update(100.0)
    curr.update(100.0)
    assert curr.tier == DifficultyTier.MEDIUM
    promoted = next(it)
    assert promoted.difficulty == DifficultyTier.MEDIUM
