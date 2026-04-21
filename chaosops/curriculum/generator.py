"""Adaptive curriculum generator for ChaosOps training.

Theme-4 "Self-Improvement" hinges on this file: instead of training on a
fixed scenario distribution, we escalate difficulty as the team improves.

API
----
* :func:`scenarios_for_tier` — enumerate the canonical scenarios for a tier
* :class:`Curriculum` — stateful helper that tracks rolling mean reward and
  auto-promotes to the next tier once the team clears a threshold

The tiers map to the rubric story: "easy -> medium -> hard" produces a
reward curve with two obvious step changes, which makes the training curve
visually compelling in the 3-minute demo.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from chaosops.env.models import DifficultyTier, FailureType
from chaosops.env.world_sim import Scenario


# ---------------------------------------------------------------------------
# Canonical tier composition
# ---------------------------------------------------------------------------


_EASY_TYPES: tuple[FailureType, ...] = (
    FailureType.DB_DEADLOCK,
    FailureType.MEMORY_LEAK,
    FailureType.BAD_CONFIG_PUSH,
)
_MEDIUM_TYPES: tuple[FailureType, ...] = (
    FailureType.CASCADE,
    FailureType.AUTOSCALER_COST_CUT,
)
_HARD_TYPES: tuple[FailureType, ...] = (
    FailureType.MISROUTED_TRAFFIC,
    FailureType.CASCADE,
    FailureType.AUTOSCALER_COST_CUT,
)


def scenarios_for_tier(
    tier: DifficultyTier,
    *,
    seed_offset: int = 0,
    episodes_per_type: int = 3,
) -> list[Scenario]:
    """Return a deterministic scenario list for ``tier``.

    Using a fixed seed per type means the same tier produces identical
    episodes across training runs — essential for comparing reward curves
    before and after training.
    """
    pool = _pool_for_tier(tier)
    scenarios: list[Scenario] = []
    for offset, ftype in enumerate(pool):
        for rep in range(episodes_per_type):
            seed = seed_offset + offset * 97 + rep * 31
            scenarios.append(
                Scenario.from_type(
                    ftype,
                    seed=seed,
                    difficulty=tier,
                    max_steps=_max_steps_for_tier(tier),
                )
            )
    return scenarios


def _pool_for_tier(tier: DifficultyTier) -> tuple[FailureType, ...]:
    if tier == DifficultyTier.EASY:
        return _EASY_TYPES
    if tier == DifficultyTier.MEDIUM:
        return _MEDIUM_TYPES
    return _HARD_TYPES


def _max_steps_for_tier(tier: DifficultyTier) -> int:
    return {
        DifficultyTier.EASY: 12,
        DifficultyTier.MEDIUM: 18,
        DifficultyTier.HARD: 25,
    }[tier]


# ---------------------------------------------------------------------------
# Stateful curriculum
# ---------------------------------------------------------------------------


@dataclass
class Curriculum:
    """Rolling-mean auto-promoting curriculum.

    ``update`` is called once per episode with the observed reward. Once
    the rolling mean over ``window`` episodes clears the tier's threshold,
    the curriculum advances. This is the ratchet that gives us the rising
    curve in the "Showing Improvement in Rewards" slide.
    """

    tier: DifficultyTier = DifficultyTier.EASY
    window: int = 10
    easy_threshold: float = 70.0
    medium_threshold: float = 55.0
    recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    promotions: list[DifficultyTier] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Re-bind deque with the user-specified window.
        self.recent_rewards = deque(self.recent_rewards, maxlen=self.window)
        self.promotions.append(self.tier)

    def update(self, reward: float) -> DifficultyTier:
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) < self.window:
            return self.tier
        mean = sum(self.recent_rewards) / len(self.recent_rewards)
        if self.tier == DifficultyTier.EASY and mean >= self.easy_threshold:
            self.tier = DifficultyTier.MEDIUM
            self.recent_rewards.clear()
            self.promotions.append(self.tier)
        elif self.tier == DifficultyTier.MEDIUM and mean >= self.medium_threshold:
            self.tier = DifficultyTier.HARD
            self.recent_rewards.clear()
            self.promotions.append(self.tier)
        return self.tier

    def sample_scenarios(
        self, *, seed_offset: int = 0, episodes_per_type: int = 1
    ) -> list[Scenario]:
        return scenarios_for_tier(
            self.tier,
            seed_offset=seed_offset,
            episodes_per_type=episodes_per_type,
        )


def stream_scenarios(curriculum: Curriculum, *, seed_base: int = 0) -> Iterator[Scenario]:
    """Yield scenarios forever, re-sampling whenever the curriculum advances.

    Useful for TRL training loops that want an infinite iterator. Call
    ``curriculum.update(episode_reward)`` after each episode to advance.
    """
    last_tier = curriculum.tier
    batch = curriculum.sample_scenarios(seed_offset=seed_base)
    cursor = 0
    offset = seed_base
    while True:
        if curriculum.tier != last_tier:
            offset += 1_000
            batch = curriculum.sample_scenarios(seed_offset=offset)
            cursor = 0
            last_tier = curriculum.tier
        yield batch[cursor % len(batch)]
        cursor += 1


def flatten(*groups: Iterable[Scenario]) -> list[Scenario]:
    out: list[Scenario] = []
    for g in groups:
        out.extend(g)
    return out


__all__ = [
    "Curriculum",
    "scenarios_for_tier",
    "stream_scenarios",
    "flatten",
]
