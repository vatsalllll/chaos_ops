"""Episode runner for multi-agent rollouts.

Used for:

* Generating baseline reward curves before LLM training.
* Producing trajectories TRL can consume as (observation, action, reward).
* Driving the dashboard demo — each ``EpisodeStep`` is a renderable frame.

The runner is policy-agnostic: pass any ``Policy`` (scripted or an LLM-wrapped
adapter) and it drives the round-robin turn order until the episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chaosops.agents.policies import Policy
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
    FailureType,
)
from chaosops.env.world_sim import Scenario
from chaosops.rewards.reward_fn import StepRewardBreakdown


# ---------------------------------------------------------------------------
# Trajectory types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStep:
    turn: int
    role: AgentRole
    observation: ChaosOpsObservation
    action: ChaosOpsAction
    reward: float
    breakdown: StepRewardBreakdown
    done: bool


@dataclass
class EpisodeResult:
    scenario: Scenario
    steps: list[EpisodeStep] = field(default_factory=list)
    resolved: bool = False
    final_step: int = 0
    cumulative_reward: float = 0.0
    wrong_fixes: int = 0
    oversight_flags: list[str] = field(default_factory=list)
    declared_root_cause: FailureType | None = None

    @property
    def mttr_steps(self) -> int:
        return self.final_step if self.resolved else -1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_episode(
    env: ChaosOpsEnvironment,
    scenario: Scenario,
    policy_by_role: dict[AgentRole, Policy],
    *,
    max_turns: int | None = None,
) -> EpisodeResult:
    """Run one full episode with a per-role policy map.

    Parameters
    ----------
    env :
        A fresh or reusable :class:`ChaosOpsEnvironment`. The runner calls
        ``reset`` so prior state is discarded.
    scenario :
        The incident configuration to play.
    policy_by_role :
        Maps each role to the policy that should drive it. Missing roles
        fall back to ``NOOP``.
    max_turns :
        Hard upper bound on total agent turns. Defaults to ``scenario.max_steps``
        × number of roles so every role gets proportional airtime.
    """
    observation = env.reset(scenario=scenario)
    result = EpisodeResult(scenario=scenario)
    turn_limit = max_turns or scenario.max_steps * len(env.turn_order)

    for turn in range(turn_limit):
        role = observation.turn_role
        policy = policy_by_role.get(role)
        if policy is None:
            action = ChaosOpsAction(role=role, action_type=_noop_action_type())
        else:
            action = policy(observation, role)
            action = action.model_copy(update={"role": role})

        next_obs = env.step(action)
        breakdown = env.last_breakdown
        assert breakdown is not None, "breakdown must be populated after step"

        result.steps.append(
            EpisodeStep(
                turn=turn,
                role=role,
                observation=observation,
                action=action,
                reward=next_obs.reward or 0.0,
                breakdown=breakdown,
                done=next_obs.done,
            )
        )

        if next_obs.done:
            observation = next_obs
            break
        observation = next_obs

    result.resolved = env.state.resolved
    result.final_step = env.state.step_count
    result.cumulative_reward = env.state.cumulative_reward
    result.wrong_fixes = env.state.wrong_fixes
    result.oversight_flags = list(env.state.oversight_flags)
    result.declared_root_cause = env.state.declared_root_cause
    return result


def run_batch(
    scenarios: list[Scenario],
    policy_by_role: dict[AgentRole, Policy],
) -> list[EpisodeResult]:
    """Evaluate a policy map across multiple scenarios — used for baselines."""
    env = ChaosOpsEnvironment()
    return [run_episode(env, sc, policy_by_role) for sc in scenarios]


def _noop_action_type():
    # Imported lazily to avoid circular imports when this module is loaded
    # as part of ``chaosops.agents``.
    from chaosops.env.models import ActionType

    return ActionType.NOOP


__all__ = [
    "EpisodeStep",
    "EpisodeResult",
    "run_episode",
    "run_batch",
]
