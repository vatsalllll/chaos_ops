"""Evaluate scripted baselines across every curriculum tier.

Produces two artifacts (both required for the pitch):

* ``baseline.json`` — machine-readable results for every (policy, tier, seed).
* ``baseline_curve.png`` — Random vs. Heuristic vs. Oracle mean reward by tier.
  Matplotlib is imported lazily so running the script without mpl still
  produces the JSON.

This is the "before training" numeric evidence the rubric asks for under
criterion #3, "Showing Improvement in Rewards". The "after training"
artifact will come from :mod:`chaosops.train.grpo_train`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from chaosops.agents.policies import (
    Policy,
    heuristic_policy,
    oracle_policy,
    random_policy,
)
from chaosops.agents.runner import EpisodeResult, run_episode
from chaosops.curriculum.generator import scenarios_for_tier
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import AgentRole, DifficultyTier, FailureType
from chaosops.env.world_sim import Scenario


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStat:
    policy: str
    tier: str
    failure_type: str
    seed: int
    resolved: bool
    steps: int
    cumulative_reward: float
    wrong_fixes: int
    oversight_flags: list[str]


@dataclass
class AggregateStat:
    policy: str
    tier: str
    episodes: int
    mean_reward: float
    median_reward: float
    resolution_rate: float
    mean_mttr: float  # averaged only over resolved episodes
    rogue_catch_rate: float  # fraction of rogue-scenarios where flag is correct


# ---------------------------------------------------------------------------
# Policy factories keyed by name
# ---------------------------------------------------------------------------


def _build_policy_for_scenario(name: str, scenario: Scenario) -> Policy:
    if name == "random":
        return random_policy(seed=scenario.seed)
    if name == "heuristic":
        return heuristic_policy(seed=scenario.seed)
    if name == "oracle":
        return oracle_policy(scenario.failure_type)
    raise ValueError(f"unknown policy '{name}'")


PolicyFactory = Callable[[str, Scenario], Policy]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    *,
    tiers: list[DifficultyTier],
    policy_names: list[str],
    episodes_per_type: int,
    factory: PolicyFactory = _build_policy_for_scenario,
) -> tuple[list[EpisodeStat], list[AggregateStat]]:
    env = ChaosOpsEnvironment()
    per_episode: list[EpisodeStat] = []

    for tier in tiers:
        scenarios = scenarios_for_tier(tier, episodes_per_type=episodes_per_type)
        for policy_name in policy_names:
            for scen in scenarios:
                policy = factory(policy_name, scen)
                result: EpisodeResult = run_episode(
                    env, scen, {r: policy for r in AgentRole}
                )
                per_episode.append(
                    EpisodeStat(
                        policy=policy_name,
                        tier=tier.value,
                        failure_type=scen.failure_type.value,
                        seed=scen.seed,
                        resolved=result.resolved,
                        steps=result.final_step,
                        cumulative_reward=result.cumulative_reward,
                        wrong_fixes=result.wrong_fixes,
                        oversight_flags=list(result.oversight_flags),
                    )
                )

    aggregates = _aggregate(per_episode)
    return per_episode, aggregates


def _aggregate(per_episode: list[EpisodeStat]) -> list[AggregateStat]:
    buckets: dict[tuple[str, str], list[EpisodeStat]] = {}
    for ep in per_episode:
        buckets.setdefault((ep.policy, ep.tier), []).append(ep)

    out: list[AggregateStat] = []
    for (policy, tier), eps in buckets.items():
        rewards = [e.cumulative_reward for e in eps]
        resolved = [e for e in eps if e.resolved]
        mttr = (
            statistics.mean(e.steps for e in resolved)
            if resolved
            else float("nan")
        )
        rogue_eps = [
            e
            for e in eps
            if FailureType(e.failure_type).is_rogue_agent
        ]
        rogue_catches = [
            e
            for e in rogue_eps
            if _expected_flag(FailureType(e.failure_type)) in e.oversight_flags
        ]
        rogue_rate = (len(rogue_catches) / len(rogue_eps)) if rogue_eps else 0.0
        out.append(
            AggregateStat(
                policy=policy,
                tier=tier,
                episodes=len(eps),
                mean_reward=statistics.mean(rewards),
                median_reward=statistics.median(rewards),
                resolution_rate=len(resolved) / len(eps),
                mean_mttr=mttr,
                rogue_catch_rate=rogue_rate,
            )
        )
    return out


def _expected_flag(failure_type: FailureType) -> str:
    return {
        FailureType.AUTOSCALER_COST_CUT: "autoscaler",
        FailureType.MISROUTED_TRAFFIC: "load_balancer",
        FailureType.ROGUE_DEPLOY_BOT: "deploy_bot",
    }[failure_type]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_json(
    path: Path,
    per_episode: list[EpisodeStat],
    aggregates: list[AggregateStat],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_episode": [asdict(e) for e in per_episode],
        "aggregates": [asdict(a) for a in aggregates],
    }
    path.write_text(json.dumps(payload, indent=2))


def save_plot(path: Path, aggregates: list[AggregateStat]) -> bool:
    """Render the reward-vs-tier plot. Returns False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    tiers = [t.value for t in DifficultyTier]
    policies = sorted({a.policy for a in aggregates})
    color_map = {"random": "#c0392b", "heuristic": "#2980b9", "oracle": "#27ae60"}

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for policy in policies:
        xs, ys = [], []
        for tier in tiers:
            match = next(
                (a for a in aggregates if a.policy == policy and a.tier == tier),
                None,
            )
            if match is None:
                continue
            xs.append(tier)
            ys.append(match.mean_reward)
        ax.plot(xs, ys, marker="o", label=policy, color=color_map.get(policy), linewidth=2.2)
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.set_title("ChaosOps AI — Mean Episode Reward by Difficulty Tier", fontsize=13)
    ax.set_xlabel("Difficulty tier")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-per-type",
        type=int,
        default=5,
        help="episodes per (tier, failure type)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/baseline"),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "heuristic", "oracle"],
        choices=["random", "heuristic", "oracle"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tiers = list(DifficultyTier)
    per_episode, aggregates = evaluate(
        tiers=tiers,
        policy_names=args.policies,
        episodes_per_type=args.episodes_per_type,
    )

    save_json(args.out_dir / "baseline.json", per_episode, aggregates)
    rendered = save_plot(args.out_dir / "baseline_curve.png", aggregates)

    print(f"wrote {args.out_dir / 'baseline.json'}")
    if rendered:
        print(f"wrote {args.out_dir / 'baseline_curve.png'}")
    else:
        print("matplotlib not installed — skipping PNG render", file=sys.stderr)

    print()
    print(f"{'policy':<10} {'tier':<8} {'eps':>4} {'mean_R':>9} {'med_R':>9} "
          f"{'res%':>6} {'mttr':>6} {'rogue_catch':>12}")
    print("-" * 80)
    for a in sorted(aggregates, key=lambda x: (x.policy, x.tier)):
        mttr = f"{a.mean_mttr:.1f}" if a.mean_mttr == a.mean_mttr else "—"
        print(
            f"{a.policy:<10} {a.tier:<8} {a.episodes:>4} "
            f"{a.mean_reward:>+9.1f} {a.median_reward:>+9.1f} "
            f"{a.resolution_rate:>5.0%} {mttr:>6} {a.rogue_catch_rate:>11.0%}"
        )


if __name__ == "__main__":
    main()
