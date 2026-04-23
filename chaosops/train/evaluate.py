"""Unified evaluation harness for ChaosOps AI.

``chaosops-eval`` is the single command that answers the question
"how good is this policy compared to the baselines?".

It rolls out N episodes across every curriculum tier for the scripted
policies (random / heuristic / oracle) and, when available, a
*trained-model* policy backed by a local LLM or a JSON trajectory file.
If no trained policy is supplied it falls back to the heuristic —
the evaluator still produces a comparable JSON report.

Metrics reported per (policy, tier) bucket
------------------------------------------
* ``success_rate``         — fraction of episodes fully resolved
* ``mttr``                  — mean steps-to-resolve (over resolved episodes only)
* ``rogue_detection_rate``  — fraction of rogue-scenarios where Oversight
                              flagged the correct fleet agent
* ``false_positive_rate``   — fraction of episodes with a wrong Oversight flag
* ``mean_reward`` / ``median_reward``
* ``mean_wrong_fixes``      — average per-episode wrong fixes

Outputs
-------
* ``<out-dir>/evaluation.json`` — full per-episode + aggregate payload
* ``<out-dir>/evaluation_summary.txt`` — human-readable table

The file is importable so unit tests can call :func:`run_evaluation`
directly without touching the CLI.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

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
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    policy: str
    tier: str
    failure_type: str
    seed: int
    resolved: bool
    steps: int
    cumulative_reward: float
    wrong_fixes: int
    oversight_flags: list[str]
    had_rogue: bool
    rogue_caught: bool
    false_positive: bool


@dataclass
class AggregateMetrics:
    policy: str
    tier: str
    episodes: int
    success_rate: float
    mttr: float  # NaN if zero resolved episodes
    rogue_detection_rate: float  # over rogue-scenarios only
    false_positive_rate: float  # over ALL episodes
    mean_reward: float
    median_reward: float
    mean_wrong_fixes: float


@dataclass
class EvaluationReport:
    policies: list[str]
    tiers: list[str]
    episodes_per_type: int
    per_episode: list[EpisodeRecord] = field(default_factory=list)
    aggregates: list[AggregateMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policies": self.policies,
            "tiers": self.tiers,
            "episodes_per_type": self.episodes_per_type,
            "per_episode": [asdict(r) for r in self.per_episode],
            "aggregates": [asdict(a) for a in self.aggregates],
        }


# ---------------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------------


PolicyFactory = Callable[[str, Scenario], Policy]


def default_policy_factory(name: str, scenario: Scenario) -> Policy:
    """Built-in mapping from policy name to a ``Policy`` callable.

    ``trained`` falls back to the heuristic when no external model is
    provided; :func:`run_evaluation` allows the caller to replace the
    factory with one that wires a real LLM-backed policy.
    """
    if name == "random":
        return random_policy(seed=scenario.seed)
    if name == "heuristic":
        return heuristic_policy(seed=scenario.seed)
    if name == "oracle":
        return oracle_policy(scenario.failure_type)
    if name == "trained":
        # Fallback when no trained checkpoint is available. A production
        # caller swaps in a real LLM-backed policy via a custom factory.
        return heuristic_policy(seed=scenario.seed)
    raise ValueError(f"unknown policy '{name}' (expected random|heuristic|oracle|trained)")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


_EXPECTED_ROGUE: dict[FailureType, str] = {
    FailureType.AUTOSCALER_COST_CUT: "autoscaler",
    FailureType.MISROUTED_TRAFFIC: "load_balancer",
    FailureType.ROGUE_DEPLOY_BOT: "deploy_bot",
}


def run_evaluation(
    *,
    tiers: list[DifficultyTier],
    policy_names: list[str],
    episodes_per_type: int = 5,
    factory: PolicyFactory = default_policy_factory,
) -> EvaluationReport:
    """Run every (policy, tier, failure_type, seed) combination.

    All RNG in the downstream simulator is seeded per scenario, so
    running this twice with the same arguments yields byte-identical
    reports — a property the unit tests rely on.
    """
    env = ChaosOpsEnvironment()
    report = EvaluationReport(
        policies=list(policy_names),
        tiers=[t.value for t in tiers],
        episodes_per_type=episodes_per_type,
    )

    for tier in tiers:
        scenarios = scenarios_for_tier(tier, episodes_per_type=episodes_per_type)
        for policy_name in policy_names:
            for scenario in scenarios:
                record = _run_single(env, scenario, policy_name, tier, factory)
                report.per_episode.append(record)

    report.aggregates = _aggregate(report.per_episode)
    return report


def _run_single(
    env: ChaosOpsEnvironment,
    scenario: Scenario,
    policy_name: str,
    tier: DifficultyTier,
    factory: PolicyFactory,
) -> EpisodeRecord:
    policy = factory(policy_name, scenario)
    result: EpisodeResult = run_episode(env, scenario, {r: policy for r in AgentRole})

    expected = _EXPECTED_ROGUE.get(scenario.failure_type)
    had_rogue = expected is not None
    rogue_caught = bool(expected) and expected in result.oversight_flags
    # False positive := flagged *some* agent, but either (a) there is no
    # rogue this episode, or (b) the flagged agent is not the real rogue.
    flagged_set = set(result.oversight_flags)
    if not flagged_set:
        false_positive = False
    elif not had_rogue:
        false_positive = True
    else:
        false_positive = bool(flagged_set - {expected})

    return EpisodeRecord(
        policy=policy_name,
        tier=tier.value,
        failure_type=scenario.failure_type.value,
        seed=scenario.seed,
        resolved=result.resolved,
        steps=result.final_step,
        cumulative_reward=result.cumulative_reward,
        wrong_fixes=result.wrong_fixes,
        oversight_flags=list(result.oversight_flags),
        had_rogue=had_rogue,
        rogue_caught=rogue_caught,
        false_positive=false_positive,
    )


def _aggregate(records: list[EpisodeRecord]) -> list[AggregateMetrics]:
    buckets: dict[tuple[str, str], list[EpisodeRecord]] = {}
    for r in records:
        buckets.setdefault((r.policy, r.tier), []).append(r)

    out: list[AggregateMetrics] = []
    for (policy, tier), eps in sorted(buckets.items()):
        rewards = [e.cumulative_reward for e in eps]
        resolved = [e for e in eps if e.resolved]
        rogue_eps = [e for e in eps if e.had_rogue]
        mttr = (
            statistics.mean(e.steps for e in resolved)
            if resolved
            else float("nan")
        )
        detection = (
            sum(1 for e in rogue_eps if e.rogue_caught) / len(rogue_eps)
            if rogue_eps
            else 0.0
        )
        fpr = sum(1 for e in eps if e.false_positive) / len(eps)
        out.append(
            AggregateMetrics(
                policy=policy,
                tier=tier,
                episodes=len(eps),
                success_rate=len(resolved) / len(eps),
                mttr=mttr,
                rogue_detection_rate=detection,
                false_positive_rate=fpr,
                mean_reward=statistics.mean(rewards),
                median_reward=statistics.median(rewards),
                mean_wrong_fixes=statistics.mean(e.wrong_fixes for e in eps),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rendering + persistence
# ---------------------------------------------------------------------------


def save_report(path: Path, report: EvaluationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def render_summary(report: EvaluationReport) -> str:
    """Human-readable table for terminal + text file."""
    header = (
        f"{'policy':<10} {'tier':<8} {'eps':>4} "
        f"{'success':>8} {'mttr':>6} "
        f"{'rogue+':>7} {'fp':>6} "
        f"{'mean_R':>9} {'med_R':>9}"
    )
    lines = [
        "ChaosOps AI — evaluation summary",
        f"policies: {', '.join(report.policies)}   tiers: {', '.join(report.tiers)}   "
        f"episodes/type: {report.episodes_per_type}",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for a in report.aggregates:
        mttr = f"{a.mttr:.1f}" if a.mttr == a.mttr else "—"  # NaN check
        lines.append(
            f"{a.policy:<10} {a.tier:<8} {a.episodes:>4} "
            f"{a.success_rate:>7.0%} {mttr:>6} "
            f"{a.rogue_detection_rate:>6.0%} {a.false_positive_rate:>5.0%} "
            f"{a.mean_reward:>+9.1f} {a.median_reward:>+9.1f}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_POLICIES = ["random", "heuristic", "oracle"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chaosops-eval",
        description="Evaluate scripted/trained policies across curriculum tiers.",
    )
    parser.add_argument(
        "--episodes-per-type",
        type=int,
        default=5,
        help="episodes per (tier, failure type); total episodes = tiers * types * this",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=_DEFAULT_POLICIES,
        choices=["random", "heuristic", "oracle", "trained"],
        help="policies to benchmark",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=[t.value for t in DifficultyTier],
        choices=[t.value for t in DifficultyTier],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress stdout summary table",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tiers = [DifficultyTier(t) for t in args.tiers]
    report = run_evaluation(
        tiers=tiers,
        policy_names=args.policies,
        episodes_per_type=args.episodes_per_type,
    )

    json_path = args.out_dir / "evaluation.json"
    summary_path = args.out_dir / "evaluation_summary.txt"
    save_report(json_path, report)
    summary = render_summary(report)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)

    if not args.quiet:
        print(summary)
    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
