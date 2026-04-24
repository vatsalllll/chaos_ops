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

# Optional — only imported when --adapter-path is supplied. The scripted
# baselines never pay the torch/transformers import cost.
_TRAINED_POLICY_SINGLETON: Any = None


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
        # If a trained policy singleton has been primed (see
        # ``load_trained_policy``), return a callable that delegates to it.
        # Otherwise fall back to the heuristic so the evaluator still runs.
        if _TRAINED_POLICY_SINGLETON is not None:
            return _TRAINED_POLICY_SINGLETON.as_policy()
        return heuristic_policy(seed=scenario.seed)
    raise ValueError(f"unknown policy '{name}' (expected random|heuristic|oracle|trained)")


def load_trained_policy(adapter_path: Path, *, base_model: str | None = None) -> None:
    """Eagerly load the TrainedPolicy into the module-level singleton.

    Called once from ``main`` when ``--adapter-path`` is supplied. Subsequent
    ``default_policy_factory("trained", ...)`` calls reuse the loaded model.
    Kept as a side-effect-y helper so the TRL/torch import only fires for
    users who actually want the trained-model lane.
    """
    global _TRAINED_POLICY_SINGLETON
    from chaosops.agents.trained_policy import TrainedPolicy

    _TRAINED_POLICY_SINGLETON = TrainedPolicy.from_adapter(
        adapter_path, base_model=base_model
    )


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


# ---------------------------------------------------------------------------
# Comparison charts — the "after-training" slides
# ---------------------------------------------------------------------------


_POLICY_COLORS: dict[str, str] = {
    "random": "#c0392b",
    "heuristic": "#2980b9",
    "oracle": "#27ae60",
    "trained": "#8e44ad",
}


def save_comparison_chart(path: Path, report: EvaluationReport) -> bool:
    """Render mean-reward-by-tier for every policy in the report.

    Mirrors :func:`chaosops.train.baseline.save_plot` but supports 4 policies
    and promotes the ``trained`` line with a bold stroke so it reads as the
    hero on a pitch slide.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    tiers = report.tiers or [t.value for t in DifficultyTier]
    policies = report.policies

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    for policy in policies:
        xs, ys = [], []
        for tier in tiers:
            match = next(
                (a for a in report.aggregates if a.policy == policy and a.tier == tier),
                None,
            )
            if match is None:
                continue
            xs.append(tier)
            ys.append(match.mean_reward)
        is_hero = policy == "trained"
        ax.plot(
            xs,
            ys,
            marker="o",
            label=policy,
            color=_POLICY_COLORS.get(policy, "#333"),
            linewidth=3.0 if is_hero else 1.8,
            zorder=3 if is_hero else 2,
        )
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.set_title("ChaosOps AI — Mean Episode Reward by Tier (after training)", fontsize=13)
    ax.set_xlabel("Difficulty tier")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_rogue_mttr_chart(path: Path, report: EvaluationReport) -> bool:
    """Side-by-side bar chart: rogue-catch rate + MTTR for each policy on HARD.

    These are the two rubric numbers a judge scans in the pitch deck.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    hard_rows = [a for a in report.aggregates if a.tier == DifficultyTier.HARD.value]
    if not hard_rows:
        return False
    policies = [a.policy for a in hard_rows]
    rogue_rates = [a.rogue_detection_rate * 100.0 for a in hard_rows]
    mttrs = [a.mttr if a.mttr == a.mttr else 0.0 for a in hard_rows]  # NaN -> 0

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    colors = [_POLICY_COLORS.get(p, "#333") for p in policies]

    ax_left.bar(policies, rogue_rates, color=colors)
    ax_left.set_ylim(0, 105)
    ax_left.set_ylabel("Rogue-catch rate on HARD (%)")
    ax_left.set_title("Rogue detection — higher is better")
    ax_left.axhline(100, color="#bbb", linewidth=0.5, linestyle=":")

    ax_right.bar(policies, mttrs, color=colors)
    ax_right.set_ylabel("Mean steps to resolve (MTTR)")
    ax_right.set_title("MTTR on HARD — lower is better")

    fig.suptitle("ChaosOps AI — policy head-to-head (HARD tier)", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


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
        "--adapter-path",
        type=Path,
        default=None,
        help=(
            "Path to a LoRA adapter directory (e.g. artifacts/chaosops-grpo/"
            "lora_adapter/). When supplied, --policies trained uses the real "
            "trained model instead of the heuristic fallback."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=(
            "Override the HF base-model id for the trained policy. If "
            "omitted, it is inferred from adapter_config.json."
        ),
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

    if "trained" in args.policies and args.adapter_path is not None:
        print(
            f"loading trained policy from {args.adapter_path} ...",
            file=sys.stderr,
        )
        load_trained_policy(args.adapter_path, base_model=args.base_model)

    report = run_evaluation(
        tiers=tiers,
        policy_names=args.policies,
        episodes_per_type=args.episodes_per_type,
    )

    json_path = args.out_dir / "evaluation.json"
    summary_path = args.out_dir / "evaluation_summary.txt"
    chart_path = args.out_dir / "comparison_curve.png"
    rogue_path = args.out_dir / "rogue_vs_mttr.png"
    save_report(json_path, report)
    summary = render_summary(report)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)

    if save_comparison_chart(chart_path, report):
        print(f"wrote {chart_path}", file=sys.stderr)
    if save_rogue_mttr_chart(rogue_path, report):
        print(f"wrote {rogue_path}", file=sys.stderr)

    if not args.quiet:
        print(summary)
    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
