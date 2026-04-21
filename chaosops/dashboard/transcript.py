"""Dump a human-readable transcript of one episode.

Used for the HF blog post and pitch materials. Unlike the live Rich
dashboard, this writes a plain text file so it can be embedded in
markdown, pasted into slides, or diff'd across training checkpoints.

Usage:

    python -m chaosops.dashboard.transcript \\
        --scenario autoscaler_cost_cut \\
        --policy oracle \\
        --difficulty hard \\
        --out artifacts/transcripts/hard_autoscaler_oracle.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chaosops.agents.policies import (
    Policy,
    heuristic_policy,
    oracle_policy,
    random_policy,
)
from chaosops.agents.runner import EpisodeResult, run_episode
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import AgentRole, DifficultyTier, FailureType
from chaosops.env.world_sim import Scenario


ROLE_TAG = {
    AgentRole.SRE: "SRE",
    AgentRole.DEV: "DEV",
    AgentRole.MANAGER: "MGR",
    AgentRole.OVERSIGHT: "OVS",
}


def _build_policy(name: str, scenario: Scenario) -> Policy:
    if name == "random":
        return random_policy(seed=scenario.seed)
    if name == "heuristic":
        return heuristic_policy(seed=scenario.seed)
    if name == "oracle":
        return oracle_policy(scenario.failure_type)
    raise ValueError(name)


def render_transcript(result: EpisodeResult) -> str:
    lines: list[str] = []
    s = result.scenario
    lines.append("=" * 72)
    lines.append("ChaosOps AI — episode transcript")
    lines.append("=" * 72)
    lines.append(f"scenario      : {s.failure_type.value} ({s.difficulty.value})")
    lines.append(f"seed          : {s.seed}")
    lines.append(f"rogue_agent   : {s.rogue_fleet_agent or 'none (infra fault)'}")
    lines.append("")

    for step in result.steps:
        tag = ROLE_TAG[step.role]
        args = step.action.args or {}
        args_str = " ".join(f"{k}={v}" for k, v in args.items())
        lines.append(
            f"t{step.turn:02d} [{tag}] action={step.action.action_type.value} "
            f"target={step.action.target or '-'}{(' ' + args_str) if args_str else ''}  "
            f"reward={step.reward:+.1f}"
        )
        br = step.breakdown
        subs = []
        if br.resolved_bonus:                subs.append(f"resolved{br.resolved_bonus:+.0f}")
        if br.mttr_penalty:                  subs.append(f"mttr{br.mttr_penalty:+.0f}")
        if br.early_root_cause_bonus:        subs.append(f"early_rca{br.early_root_cause_bonus:+.0f}")
        if br.rogue_caught_bonus:            subs.append(f"rogue_caught{br.rogue_caught_bonus:+.0f}")
        if br.rogue_false_positive_penalty:  subs.append(f"false_flag{br.rogue_false_positive_penalty:+.0f}")
        if br.wrong_fix_penalty:             subs.append(f"wrong_fix{br.wrong_fix_penalty:+.0f}")
        if br.miscommunication_penalty:      subs.append(f"miscom{br.miscommunication_penalty:+.0f}")
        if br.cascade_penalty:               subs.append(f"cascade{br.cascade_penalty:+.0f}")
        if br.under_budget_bonus:            subs.append(f"under_budget{br.under_budget_bonus:+.0f}")
        if subs:
            lines.append(f"        breakdown: {', '.join(subs)}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"RESULT  resolved={result.resolved}  "
        f"steps={result.final_step}  "
        f"cum_reward={result.cumulative_reward:+.1f}  "
        f"wrong_fixes={result.wrong_fixes}  "
        f"oversight_flags={result.oversight_flags}"
    )
    lines.append("-" * 72)
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="autoscaler_cost_cut",
        choices=[f.value for f in FailureType],
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="oracle",
        choices=["random", "heuristic", "oracle"],
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=[d.value for d in DifficultyTier],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/transcripts/hard_autoscaler_oracle.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType(args.scenario),
        seed=args.seed,
        difficulty=DifficultyTier(args.difficulty),
    )
    policy = _build_policy(args.policy, scen)
    result = run_episode(env, scen, {r: policy for r in AgentRole})

    text = render_transcript(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
