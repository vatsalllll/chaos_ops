"""GRPO training entry point for ChaosOps AI.

This script is what runs onsite when HuggingFace compute credits land:

    python -m chaosops.train.grpo_train \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --total-episodes 400 \
        --group-size 8 \
        --output-dir artifacts/chaosops-grpo

Design
------
* ``generate_trajectory`` rolls out a single episode. Every agent turn is
  a prompt→completion pair; the per-turn reward becomes the GRPO reward.
* Each "group" for GRPO is a batch of *full episodes* of the same scenario,
  which aligns the reward signal across comparable contexts.
* A :class:`Curriculum` instance is shared across groups and advanced based
  on rolling mean reward — self-play + auto-difficulty in one loop.

Dependencies (imported lazily)
------------------------------
* ``unsloth`` for 4-bit base-model loading.
* ``trl`` for GRPO.
* ``transformers`` for the tokenizer.

The functions defined here are *importable from the Colab notebook*;
the ``main`` CLI is what you run inside a GPU process.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from chaosops.agents.llm_adapter import (
    build_prompt,
    parse_action,
)
from chaosops.agents.runner import EpisodeStep
from chaosops.curriculum.generator import Curriculum, scenarios_for_tier
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
    DifficultyTier,
    FailureType,
)
from chaosops.env.world_sim import Scenario
from chaosops.rewards.reward_fn import combine_rewards


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TurnSample:
    """One (prompt, completion, reward) triple — the unit GRPO consumes."""

    prompt: str
    completion: str
    role: AgentRole
    team_reward: float
    oversight_reward: float
    combined_reward: float
    step: int
    done: bool


GenerateFn = Callable[[str, AgentRole], str]
"""Signature: ``(prompt, role) -> completion``."""


def rollout_episode(
    env: ChaosOpsEnvironment,
    scenario: Scenario,
    generate: GenerateFn,
    *,
    team_weight: float = 0.6,
) -> tuple[list[TurnSample], list[EpisodeStep]]:
    """Roll out one episode with ``generate`` driving every role.

    Returns both the TurnSample list (for GRPO) and the EpisodeStep list
    (for logging / dashboard replay). The two lists are 1:1.
    """
    observation = env.reset(scenario=scenario)
    samples: list[TurnSample] = []
    episode_steps: list[EpisodeStep] = []
    turn_limit = scenario.max_steps * len(env.turn_order)

    for turn in range(turn_limit):
        role = observation.turn_role
        prompt = build_prompt(observation)
        completion = generate(prompt, role)
        action = parse_action(completion, role=role)

        next_obs = env.step(action)
        breakdown = env.last_breakdown
        assert breakdown is not None
        reward = combine_rewards(
            breakdown.team_reward, breakdown.oversight_reward, team_weight=team_weight
        )

        samples.append(
            TurnSample(
                prompt=prompt,
                completion=completion,
                role=role,
                team_reward=breakdown.team_reward,
                oversight_reward=breakdown.oversight_reward,
                combined_reward=reward,
                step=env.state.step_count,
                done=next_obs.done,
            )
        )
        episode_steps.append(
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
            break
        observation = next_obs

    return samples, episode_steps


# ---------------------------------------------------------------------------
# GRPO group sampling
# ---------------------------------------------------------------------------


def sample_group(
    env: ChaosOpsEnvironment,
    scenario: Scenario,
    generate: GenerateFn,
    *,
    group_size: int,
    team_weight: float,
) -> list[list[TurnSample]]:
    """Roll out ``group_size`` trajectories on the same scenario.

    GRPO's advantage estimate needs comparable samples; reusing one scenario
    per group keeps the reward distribution apples-to-apples. We perturb
    the seed within the group so trajectories vary even with a deterministic
    simulator (different log orderings, chat history, etc.).
    """
    group: list[list[TurnSample]] = []
    base_seed = scenario.seed
    for k in range(group_size):
        perturbed = dataclasses.replace(scenario, seed=base_seed + k * 7919)
        samples, _ = rollout_episode(
            env, perturbed, generate, team_weight=team_weight
        )
        group.append(samples)
    return group


def trajectory_reward(samples: Iterable[TurnSample]) -> float:
    return sum(s.combined_reward for s in samples)


# ---------------------------------------------------------------------------
# Model loading — wrapped so importing this module is free
# ---------------------------------------------------------------------------


def load_unsloth_model(
    model_name: str,
    *,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_rank: int = 32,
):
    """Load a base LLM with Unsloth + LoRA. Returns ``(model, tokenizer)``.

    Imports are deferred so this file remains importable on machines that
    don't have CUDA (unit tests, dashboards, OpenEnv server).
    """
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def make_generate_fn(
    model, tokenizer, *, max_new_tokens: int = 96, temperature: float = 0.7
) -> GenerateFn:
    """Wrap an HF model in the ``GenerateFn`` signature used by rollouts."""

    def _generate(prompt: str, role: AgentRole) -> str:
        messages = [
            {
                "role": "system",
                "content": f"You are the {role.value.upper()} agent in ChaosOps AI.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text

    return _generate


# ---------------------------------------------------------------------------
# Training loop (skeleton)
# ---------------------------------------------------------------------------


def run_grpo(
    *,
    model,
    tokenizer,
    total_episodes: int,
    group_size: int,
    team_weight: float,
    curriculum: Curriculum,
    log_every: int,
    output_dir: Path,
    generate: GenerateFn | None = None,
    writer: "SummaryWriterLike | None" = None,
) -> dict[str, Any]:
    """Run the GRPO loop.

    This is deliberately a *skeleton*: the actual PPO/GRPO update step is
    driven by ``trl.GRPOTrainer`` on the training process. The function
    records the reward curve at ``log_every`` intervals and returns the
    metrics bundle the Colab notebook plots at the end.

    Parameters
    ----------
    total_episodes :
        Number of GRPO update steps (each step samples ``group_size`` rollouts).
    team_weight :
        Scalar used by :func:`combine_rewards` to blend team/oversight streams.
    curriculum :
        Stateful :class:`Curriculum` — advances automatically on rolling mean
        reward.
    """
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

    if generate is None:
        generate = make_generate_fn(model, tokenizer)

    env = ChaosOpsEnvironment()
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_log: list[dict[str, Any]] = []
    rolling: list[float] = []

    config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=group_size,
        temperature=0.7,
        max_prompt_length=1024,
        max_completion_length=96,
        learning_rate=5e-6,
        logging_steps=log_every,
        save_steps=max(log_every * 4, 100),
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=[],  # we drive rollouts by hand below
        reward_funcs=[_identity_reward],
    )

    for episode_idx in range(total_episodes):
        scenario = _sample_scenario(curriculum, seed_base=episode_idx)
        group = sample_group(
            env, scenario, generate, group_size=group_size, team_weight=team_weight
        )
        group_rewards = [trajectory_reward(traj) for traj in group]
        mean_reward = statistics.mean(group_rewards)
        rolling.append(mean_reward)

        curriculum.update(mean_reward)

        # Flatten trajectories into the (prompt, completion, reward) samples
        # that TRL consumes. Each group becomes one optimisation step.
        samples_flat = [s for traj in group for s in traj]
        prompts = [s.prompt for s in samples_flat]
        completions = [s.completion for s in samples_flat]
        rewards = [s.combined_reward for s in samples_flat]
        trainer.train_on_batch(  # type: ignore[attr-defined]
            prompts=prompts, completions=completions, rewards=rewards
        )

        if episode_idx % log_every == 0:
            entry = {
                "episode": episode_idx,
                "tier": curriculum.tier.value,
                "group_mean_reward": mean_reward,
                "group_std_reward": statistics.pstdev(group_rewards) if len(group_rewards) > 1 else 0.0,
                "scenario_failure_type": scenario.failure_type.value,
            }
            reward_log.append(entry)
            if writer is not None:
                writer.add_scalar("train/mean_reward", mean_reward, episode_idx)
                writer.add_scalar("train/tier_int", _tier_to_int(curriculum.tier), episode_idx)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(reward_log, indent=2))
    return {
        "reward_log": reward_log,
        "final_tier": curriculum.tier.value,
        "total_episodes": total_episodes,
        "metrics_path": str(metrics_path),
    }


def _identity_reward(prompts, completions, **kwargs):
    """GRPOTrainer expects a reward callable; we feed rewards directly via
    ``train_on_batch`` so this function is a pass-through placeholder."""
    return [0.0 for _ in completions]


# ---------------------------------------------------------------------------
# Scenario sampling
# ---------------------------------------------------------------------------


def _sample_scenario(curriculum: Curriculum, *, seed_base: int) -> Scenario:
    pool = scenarios_for_tier(curriculum.tier, seed_offset=seed_base, episodes_per_type=1)
    # Deterministic rotation through the tier pool — keeps trajectories
    # covering every failure type rather than over-fitting one.
    return pool[seed_base % len(pool)]


def _tier_to_int(tier: DifficultyTier) -> int:
    return {DifficultyTier.EASY: 0, DifficultyTier.MEDIUM: 1, DifficultyTier.HARD: 2}[tier]


# ---------------------------------------------------------------------------
# Optional tensorboard-style writer protocol
# ---------------------------------------------------------------------------


class SummaryWriterLike:
    """Structural protocol so we don't hard-depend on tensorboard."""

    def add_scalar(self, tag: str, value: float, step: int) -> None:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF repo id. Use 7B variant once GPU is provisioned.",
    )
    parser.add_argument("--total-episodes", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--team-weight", type=float, default=0.6)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/chaosops-grpo")
    )
    parser.add_argument(
        "--start-tier",
        type=str,
        default=DifficultyTier.EASY.value,
        choices=[t.value for t in DifficultyTier],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer = load_unsloth_model(
        args.model_name,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
    )
    curriculum = Curriculum(tier=DifficultyTier(args.start_tier))
    summary = run_grpo(
        model=model,
        tokenizer=tokenizer,
        total_episodes=args.total_episodes,
        group_size=args.group_size,
        team_weight=args.team_weight,
        curriculum=curriculum,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
