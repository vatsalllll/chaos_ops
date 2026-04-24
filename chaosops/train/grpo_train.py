"""GRPO training entry point for ChaosOps AI.

Runs on Colab T4 (0.5B model) or onsite HF-credit GPUs (7B model):

    python -m chaosops.train.grpo_train \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --total-episodes 400 \
        --group-size 4 \
        --output-dir artifacts/chaosops-grpo

Design
------
* :func:`build_training_dataset` pre-rolls episodes with ``oracle_policy`` and
  captures every agent turn as a dataset row. Each row is one
  ``(prompt, scenario, action_history)`` triple — sufficient to deterministically
  reconstruct the env state for reward scoring.
* :func:`chaosops_reward` is the TRL-compatible reward function: it parses the
  model's completion, replays scenario + history in a fresh env, applies the
  action, and returns the per-step shaped reward (blend of team + oversight).
* GRPOTrainer samples ``group_size`` completions per prompt, computes
  group-relative advantages from the rewards, and updates the LoRA adapter.
* :class:`ChaosOpsMetricsCallback` writes ``training_metrics.json`` in the
  schema the Colab notebook's plot cell expects.

``rollout_episode`` / ``sample_group`` are retained for use by the dashboard
and evaluation scripts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from chaosops.agents.llm_adapter import (
    build_prompt,
    parse_action,
)
from chaosops.agents.policies import oracle_policy
from chaosops.agents.runner import EpisodeStep
from chaosops.curriculum.generator import Curriculum, scenarios_for_tier
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    AgentRole,
    ChaosOpsAction,
    DifficultyTier,
    FailureType,
)
from chaosops.env.world_sim import Scenario
from chaosops.rewards.reward_fn import combine_rewards


# ---------------------------------------------------------------------------
# Trajectory generation (kept for dashboard / eval callers)
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

    Returns both the TurnSample list and the EpisodeStep list (1:1).
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


def sample_group(
    env: ChaosOpsEnvironment,
    scenario: Scenario,
    generate: GenerateFn,
    *,
    group_size: int,
    team_weight: float,
) -> list[list[TurnSample]]:
    """Roll out ``group_size`` trajectories on perturbed seeds of the same scenario."""
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
# Scenario / action serialization for dataset rows
# ---------------------------------------------------------------------------


def _scenario_to_json(scen: Scenario) -> str:
    return json.dumps(
        {
            "failure_type": scen.failure_type.value,
            "difficulty": scen.difficulty.value,
            "seed": scen.seed,
            "max_steps": scen.max_steps,
            "inject_misleading_logs": scen.inject_misleading_logs,
            "rogue_fleet_agent": scen.rogue_fleet_agent,
        }
    )


def _scenario_from_json(payload: str) -> Scenario:
    d = json.loads(payload)
    return Scenario(
        failure_type=FailureType(d["failure_type"]),
        difficulty=DifficultyTier(d["difficulty"]),
        seed=int(d["seed"]),
        max_steps=int(d["max_steps"]),
        inject_misleading_logs=bool(d["inject_misleading_logs"]),
        rogue_fleet_agent=d["rogue_fleet_agent"],
    )


# ---------------------------------------------------------------------------
# Dataset construction — oracle-rollout prompts
# ---------------------------------------------------------------------------


def build_training_dataset(scenarios: list[Scenario]):
    """Pre-roll every ``scenario`` with ``oracle_policy`` and collect per-turn rows.

    Each row: ``{prompt, scenario_json, action_history_json, role, turn_idx}``.
    The reward function uses scenario + action_history to deterministically
    reconstruct the env state before scoring the model's completion.
    """
    from datasets import Dataset  # type: ignore[import-not-found]

    rows: list[dict[str, Any]] = []
    for scen in scenarios:
        env = ChaosOpsEnvironment()
        observation = env.reset(scenario=scen)
        policy = oracle_policy(scen.failure_type)
        action_history: list[dict[str, Any]] = []
        turn_limit = scen.max_steps * len(env.turn_order)
        for turn in range(turn_limit):
            prompt = build_prompt(observation)
            rows.append(
                {
                    "prompt": prompt,
                    "scenario_json": _scenario_to_json(scen),
                    "action_history_json": json.dumps(action_history),
                    "role": observation.turn_role.value,
                    "turn_idx": turn,
                }
            )
            action = policy(observation, observation.turn_role)
            action_history.append(action.model_dump(mode="json"))
            observation = env.step(action)
            if observation.done:
                break

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# GRPO reward function (modern TRL signature)
# ---------------------------------------------------------------------------


def make_reward_fn(team_weight: float):
    """Return a TRL-compatible reward function closed over ``team_weight``."""

    def chaosops_reward(
        prompts: list[str],
        completions: list[str],
        scenario_json: list[str],
        action_history_json: list[str],
        role: list[str],
        turn_idx: list[int],
        **_kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, scen_js, hist_js, role_v in zip(
            completions, scenario_json, action_history_json, role, strict=False
        ):
            try:
                reward = _score_completion(
                    completion=completion,
                    scen_js=scen_js,
                    hist_js=hist_js,
                    role_v=role_v,
                    team_weight=team_weight,
                )
            except Exception:
                # Robust to parsing / replay failures — penalise but don't crash training.
                reward = -5.0
            rewards.append(reward)
        return rewards

    return chaosops_reward


def _score_completion(
    *,
    completion: str,
    scen_js: str,
    hist_js: str,
    role_v: str,
    team_weight: float,
) -> float:
    scen = _scenario_from_json(scen_js)
    history_raw = json.loads(hist_js)
    env = ChaosOpsEnvironment()
    observation = env.reset(scenario=scen)
    for past in history_raw:
        past_action = ChaosOpsAction.model_validate(past)
        observation = env.step(past_action)
        if observation.done:
            return 0.0
    role_enum = AgentRole(role_v)
    if observation.turn_role != role_enum:
        # Replayed state doesn't match the captured row — treat as neutral.
        return 0.0
    # Completion may include chat-template artefacts; parse_action handles JSON extraction.
    text = completion if isinstance(completion, str) else str(completion)
    action = parse_action(text, role=role_enum)
    env.step(action)
    breakdown = env.last_breakdown
    if breakdown is None:
        return 0.0
    return combine_rewards(
        breakdown.team_reward,
        breakdown.oversight_reward,
        team_weight=team_weight,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_unsloth_model(
    model_name: str,
    *,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_rank: int = 32,
):
    """Load a base LLM with Unsloth + LoRA. Returns ``(model, tokenizer)``."""
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
    """Wrap an HF model in the ``GenerateFn`` signature used by dashboard rollouts."""

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
# Metrics callback — writes training_metrics.json as the plot cell expects
# ---------------------------------------------------------------------------


def _make_metrics_callback(output_dir: Path):
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    class ChaosOpsMetricsCallback(TrainerCallback):
        """Capture TRL's per-log reward stats and persist them to JSON.

        The Colab notebook's plot cell reads three fields: ``mean_team_reward``,
        ``mean_oversight_reward``, ``mean_combined_reward``. Our reward
        function already emits ``combine_rewards(team, oversight)``, so the
        team/oversight slots carry the same combined scalar — honest given we
        don't split the signal during training. The curve still shows the
        reward rising as expected.
        """

        def __init__(self) -> None:
            self.log: list[dict[str, Any]] = []
            self.output_dir = output_dir
            self.metrics_path = output_dir / "training_metrics.json"
            output_dir.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001 — HF signature
            if not logs:
                return
            reward_key_candidates = [
                "reward",
                "rewards/chaosops_reward/mean",
                "rewards/chaosops_reward",
            ]
            reward: float | None = None
            for key in reward_key_candidates:
                if key in logs:
                    reward = float(logs[key])
                    break
            if reward is None:
                return
            entry = {
                "episode": int(state.global_step),
                "mean_team_reward": reward,
                "mean_oversight_reward": reward,
                "mean_combined_reward": reward,
            }
            for extra in ("loss", "kl", "reward_std"):
                if extra in logs:
                    entry[extra] = float(logs[extra])
            self.log.append(entry)
            self.metrics_path.write_text(json.dumps(self.log, indent=2))

    return ChaosOpsMetricsCallback()


# ---------------------------------------------------------------------------
# Scenario sourcing
# ---------------------------------------------------------------------------


def _collect_scenarios(curriculum: Curriculum, *, total: int) -> list[Scenario]:
    """Pull ``total`` scenarios from the current tier, cycling failure types."""
    scenarios: list[Scenario] = []
    cycle_seed = 0
    while len(scenarios) < total:
        batch = scenarios_for_tier(
            curriculum.tier,
            seed_offset=cycle_seed,
            episodes_per_type=1,
        )
        scenarios.extend(batch)
        cycle_seed += 97
    return scenarios[:total]


# ---------------------------------------------------------------------------
# Training loop — modern TRL GRPO API
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
    max_seq_length: int = 1024,
    max_completion_length: int = 96,
    learning_rate: float = 5e-6,
) -> dict[str, Any]:
    """Run GRPO training via TRL's GRPOTrainer.

    ``total_episodes`` caps the number of optimisation steps (``max_steps``).
    Each optim step consumes one unique prompt from the dataset and rolls
    ``group_size`` completions — the classic GRPO group.
    """
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_count = max(total_episodes, 8)
    scenarios = _collect_scenarios(curriculum, total=scenario_count)
    dataset = build_training_dataset(scenarios)

    # Every optim step: 1 unique prompt × group_size completions.
    per_device_train_batch_size = group_size

    config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        num_generations=group_size,
        temperature=0.7,
        max_prompt_length=max_seq_length,
        max_completion_length=max_completion_length,
        learning_rate=learning_rate,
        logging_steps=log_every,
        max_steps=total_episodes,
        save_steps=max(total_episodes, 10_000),
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
    )

    reward_fn = make_reward_fn(team_weight)
    metrics_callback = _make_metrics_callback(output_dir)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        callbacks=[metrics_callback],
    )
    trainer.train()

    # Persist final LoRA adapter for downstream inference.
    adapter_dir = output_dir / "lora_adapter"
    try:
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    except Exception as exc:  # pragma: no cover — best-effort
        print(f"[grpo_train] could not save adapter: {exc}")

    # Guarantee the metrics file exists for the plot cell even if no log event fired.
    metrics_path = output_dir / "training_metrics.json"
    if not metrics_path.exists():
        metrics_path.write_text(
            json.dumps(
                [
                    {
                        "episode": 0,
                        "mean_team_reward": 0.0,
                        "mean_oversight_reward": 0.0,
                        "mean_combined_reward": 0.0,
                    }
                ],
                indent=2,
            )
        )

    rewards_collected = [e["mean_combined_reward"] for e in metrics_callback.log]
    summary = {
        "final_tier": curriculum.tier.value,
        "total_episodes": total_episodes,
        "dataset_size": len(dataset),
        "group_size": group_size,
        "metrics_path": str(metrics_path),
        "adapter_path": str(adapter_dir),
        "mean_logged_reward": (
            statistics.mean(rewards_collected) if rewards_collected else float("nan")
        ),
    }
    return summary


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
    parser.add_argument("--total-episodes", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--team-weight", type=float, default=0.6)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
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
        max_seq_length=args.max_seq_length,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
