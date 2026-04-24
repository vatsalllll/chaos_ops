"""ChaosOps AI — Hugging Face Space entry point.

Gradio UI that lets a judge replay any incident scenario with any policy
(random / heuristic / oracle / trained) and watch the multi-agent response
unfold step-by-step. The trained-policy lane activates when the environment
variable ``CHAOSOPS_ADAPTER_PATH`` points at a LoRA adapter directory —
otherwise the Space still runs, silently falling back to the heuristic so
the UI works during cold-start or when no checkpoint has been uploaded yet.

Deploy layout:
    hf_space/
        app.py            — this file (entry point HF Spaces picks up)
        requirements.txt  — pulls chaosops from GitHub + Gradio + torch stack
        README.md         — HF Space card (YAML frontmatter)
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from chaosops.agents.policies import (
    Policy,
    heuristic_policy,
    oracle_policy,
    random_policy,
)
from chaosops.agents.runner import EpisodeResult, run_episode
from chaosops.dashboard.transcript import ROLE_TAG, render_transcript
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import AgentRole, DifficultyTier, FailureType
from chaosops.env.world_sim import Scenario


ADAPTER_ENV = "CHAOSOPS_ADAPTER_PATH"
_TRAINED_POLICY_CACHE = None


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


def _lazy_trained_policy():
    """Load the trained LoRA adapter once per process, lazily.

    Kept behind an env var so the Space still starts (and the UI still works)
    on cold boot before the adapter is uploaded.
    """
    global _TRAINED_POLICY_CACHE
    if _TRAINED_POLICY_CACHE is not None:
        return _TRAINED_POLICY_CACHE
    adapter_path = os.environ.get(ADAPTER_ENV)
    if not adapter_path or not Path(adapter_path).exists():
        return None
    from chaosops.agents.trained_policy import TrainedPolicy

    _TRAINED_POLICY_CACHE = TrainedPolicy.from_adapter(adapter_path)
    return _TRAINED_POLICY_CACHE


def _build_policy(name: str, scenario: Scenario) -> Policy:
    if name == "random":
        return random_policy(seed=scenario.seed)
    if name == "heuristic":
        return heuristic_policy(seed=scenario.seed)
    if name == "oracle":
        return oracle_policy(scenario.failure_type)
    if name == "trained":
        trained = _lazy_trained_policy()
        if trained is None:
            # Graceful fallback — Space is still useful before adapter lands.
            return heuristic_policy(seed=scenario.seed)
        return trained.as_policy()
    raise ValueError(f"unknown policy '{name}'")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


_ROLE_COLOR: dict[str, str] = {
    "SRE": "#2980b9",
    "DEV": "#16a085",
    "MGR": "#8e44ad",
    "OVS": "#c0392b",
}


def _render_chat_html(result: EpisodeResult) -> str:
    """Render the episode as a coloured chat log for the Gradio HTML widget."""
    blocks: list[str] = []
    for step in result.steps:
        tag = ROLE_TAG[step.role]
        color = _ROLE_COLOR.get(tag, "#333")
        args = step.action.args or {}
        args_str = " ".join(f"{k}={v}" for k, v in args.items())
        target = step.action.target or "-"
        summary = (
            f"{step.action.action_type.value} target={target}"
            + (f" {args_str}" if args_str else "")
        )
        blocks.append(
            f'<div style="margin-bottom:6px;">'
            f'<span style="color:{color};font-weight:600;">t{step.turn:02d} [{tag}]</span> '
            f'<span style="font-family:monospace;">{html.escape(summary)}</span> '
            f'<span style="color:#888;">reward={step.reward:+.1f}</span>'
            f"</div>"
        )
    footer = (
        f'<hr style="margin:10px 0;">'
        f'<div><b>resolved:</b> {result.resolved} · '
        f'<b>steps:</b> {result.final_step} · '
        f'<b>cum_reward:</b> {result.cumulative_reward:+.1f} · '
        f'<b>wrong_fixes:</b> {result.wrong_fixes} · '
        f'<b>oversight_flags:</b> {result.oversight_flags or "[]"}</div>'
    )
    return '<div style="font-size:13px;line-height:1.5;">' + "".join(blocks) + footer + "</div>"


# ---------------------------------------------------------------------------
# Episode runner (called from the Gradio button)
# ---------------------------------------------------------------------------


def run_scenario(failure: str, difficulty: str, policy_name: str, seed: int):
    scenario = Scenario.from_type(
        FailureType(failure),
        seed=int(seed),
        difficulty=DifficultyTier(difficulty),
    )
    policy = _build_policy(policy_name, scenario)
    env = ChaosOpsEnvironment()
    result = run_episode(env, scenario, {r: policy for r in AgentRole})

    chat_html = _render_chat_html(result)
    transcript = render_transcript(result)

    summary = {
        "failure_type": failure,
        "difficulty": difficulty,
        "policy": policy_name,
        "seed": int(seed),
        "resolved": result.resolved,
        "steps_to_resolve": result.final_step if result.resolved else None,
        "cumulative_reward": round(result.cumulative_reward, 2),
        "wrong_fixes": result.wrong_fixes,
        "oversight_flags": result.oversight_flags,
    }
    return chat_html, summary, transcript


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


INTRO_MARKDOWN = """
# ChaosOps AI — Multi-Agent Incident-Response Gym

A reinforcement-learning environment where a **four-agent fleet**
(SRE · Dev · Manager · **Oversight**) resolves a randomly injected
infrastructure incident. The fourth agent is a **scalable-oversight model**
whose job is to detect when *another AI in the fleet* (autoscaler,
load_balancer, deploy_bot) caused the incident — before the remediation
team touches the services.

**Policies**
- `random` · hard lower bound
- `heuristic` · what a decent human SRE would try
- `oracle` · cheats (knows ground truth) — upper-bound curve
- `trained` · our GRPO-tuned Qwen 2.5 0.5B LoRA checkpoint

Pick a failure type, smash **Run episode**, watch the team coordinate (or fail).
"""


def build_demo() -> gr.Blocks:
    failure_choices = [f.value for f in FailureType]
    tier_choices = [t.value for t in DifficultyTier]
    policy_choices = ["random", "heuristic", "oracle", "trained"]

    with gr.Blocks(title="ChaosOps AI") as demo:
        gr.Markdown(INTRO_MARKDOWN)

        with gr.Row():
            with gr.Column(scale=1):
                failure = gr.Dropdown(
                    failure_choices,
                    value="rogue_deploy_bot",
                    label="Failure type",
                )
                difficulty = gr.Dropdown(
                    tier_choices,
                    value="hard",
                    label="Difficulty",
                )
                policy = gr.Dropdown(
                    policy_choices,
                    value="oracle",
                    label="Policy",
                )
                seed = gr.Number(value=42, precision=0, label="Seed")
                run_btn = gr.Button("▶ Run episode", variant="primary")
                gr.Markdown(
                    "_Trained policy requires `CHAOSOPS_ADAPTER_PATH` to be "
                    "set on the Space. It falls back to the heuristic otherwise._"
                )
            with gr.Column(scale=2):
                chat_out = gr.HTML(label="Episode chat")
                summary_out = gr.JSON(label="Summary")
        transcript_out = gr.Textbox(
            label="Full transcript (reward breakdown)",
            lines=18,
        )

        run_btn.click(
            run_scenario,
            inputs=[failure, difficulty, policy, seed],
            outputs=[chat_out, summary_out, transcript_out],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
