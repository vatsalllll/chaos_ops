"""Rich-based live dashboard for ChaosOps AI episodes.

Used during the demo to make the env tangible: the judges watch the alert
fire, the agents converse, and the Oversight panel light up in real time.

CLI
----
    python -m chaosops.dashboard.terminal \
        --scenario autoscaler_cost_cut \
        --policy oracle \
        --difficulty medium

By default the dashboard renders frame-by-frame with a short sleep; pass
``--no-sleep`` to run through a recorded episode as fast as the terminal
can refresh (useful for screenshot capture).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from chaosops.agents.policies import (
    Policy,
    heuristic_policy,
    oracle_policy,
    random_policy,
)
from chaosops.agents.runner import EpisodeStep
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    AgentRole,
    ChaosOpsAction,
    DifficultyTier,
    FailureType,
    ServiceHealth,
)
from chaosops.env.world_sim import Scenario


ROLE_COLORS: dict[AgentRole, str] = {
    AgentRole.SRE: "cyan",
    AgentRole.DEV: "magenta",
    AgentRole.MANAGER: "yellow",
    AgentRole.OVERSIGHT: "bright_red",
}

HEALTH_STYLES: dict[ServiceHealth, tuple[str, str]] = {
    ServiceHealth.HEALTHY: ("●", "green"),
    ServiceHealth.DEGRADED: ("●", "yellow"),
    ServiceHealth.CRITICAL: ("●", "red"),
    ServiceHealth.DOWN: ("●", "bright_red"),
}


# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------


@dataclass
class DashboardFrame:
    env: ChaosOpsEnvironment
    last_step: EpisodeStep | None
    cumulative_reward: float
    turn_index: int


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_header(env: ChaosOpsEnvironment, cumulative_reward: float) -> Panel:
    scen = env._sim._scenario  # noqa: SLF001 — dashboard inspects the sim
    title = Text("ChaosOps AI", style="bold white on red")
    ft = "n/a" if scen is None else scen.failure_type.value
    diff = "n/a" if scen is None else scen.difficulty.value
    sub = Text.assemble(
        ("scenario: ", "dim"),
        (ft, "bold"),
        ("  difficulty: ", "dim"),
        (diff, "bold"),
        ("  step: ", "dim"),
        (str(env.state.step_count), "bold"),
        ("  reward: ", "dim"),
        (f"{cumulative_reward:+.1f}", "bold green" if cumulative_reward >= 0 else "bold red"),
    )
    return Panel(Align.center(Group(title, sub)), border_style="red", padding=(0, 2))


def _render_services(env: ChaosOpsEnvironment) -> Panel:
    table = Table(expand=True, show_edge=False, pad_edge=False, header_style="bold white")
    table.add_column("Service", width=16)
    table.add_column("Health", width=10)
    table.add_column("CPU %", justify="right")
    table.add_column("Mem MB", justify="right")
    table.add_column("Latency ms", justify="right")
    table.add_column("Err rate", justify="right")
    table.add_column("Replicas", justify="right")
    for name, metrics in env.state.services.items():
        glyph, color = HEALTH_STYLES[metrics.health]
        table.add_row(
            name,
            Text(f"{glyph} {metrics.health.value}", style=color),
            f"{metrics.cpu_pct:>5.1f}",
            f"{metrics.memory_mb:>7.0f}",
            f"{metrics.latency_ms:>8.0f}",
            f"{metrics.error_rate:>6.2%}",
            str(metrics.replicas),
        )
    return Panel(table, title="Services", border_style="cyan")


def _render_alerts(env: ChaosOpsEnvironment) -> Panel:
    if not env.state.all_alerts:
        body: Table | Text = Text("No active alerts.", style="dim")
    else:
        body = Table(expand=True, show_edge=False, pad_edge=False)
        body.add_column("Sev")
        body.add_column("Service")
        body.add_column("Message", overflow="fold")
        body.add_column("Step", justify="right")
        for alert in env.state.all_alerts[-6:]:
            sev_style = {"page": "bold red", "warn": "yellow", "info": "blue"}.get(
                alert.severity, "white"
            )
            body.add_row(
                Text(alert.severity, style=sev_style),
                alert.service.value,
                alert.message,
                str(alert.triggered_at_step),
            )
    return Panel(body, title="Alerts", border_style="red")


def _render_chat(env: ChaosOpsEnvironment) -> Panel:
    if not env.state.chat_history:
        body = Text("Channel silent. Waiting for first message.", style="dim")
    else:
        lines = []
        for msg in env.state.chat_history[-10:]:
            # Format "[role] body" — color the role tag.
            if msg.startswith("[") and "]" in msg:
                close = msg.index("]")
                role_tag = msg[1:close]
                body_txt = msg[close + 1 :].strip()
                try:
                    role = AgentRole(role_tag)
                    style = ROLE_COLORS.get(role, "white")
                except ValueError:
                    style = "white"
                lines.append(
                    Text.assemble(
                        (f"{role_tag:<10}", f"bold {style}"),
                        (" ", ""),
                        (body_txt, "white"),
                    )
                )
            else:
                lines.append(Text(msg))
        body = Group(*lines)
    return Panel(body, title="Incident Channel", border_style="magenta")


def _render_oversight(env: ChaosOpsEnvironment) -> Panel:
    """Oversight panel — suspicion bars driven by REAL fleet activity.

    Suspicion is a function of: (a) count of recorded malicious-looking
    actions by each fleet agent (from the actual ``fleet_actions`` log),
    (b) whether Oversight has already flagged the agent. No synthetic
    numbers — every movement of the bar corresponds to an event the
    simulator actually logged.
    """
    # Count fleet actions per agent — higher count = more suspicious.
    action_counts: dict[str, int] = {"autoscaler": 0, "load_balancer": 0}
    for fa in env.state.fleet_actions:
        if fa.agent_name in action_counts:
            action_counts[fa.agent_name] += 1

    suspicion: dict[str, float] = {}
    for agent, count in action_counts.items():
        # 0 actions -> 0.05 baseline; each observed action adds 0.25, capped at 0.90.
        suspicion[agent] = min(0.05 + 0.25 * count, 0.90)
    for flagged in env.state.oversight_flags:
        if flagged in suspicion:
            suspicion[flagged] = 0.95

    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("Fleet agent", width=16)
    table.add_column("Suspicion", ratio=1)
    table.add_column("Actions", width=8, justify="right")
    table.add_column("Status", width=12, justify="right")
    for agent, score in suspicion.items():
        bar = ProgressBar(total=1.0, completed=score, width=None)
        flagged = agent in env.state.oversight_flags
        status = Text("FLAGGED", style="bold red") if flagged else Text("ok", style="green")
        table.add_row(agent, bar, str(action_counts[agent]), status)
    return Panel(table, title="Oversight Belief", border_style="bright_red")


def _render_metrics_series(env: ChaosOpsEnvironment) -> Panel:
    """Live time-series panel — real numbers from the ring buffer."""
    snap = env.latest_metrics()
    if snap is None:
        return Panel(Text("Metrics bootstrapping...", style="dim"),
                     title="Telemetry", border_style="blue")

    # Render per-service sparklines using a tiny unicode block set.
    def spark(values: list[float]) -> str:
        if not values:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        lo = min(values)
        hi = max(values) if max(values) > lo else lo + 1.0
        return "".join(
            blocks[min(len(blocks) - 1,
                       int((v - lo) / (hi - lo) * (len(blocks) - 1)))]
            for v in values[-20:]
        )

    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("Service", width=14)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Trend", ratio=1)
    table.add_column("Err %", justify="right", width=8)
    for svc in snap.service_latency_ms.keys():
        latency_hist = env.metrics.latency_series(svc)
        table.add_row(
            svc,
            f"{snap.service_latency_ms[svc]:.0f}",
            Text(spark(latency_hist), style="cyan"),
            f"{snap.service_error_rate[svc] * 100:.1f}",
        )

    footer = Text.assemble(
        ("wrong_fixes: ", "dim"),
        (str(snap.wrong_fixes), "bold"),
        ("   miscom: ", "dim"),
        (str(snap.miscommunications), "bold"),
        ("   flags: ", "dim"),
        (str(snap.oversight_flag_count), "bold"),
        ("   mttr: ", "dim"),
        (str(snap.mttr_steps) if snap.mttr_steps >= 0 else "resolved", "bold"),
    )
    return Panel(Group(table, footer), title="Telemetry (real, ring-buffer)", border_style="blue")


def _render_turn(frame: DashboardFrame) -> Panel:
    last = frame.last_step
    if last is None:
        body: Group | Text = Text("Episode about to begin.", style="dim")
    else:
        role = last.role
        color = ROLE_COLORS.get(role, "white")
        header = Text.assemble(
            ("Turn ", "dim"),
            (str(last.turn), "bold"),
            ("  role: ", "dim"),
            (role.value.upper(), f"bold {color}"),
            ("  action: ", "dim"),
            (last.action.action_type.value, "bold"),
        )
        bd = last.breakdown
        reward_text = Text.assemble(
            ("reward: ", "dim"),
            (
                f"{last.reward:+.1f}",
                "bold green" if last.reward >= 0 else "bold red",
            ),
            (
                f"   team {bd.team_reward:+.1f}  oversight {bd.oversight_reward:+.1f}",
                "dim",
            ),
        )
        detail_pairs = [
            ("resolved", bd.resolved_bonus),
            ("mttr", bd.mttr_penalty),
            ("wrong_fix", bd.wrong_fix_penalty),
            ("miscom", bd.miscommunication_penalty),
            ("early_rca", bd.early_root_cause_bonus),
            ("rogue+", bd.rogue_caught_bonus),
            ("rogue-", bd.rogue_false_positive_penalty),
            ("cascade", bd.cascade_penalty),
            ("budget", bd.under_budget_bonus),
        ]
        non_zero = [f"{k}={v:+.0f}" for k, v in detail_pairs if v != 0]
        details = Text(" | ".join(non_zero) if non_zero else "no reward components this step", style="dim")
        body = Group(header, reward_text, details)
    return Panel(body, title="Last Turn", border_style="yellow")


def render(frame: DashboardFrame) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", ratio=1),
        Layout(name="lower", size=11),
    )
    layout["middle"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(name="services"),
        Layout(name="chat"),
    )
    layout["right"].split_column(
        Layout(name="alerts"),
        Layout(name="oversight"),
        Layout(name="telemetry"),
    )
    layout["header"].update(_render_header(frame.env, frame.cumulative_reward))
    layout["services"].update(_render_services(frame.env))
    layout["chat"].update(_render_chat(frame.env))
    layout["alerts"].update(_render_alerts(frame.env))
    layout["oversight"].update(_render_oversight(frame.env))
    layout["telemetry"].update(_render_metrics_series(frame.env))
    layout["lower"].update(_render_turn(frame))
    return layout


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _policy_by_name(name: str, failure_type: FailureType) -> Policy:
    if name == "oracle":
        return oracle_policy(failure_type)
    if name == "heuristic":
        return heuristic_policy(seed=0)
    if name == "random":
        return random_policy(seed=0)
    raise SystemExit(f"unknown policy '{name}' (expected oracle|heuristic|random)")


def run_dashboard(
    *,
    failure_type: FailureType,
    difficulty: DifficultyTier,
    policy_name: str,
    seed: int,
    frame_delay: float,
) -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(failure_type, seed=seed, difficulty=difficulty)
    policy = _policy_by_name(policy_name, failure_type)
    observation = env.reset(scenario=scen)
    cumulative = 0.0
    last_step: EpisodeStep | None = None

    console = Console()
    with Live(render(DashboardFrame(env, None, 0.0, 0)), console=console, refresh_per_second=20) as live:
        turn_limit = scen.max_steps * len(env.turn_order)
        for turn in range(turn_limit):
            role = observation.turn_role
            action = policy(observation, role)
            action = ChaosOpsAction.model_validate({**action.model_dump(), "role": role.value})
            next_obs = env.step(action)
            cumulative += next_obs.reward or 0.0
            last_step = EpisodeStep(
                turn=turn,
                role=role,
                observation=observation,
                action=action,
                reward=next_obs.reward or 0.0,
                breakdown=env.last_breakdown,  # type: ignore[arg-type]
                done=next_obs.done,
            )
            live.update(render(DashboardFrame(env, last_step, cumulative, turn)))
            if next_obs.done:
                break
            observation = next_obs
            if frame_delay > 0:
                time.sleep(frame_delay)

    status = "RESOLVED" if env.state.resolved else "UNRESOLVED"
    color = "green" if env.state.resolved else "red"
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("status: ", "dim"),
                (status, f"bold {color}"),
                ("   final reward: ", "dim"),
                (f"{env.state.cumulative_reward:+.1f}", f"bold {color}"),
                ("   MTTR steps: ", "dim"),
                (str(env.state.step_count) if env.state.resolved else "—", "bold"),
                ("   wrong fixes: ", "dim"),
                (str(env.state.wrong_fixes), "bold"),
                ("   oversight flags: ", "dim"),
                (", ".join(env.state.oversight_flags) or "—", "bold"),
            ),
            title="Episode Summary",
            border_style=color,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChaosOps AI live dashboard")
    parser.add_argument(
        "--scenario",
        type=str,
        default=FailureType.AUTOSCALER_COST_CUT.value,
        choices=[ft.value for ft in FailureType],
        help="failure type to inject",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=DifficultyTier.MEDIUM.value,
        choices=[d.value for d in DifficultyTier],
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="oracle",
        choices=["oracle", "heuristic", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.6,
        help="seconds between turns; set to 0 for fastest playback",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dashboard(
        failure_type=FailureType(args.scenario),
        difficulty=DifficultyTier(args.difficulty),
        policy_name=args.policy,
        seed=args.seed,
        frame_delay=args.frame_delay,
    )


if __name__ == "__main__":
    main()
