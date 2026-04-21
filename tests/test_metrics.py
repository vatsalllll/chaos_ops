"""Tests for ``chaosops.env.metrics`` — the ring-buffer telemetry."""

from __future__ import annotations

from chaosops.agents.policies import oracle_policy
from chaosops.agents.runner import run_episode
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.metrics import MetricsRecorder
from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    ChaosOpsState,
    DifficultyTier,
    FailureType,
    ServiceMetrics,
)
from chaosops.env.world_sim import Scenario


def _state(**kw) -> ChaosOpsState:
    defaults = dict(
        step_count=1,
        failure_type=FailureType.DB_DEADLOCK,
        services={
            "db": ServiceMetrics(
                cpu_pct=50, memory_mb=500, latency_ms=800,
                error_rate=0.3, replicas=1
            ),
        },
    )
    defaults.update(kw)
    return ChaosOpsState(**defaults)


# ---------------------------------------------------------------------------
# MetricsRecorder unit behaviour
# ---------------------------------------------------------------------------


def test_recorder_appends_snapshot_per_step() -> None:
    rec = MetricsRecorder()
    assert rec.latest() is None

    rec.on_step(
        _state(step_count=1),
        ChaosOpsAction(role=AgentRole.DEV, action_type=ActionType.RESTART, target="db"),
    )
    assert rec.latest() is not None
    assert rec.latest().step == 1
    assert rec.action_count(ActionType.RESTART) == 1


def test_recorder_ring_buffer_respects_capacity() -> None:
    rec = MetricsRecorder(capacity=3)
    for step in range(1, 6):
        rec.on_step(
            _state(step_count=step),
            ChaosOpsAction(role=AgentRole.SRE, action_type=ActionType.NOOP),
        )
    # Only the last 3 snapshots are retained; the first two are dropped.
    snaps = rec.as_list()
    assert len(snaps) == 3
    assert [s.step for s in snaps] == [3, 4, 5]


def test_recorder_tracks_action_histogram() -> None:
    rec = MetricsRecorder()
    actions = [ActionType.RESTART, ActionType.RESTART, ActionType.COMMUNICATE]
    for step, atype in enumerate(actions, start=1):
        rec.on_step(
            _state(step_count=step),
            ChaosOpsAction(role=AgentRole.DEV, action_type=atype),
        )
    hist = rec.action_histogram()
    assert hist[ActionType.RESTART.value] == 2
    assert hist[ActionType.COMMUNICATE.value] == 1


def test_mttr_reflects_resolution() -> None:
    rec = MetricsRecorder()
    rec.on_step(
        _state(step_count=4, resolved=False),
        ChaosOpsAction(role=AgentRole.DEV, action_type=ActionType.NOOP),
    )
    assert rec.latest().mttr_steps == 4
    rec.on_step(
        _state(step_count=5, resolved=True),
        ChaosOpsAction(role=AgentRole.DEV, action_type=ActionType.NOOP),
    )
    assert rec.latest().mttr_steps == -1


def test_snapshot_flat_dict_contains_per_service_keys() -> None:
    rec = MetricsRecorder()
    snap = rec.on_step(
        _state(step_count=1),
        ChaosOpsAction(role=AgentRole.SRE, action_type=ActionType.NOOP),
    )
    flat = snap.as_flat_dict()
    assert "latency_ms.db" in flat
    assert "error_rate.db" in flat
    assert flat["step"] == 1.0


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------


def test_environment_exposes_metrics_after_reset_and_step() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType.DB_DEADLOCK, seed=0, difficulty=DifficultyTier.EASY
    )
    policy = oracle_policy(FailureType.DB_DEADLOCK)
    run_episode(env, scen, {r: policy for r in AgentRole})

    snapshots = env.metrics.as_list()
    assert snapshots, "metrics buffer should have at least one snapshot"
    # Final snapshot reflects resolution.
    assert snapshots[-1].mttr_steps == -1
    # Every snapshot covers all four baseline services.
    assert set(snapshots[0].service_latency_ms.keys()) == {
        "auth", "payments", "notifications", "db"
    }
    # Cumulative reward is monotonic or at least ends positive.
    assert snapshots[-1].cumulative_reward > 0


def test_metrics_reset_between_episodes() -> None:
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(FailureType.DB_DEADLOCK, seed=0)
    policy = oracle_policy(FailureType.DB_DEADLOCK)
    run_episode(env, scen, {r: policy for r in AgentRole})
    first_len = len(env.metrics.as_list())
    assert first_len > 0

    run_episode(env, scen, {r: policy for r in AgentRole})
    second_len = len(env.metrics.as_list())
    # Second episode's recorder was cleared on reset — same size as first.
    assert second_len == first_len
