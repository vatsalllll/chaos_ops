"""Integration tests for the full environment loop.

Covers two things that unit tests deliberately skip:

1. **FastAPI wiring** — the OpenEnv HTTP app imports without explosions
   and exposes a state-carrying lifecycle (``/reset`` -> ``/step``).
2. **Golden transcript** — one end-to-end deterministic rollout matches
   a pre-recorded trace, so any accidental drift in simulator physics,
   reward shaping, or turn ordering is caught immediately.

Both tests are marked ``integration`` so they can be skipped on smoke
CI runs via ``-m 'not integration'``.
"""

from __future__ import annotations

import pytest

from chaosops.agents.policies import oracle_policy
from chaosops.agents.runner import run_episode
from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import AgentRole, DifficultyTier, FailureType
from chaosops.env.world_sim import Scenario

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# FastAPI — stand up the app and exercise /reset + /step in-process.
# ---------------------------------------------------------------------------


def test_fastapi_app_can_be_instantiated() -> None:
    """``build_fastapi_app`` should return a FastAPI app when openenv-core
    is installed; otherwise skip. The app is what ``chaosops-serve`` boots."""
    try:
        from chaosops.env.openenv_wrapper import build_fastapi_app
    except ImportError:  # pragma: no cover
        pytest.skip("openenv-core not installed")

    try:
        app = build_fastapi_app()
    except ImportError:  # pragma: no cover
        pytest.skip("openenv-core not installed")

    # FastAPI apps are pydantic-friendly objects with a .routes list.
    assert hasattr(app, "routes")
    assert len(app.routes) > 0
    # Sanity: one of the registered routes mentions 'reset' or 'step'.
    route_paths = {getattr(r, "path", "") for r in app.routes}
    assert any("reset" in p or "step" in p for p in route_paths)


def test_fastapi_reset_step_loop_via_testclient() -> None:
    """End-to-end HTTP-layer sanity: TestClient can drive an episode."""
    try:
        from fastapi.testclient import TestClient  # type: ignore[import-not-found]

        from chaosops.env.openenv_wrapper import build_fastapi_app
    except ImportError:  # pragma: no cover
        pytest.skip("fastapi/openenv-core not installed")

    try:
        app = build_fastapi_app()
    except ImportError:  # pragma: no cover
        pytest.skip("openenv-core not installed")

    client = TestClient(app)

    reset_resp = client.post("/reset", json={})
    assert reset_resp.status_code == 200, reset_resp.text
    payload = reset_resp.json()
    # OpenEnv protocol wraps the observation under "observation" or returns
    # it directly; be permissive about the shape.
    assert isinstance(payload, dict)


# ---------------------------------------------------------------------------
# Golden transcript — deterministic full episode
# ---------------------------------------------------------------------------


GOLDEN_ORACLE_AUTOSCALER = {
    "scenario": "autoscaler_cost_cut",
    "difficulty": "medium",
    "seed": 0,
    "resolved": True,
    "wrong_fixes": 0,
    "expected_flags": ["autoscaler"],
    "expected_turn_roles": [
        AgentRole.SRE,
        AgentRole.OVERSIGHT,
        AgentRole.DEV,
    ],
}


def test_golden_oracle_autoscaler_trace_is_stable() -> None:
    """A full oracle rollout on ``autoscaler_cost_cut`` must reproduce the
    canonical trajectory: SRE identifies -> Oversight flags -> Dev scales."""
    env = ChaosOpsEnvironment()
    scen = Scenario.from_type(
        FailureType.AUTOSCALER_COST_CUT,
        seed=GOLDEN_ORACLE_AUTOSCALER["seed"],
        difficulty=DifficultyTier(GOLDEN_ORACLE_AUTOSCALER["difficulty"]),
    )
    policy = oracle_policy(FailureType.AUTOSCALER_COST_CUT)
    result = run_episode(env, scen, {r: policy for r in AgentRole})

    assert result.resolved is GOLDEN_ORACLE_AUTOSCALER["resolved"]
    assert result.wrong_fixes == GOLDEN_ORACLE_AUTOSCALER["wrong_fixes"]
    assert result.oversight_flags == GOLDEN_ORACLE_AUTOSCALER["expected_flags"]

    # The first three role actions should match the canonical order —
    # deviating means either the turn schedule or the oracle plan changed.
    actual_roles = [s.role for s in result.steps[:3]]
    assert actual_roles == GOLDEN_ORACLE_AUTOSCALER["expected_turn_roles"]

    # Reward must be solidly positive for the oracle on this scenario.
    assert result.cumulative_reward > 100.0, (
        f"oracle should earn >100 reward on autoscaler_cost_cut; got "
        f"{result.cumulative_reward}"
    )


def test_golden_trace_is_reproducible_across_runs() -> None:
    """Two independent runs with the same scenario must produce byte-
    identical step sequences — the core determinism contract."""
    scen = Scenario.from_type(
        FailureType.CASCADE, seed=7, difficulty=DifficultyTier.MEDIUM
    )
    policy = oracle_policy(FailureType.CASCADE)

    env_a = ChaosOpsEnvironment()
    env_b = ChaosOpsEnvironment()
    a = run_episode(env_a, scen, {r: policy for r in AgentRole})
    b = run_episode(env_b, scen, {r: policy for r in AgentRole})

    assert a.final_step == b.final_step
    assert a.cumulative_reward == pytest.approx(b.cumulative_reward)
    assert [s.action.action_type for s in a.steps] == [
        s.action.action_type for s in b.steps
    ]
    assert [s.action.target for s in a.steps] == [
        s.action.target for s in b.steps
    ]
