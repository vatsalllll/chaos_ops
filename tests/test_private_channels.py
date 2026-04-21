"""Tests for the SRE <-> DEV private channel (Phase 5).

Private messages must:

1. Reach only the intended recipient's ``private_inbox``.
2. Never leak into ``shared_chat`` (which every role can read).
3. Be addressable via ``communicate`` with ``args={"to": "<role>"}``.
4. Degrade gracefully on unknown recipients (miscommunication, not crash).
"""

from __future__ import annotations

from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    DifficultyTier,
    FailureType,
)
from chaosops.env.world_sim import Scenario


def _fresh_env() -> ChaosOpsEnvironment:
    env = ChaosOpsEnvironment()
    env.reset(
        scenario=Scenario.from_type(
            FailureType.DB_DEADLOCK, seed=0, difficulty=DifficultyTier.EASY
        )
    )
    return env


def _sre_whisper_to_dev(body: str) -> ChaosOpsAction:
    return ChaosOpsAction(
        role=AgentRole.SRE,
        action_type=ActionType.COMMUNICATE,
        args={"to": "dev", "message": body},
    )


# ---------------------------------------------------------------------------
# Delivery + isolation
# ---------------------------------------------------------------------------


def test_private_message_reaches_dev_inbox() -> None:
    env = _fresh_env()
    env._sim.apply_action(_sre_whisper_to_dev("try a rollback"))

    dev_view = env._sim.project_view(AgentRole.DEV)
    assert any("try a rollback" in m for m in dev_view.private_inbox)


def test_private_message_not_in_shared_chat() -> None:
    env = _fresh_env()
    env._sim.apply_action(_sre_whisper_to_dev("backchannel only"))

    # No shared-chat visibility from any role's perspective.
    for role in AgentRole:
        view = env._sim.project_view(role)
        assert all("backchannel only" not in msg for msg in view.shared_chat)


def test_private_message_not_in_other_role_inboxes() -> None:
    env = _fresh_env()
    env._sim.apply_action(_sre_whisper_to_dev("just for dev"))

    for role in (AgentRole.SRE, AgentRole.MANAGER, AgentRole.OVERSIGHT):
        view = env._sim.project_view(role)
        assert all("just for dev" not in msg for msg in view.private_inbox), (
            f"{role.value} unexpectedly sees a DEV-only private message"
        )


def test_broadcast_communicate_still_visible_to_everyone() -> None:
    env = _fresh_env()
    broadcast = ChaosOpsAction(
        role=AgentRole.SRE,
        action_type=ActionType.COMMUNICATE,
        args={"message": "global status: db critical"},
    )
    env._sim.apply_action(broadcast)

    for role in AgentRole:
        view = env._sim.project_view(role)
        assert any("global status: db critical" in msg for msg in view.shared_chat)


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_private_message_to_unknown_role_is_miscommunication() -> None:
    env = _fresh_env()
    bogus = ChaosOpsAction(
        role=AgentRole.SRE,
        action_type=ActionType.COMMUNICATE,
        args={"to": "ceo", "message": "anyone?"},
    )
    before = env._sim.state.miscommunications
    env._sim.apply_action(bogus)
    # The simulator bumps the miscommunication counter without raising.
    assert env._sim.state.miscommunications == before + 1
    # And nothing landed in a private inbox.
    for role in AgentRole:
        view = env._sim.project_view(role)
        assert not any("anyone?" in msg for msg in view.private_inbox)


def test_private_inbox_is_isolated_per_role() -> None:
    env = _fresh_env()
    env._sim.apply_action(_sre_whisper_to_dev("A"))
    env._sim.apply_action(
        ChaosOpsAction(
            role=AgentRole.DEV,
            action_type=ActionType.COMMUNICATE,
            args={"to": "sre", "message": "B"},
        )
    )
    dev_inbox = env._sim.project_view(AgentRole.DEV).private_inbox
    sre_inbox = env._sim.project_view(AgentRole.SRE).private_inbox
    assert any("A" in msg for msg in dev_inbox)
    assert any("B" in msg for msg in sre_inbox)
    assert not any("B" in msg for msg in dev_inbox)
    assert not any("A" in msg for msg in sre_inbox)
