"""Persistent time-series metrics for ChaosOps AI.

Design
------
* **Ring buffer** — bounded memory, O(1) append, suitable for the
  always-on dashboard and training runs of arbitrary length.
* **Drawn from real state** — no synthetic numbers: each snapshot is a
  direct projection of :class:`ChaosOpsState` at a given step, so the
  dashboard and training logs see the same signal.
* **Hashable-friendly** — dataclasses over dicts so downstream tooling
  (matplotlib, CSV, tensorboard) can consume them without parsing.

Tracked dimensions per step:

* per-service ``latency_ms`` and ``error_rate``
* cumulative ``wrong_fixes`` and ``miscommunications``
* ``mttr`` — steps since episode start while still unresolved, clamped
  at resolution; ``-1`` once resolved so plots can mark the finish.
* ``action_counts`` — running histogram of action types taken so far.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chaosops.env.models import ActionType, ChaosOpsAction, ChaosOpsState

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsSnapshot:
    """One point on the time-series, taken after a step."""

    step: int
    service_latency_ms: dict[str, float]
    service_error_rate: dict[str, float]
    wrong_fixes: int
    miscommunications: int
    mttr_steps: int  # -1 once resolved; otherwise == step_count
    cumulative_reward: float
    action_counts: dict[str, int]
    oversight_flag_count: int

    def as_flat_dict(self) -> dict[str, float]:
        """Flatten for CSV / tensorboard-style loggers."""
        flat: dict[str, float] = {
            "step": float(self.step),
            "wrong_fixes": float(self.wrong_fixes),
            "miscommunications": float(self.miscommunications),
            "mttr_steps": float(self.mttr_steps),
            "cumulative_reward": float(self.cumulative_reward),
            "oversight_flag_count": float(self.oversight_flag_count),
        }
        for svc, lat in self.service_latency_ms.items():
            flat[f"latency_ms.{svc}"] = lat
        for svc, err in self.service_error_rate.items():
            flat[f"error_rate.{svc}"] = err
        for atype, count in self.action_counts.items():
            flat[f"actions.{atype}"] = float(count)
        return flat


# ---------------------------------------------------------------------------
# Ring-buffer collector
# ---------------------------------------------------------------------------


@dataclass
class MetricsRecorder:
    """Ring-buffer collector over a single (or streamed) episode.

    Call :meth:`on_step` once per environment step with the post-action
    state + the action that produced it. Use :meth:`latest` / :meth:`as_list`
    to pull snapshots out for the dashboard or a JSON dump.
    """

    capacity: int = 512
    _buffer: deque[MetricsSnapshot] = field(init=False)
    _action_counter: Counter = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.capacity)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def on_step(self, state: ChaosOpsState, action: ChaosOpsAction) -> MetricsSnapshot:
        self._action_counter[action.action_type.value] += 1
        snapshot = MetricsSnapshot(
            step=state.step_count,
            service_latency_ms={
                name: round(metrics.latency_ms, 2)
                for name, metrics in state.services.items()
            },
            service_error_rate={
                name: round(metrics.error_rate, 4)
                for name, metrics in state.services.items()
            },
            wrong_fixes=state.wrong_fixes,
            miscommunications=state.miscommunications,
            mttr_steps=-1 if state.resolved else state.step_count,
            cumulative_reward=state.cumulative_reward,
            action_counts=dict(self._action_counter),
            oversight_flag_count=len(state.oversight_flags),
        )
        self._buffer.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def latest(self) -> MetricsSnapshot | None:
        return self._buffer[-1] if self._buffer else None

    def as_list(self) -> list[MetricsSnapshot]:
        return list(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._action_counter = Counter()

    # ------------------------------------------------------------------
    # Scalar helpers for lightweight dashboards
    # ------------------------------------------------------------------

    def latency_series(self, service: str) -> list[float]:
        """Return the latency history for ``service``. Empty if never recorded."""
        return [snap.service_latency_ms.get(service, 0.0) for snap in self._buffer]

    def error_series(self, service: str) -> list[float]:
        return [snap.service_error_rate.get(service, 0.0) for snap in self._buffer]

    def action_histogram(self) -> dict[str, int]:
        return dict(self._action_counter)

    def action_count(self, action_type: ActionType) -> int:
        return self._action_counter.get(action_type.value, 0)


__all__ = ["MetricsSnapshot", "MetricsRecorder"]
