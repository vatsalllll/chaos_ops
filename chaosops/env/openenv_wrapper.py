"""OpenEnv client + FastAPI server entry points.

The :class:`ChaosOpsClient` is the thing TRL-style training code imports:

    from chaosops.env.openenv_wrapper import ChaosOpsClient

    with ChaosOpsClient(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        obs = env.step(action)

The server side is created with :func:`build_fastapi_app`, which returns a
ready-to-serve FastAPI instance. ``server/app.py`` simply re-exports it so
``uvicorn chaosops.env.openenv_wrapper:app`` works out of the box.

If ``openenv-core`` is not installed the client import will raise at the
call site — consistent with the rest of the package: the simulator itself
has no hard dependency on OpenEnv so unit tests stay lightweight.
"""

from __future__ import annotations

from typing import Any

from chaosops.env.environment import ChaosOpsEnvironment
from chaosops.env.models import (
    ChaosOpsAction,
    ChaosOpsObservation,
    ChaosOpsState,
    RoleView,
)


try:  # pragma: no cover — optional dependency
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server import create_fastapi_app
    from openenv.core.client_types import StepResult

    _HAS_OPENENV = True
except ImportError:  # pragma: no cover
    _HAS_OPENENV = False
    EnvClient = object  # type: ignore[assignment, misc]
    StepResult = None  # type: ignore[assignment]
    create_fastapi_app = None  # type: ignore[assignment]


if _HAS_OPENENV:

    class ChaosOpsClient(EnvClient[ChaosOpsAction, ChaosOpsObservation, ChaosOpsState]):  # type: ignore[misc]
        """Typed OpenEnv client for ChaosOps AI."""

        def _step_payload(self, action: ChaosOpsAction) -> dict[str, Any]:
            return action.model_dump(mode="json")

        def _parse_result(self, payload: dict[str, Any]) -> "StepResult":  # type: ignore[name-defined]
            obs_data = payload.get("observation", {})
            view_data = obs_data.get("view", {})
            view = RoleView.model_validate(view_data) if view_data else None
            observation = ChaosOpsObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                view=view,  # type: ignore[arg-type]
                step=obs_data.get("step", 0),
                turn_role=obs_data.get("turn_role"),
                message=obs_data.get("message", ""),
            )
            return StepResult(  # type: ignore[operator]
                observation=observation,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict[str, Any]) -> ChaosOpsState:
            return ChaosOpsState.model_validate(payload)

    def build_fastapi_app():  # type: ignore[no-untyped-def]
        """Return a ready-to-serve FastAPI app exposing ChaosOps AI."""
        assert create_fastapi_app is not None
        return create_fastapi_app(ChaosOpsEnvironment)

    app = build_fastapi_app()
else:

    class ChaosOpsClient:  # type: ignore[no-redef]
        """Placeholder — install ``openenv-core`` to use the HTTP client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "openenv-core is not installed. Run `pip install openenv-core` "
                "to use the ChaosOps HTTP client."
            )

    def build_fastapi_app():  # type: ignore[no-untyped-def]
        raise ImportError("openenv-core is required to build the FastAPI server app.")

    app = None


__all__ = ["ChaosOpsClient", "build_fastapi_app", "app"]
