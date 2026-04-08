"""Client wrappers for SOC triage environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import requests

from soc_triage_env.models import TriageAction, TriageObservation, TriageState

ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: float | None
    done: bool
    info: dict


class SOCTriageEnvClient:
    """Minimal sync HTTP client compatible with reset/step/state endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, task_id: str = "easy") -> StepResult[TriageObservation]:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=TriageObservation(**payload["observation"]),
            reward=float(payload.get("reward", 0.01)),
            done=bool(payload.get("done", False)),
            info=dict(payload.get("info", {})),
        )

    def step(self, action: TriageAction) -> StepResult[TriageObservation]:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=TriageObservation(**payload["observation"]),
            reward=float(payload.get("reward", 0.01)),
            done=bool(payload.get("done", False)),
            info=dict(payload.get("info", {})),
        )

    def state(self) -> TriageState:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return TriageState(**response.json())
