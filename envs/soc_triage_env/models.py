"""Typed models for the SOC triage environment."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except Exception:  # pragma: no cover - fallback for local dev before openenv install
    Action = BaseModel
    Observation = BaseModel
    State = BaseModel


class AlertRecord(BaseModel):
    """Single security event record."""

    alert_id: str
    timestamp: str | None = None
    source_ip: str | None = None
    destination_ip: str | None = None
    event_type: str
    raw_log: str


class TriageAction(Action):
    """Agent action for SOC triage decisions."""

    classification: str = Field(
        ...,
        description="Severity label or comma-separated alert ids for ranking/kill-chain tasks.",
    )
    recommended_action: str = Field(
        ...,
        description="One of ignore, investigate, block, escalate, contain, eradicate, report.",
    )
    reasoning: str = Field(..., description="Free-form explanation for the decision.")


class TriageReward(BaseModel):
    """Detailed reward breakdown."""

    score: float = Field(ge=0.0, le=1.0, default=0.01)
    base_score: float = Field(ge=0.0, le=1.0, default=0.01)
    partial_credit: float = Field(ge=0.0)
    penalty: float = Field(ge=0.0)
    feedback: str


class TriageObservation(Observation):
    """Observation returned to the agent."""

    task_id: str
    difficulty: str
    step_num: int
    max_steps: int
    prompt: str
    alert: AlertRecord | None = None
    alerts: list[AlertRecord] = Field(default_factory=list)
    events: list[AlertRecord] = Field(default_factory=list)
    context_history: list[str] = Field(default_factory=list)
    done: bool = False
    reward: float = 0.01


class TriageState(State):
    """Environment state and counters for a running episode."""

    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = "easy"
    task_index: int = 0
    step_count: int = 0
    max_steps: int = 1
    done: bool = False
    total_reward: float = 0.01
    last_score: float = 0.01
    false_positives: int = 0
    correct_escalations: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
