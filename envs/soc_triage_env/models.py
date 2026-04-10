"""Typed models for the SOC triage environment."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except Exception:  # pragma: no cover - compatibility with older layouts
    try:
        from openenv.core.env_server.interfaces import Action, Observation, State
    except Exception:  # pragma: no cover - local dev before openenv install
        Action = BaseModel
        Observation = BaseModel
        State = BaseModel


ALLOWED_TOOLS = {
    "list_tools",
    "query_siem",
    "get_threat_intel",
    "pivot_alert",
    "submit_verdict",
}


class AlertRecord(BaseModel):
    """Single security event record."""

    alert_id: str
    timestamp: str | None = None
    source_ip: str | None = None
    destination_ip: str | None = None
    event_type: str
    raw_log: str


class TriageAction(Action):
    """Agent action for SOC triage investigation and verdict submission."""

    tool_name: str = Field(
        default="submit_verdict",
        description=(
            "One of list_tools, query_siem, get_threat_intel, pivot_alert, submit_verdict. "
            "If omitted, action is treated as submit_verdict for backward compatibility."
        ),
    )
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool, for example {'query': 'outbound c2'}.",
    )
    classification: str | None = Field(
        default=None,
        description=(
            "Required for submit_verdict. Severity label for easy task, comma-separated alert ids "
            "for medium/hard tasks."
        ),
    )
    recommended_action: str | None = Field(
        default=None,
        description="Operational response decision, used mainly for submit_verdict.",
    )
    reasoning: str = Field(default="", description="Free-form explanation for the decision.")

    @field_validator("tool_name", mode="before")
    @classmethod
    def _normalize_tool_name(cls, value: Any) -> str:
        if value is None:
            return "submit_verdict"
        normalized = str(value).strip().lower()
        if not normalized:
            return "submit_verdict"
        if normalized in ALLOWED_TOOLS:
            return normalized
        return "query_siem"


class TriageReward(BaseModel):
    """Detailed reward breakdown."""

    score: float = Field(gt=0.0, lt=1.0, default=0.01)
    base_score: float = Field(gt=0.0, lt=1.0, default=0.01)
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
    available_tools: list[str] = Field(default_factory=lambda: sorted(ALLOWED_TOOLS))
    investigation_notes: list[str] = Field(default_factory=list)
    known_iocs: list[str] = Field(default_factory=list)
    last_tool_result: dict[str, Any] | None = None
    context_history: list[str] = Field(default_factory=list)
    feedback: str | None = None
    done: bool = False
    reward: float = 0.01


class TriageState(State):
    """Environment state and counters for a running episode."""

    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = "easy"
    task_index: int = 0
    step_count: int = 0
    max_steps: int = 4
    done: bool = False
    total_reward: float = 0.01
    last_score: float = 0.01
    false_positives: int = 0
    correct_escalations: int = 0
    investigation_steps: int = 0
    submitted_verdict: bool = False
    tools_used: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
