"""Task metadata for the SOC triage environment."""

from __future__ import annotations

TASKS: dict[str, dict[str, object]] = {
    "easy": {
        "id": "easy",
        "difficulty": "easy",
        "description": "Classify a single security alert severity.",
        "max_steps": 1,
        "grader": "grade_easy",
        "action_schema": {
            "classification": "benign|low|medium|high|critical",
            "recommended_action": "ignore|investigate|block|escalate",
            "reasoning": "string",
        },
    },
    "medium": {
        "id": "medium",
        "difficulty": "medium",
        "description": "Rank a queue of alerts by urgency.",
        "max_steps": 1,
        "grader": "grade_medium",
        "action_schema": {
            "classification": "comma-separated alert_ids in priority order",
            "recommended_action": "monitor|investigate|escalate",
            "reasoning": "string",
        },
    },
    "hard": {
        "id": "hard",
        "difficulty": "hard",
        "description": "Identify kill-chain events from a noisy timeline.",
        "max_steps": 3,
        "grader": "grade_hard",
        "action_schema": {
            "classification": "comma-separated alert_ids that form kill chain",
            "recommended_action": "contain|eradicate|escalate|report",
            "reasoning": "string",
        },
    },
}


def get_task(task_id: str) -> dict[str, object]:
    """Return task metadata or raise a clear error."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Expected one of {list(TASKS)}")
    return TASKS[task_id]
