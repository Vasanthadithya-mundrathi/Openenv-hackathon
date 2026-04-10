"""Task metadata for the SOC triage environment."""

from __future__ import annotations

TASKS: dict[str, dict[str, object]] = {
    "easy": {
        "id": "easy",
        "difficulty": "easy",
        "description": "Investigate a single security alert and submit a severity verdict.",
        "max_steps": 4,
        "grader": "grade_easy",
        "action_schema": {
            "tool_name": "list_tools|query_siem|get_threat_intel|pivot_alert|submit_verdict",
            "tool_args": "object",
            "classification": "required for submit_verdict: benign|low|medium|high|critical",
            "recommended_action": "required for submit_verdict: ignore|investigate|block|escalate",
            "reasoning": "required for submit_verdict: string",
        },
    },
    "medium": {
        "id": "medium",
        "difficulty": "medium",
        "description": "Investigate and rank an alert queue by urgency.",
        "max_steps": 5,
        "grader": "grade_medium",
        "action_schema": {
            "tool_name": "list_tools|query_siem|get_threat_intel|pivot_alert|submit_verdict",
            "tool_args": "object",
            "classification": "required for submit_verdict: comma-separated alert_ids in priority order",
            "recommended_action": "required for submit_verdict: monitor|investigate|escalate",
            "reasoning": "required for submit_verdict: string",
        },
    },
    "hard": {
        "id": "hard",
        "difficulty": "hard",
        "description": "Investigate noisy timeline and identify the kill-chain sequence.",
        "max_steps": 6,
        "grader": "grade_hard",
        "action_schema": {
            "tool_name": "list_tools|query_siem|get_threat_intel|pivot_alert|submit_verdict",
            "tool_args": "object",
            "classification": "required for submit_verdict: comma-separated alert_ids that form kill chain",
            "recommended_action": "required for submit_verdict: contain|eradicate|escalate|report",
            "reasoning": "required for submit_verdict: string",
        },
    },
}


def get_task(task_id: str) -> dict[str, object]:
    """Return task metadata or raise a clear error."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Expected one of {list(TASKS)}")
    return TASKS[task_id]
