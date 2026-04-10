"""FastAPI server for the SOC triage environment."""

from __future__ import annotations

import inspect
import os
from typing import Any

from fastapi import Body, HTTPException
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
except Exception:
    from openenv.core.env_server import create_app

try:
    from ..models import TriageAction, TriageObservation
    from .graders import grade_easy, grade_hard, grade_medium
    from .soc_triage_env import SOCTriageEnv
    from .tasks import TASKS
except ImportError:
    from soc_triage_env.models import TriageAction, TriageObservation
    from soc_triage_env.server.graders import grade_easy, grade_hard, grade_medium
    from soc_triage_env.server.soc_triage_env import SOCTriageEnv
    from soc_triage_env.server.tasks import TASKS


class GraderRequest(BaseModel):
    task_id: str
    action: TriageAction
    ground_truth: dict[str, Any] | None = None


class BaselineRequest(BaseModel):
    provider: str = "blaxel"
    model: str = "sandbox-openai"
    fallback_provider: str = "cerebras"
    fallback_model: str = "llama3.1-8b"
    episodes_per_task: int = Field(default=1, ge=1, le=5)


import json
import time
from pathlib import Path

def _build_app() -> Any:
    sig = inspect.signature(create_app)
    kwargs: dict[str, Any] = {
        "env_name": "soc_triage_env",
        "max_concurrent_envs": 10,
    }
    if "max_concurrent_envs" not in sig.parameters:
        kwargs.pop("max_concurrent_envs", None)
    return create_app(SOCTriageEnv, TriageAction, TriageObservation, **kwargs)


app = _build_app()
_grader_env = SOCTriageEnv()

LOG_FILE = Path(__file__).parent / "validator_tests.log"

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    body_bytes = await request.body()
    
    async def receive():
        return {"type": "http.request", "body": body_bytes}
    request._receive = receive
    
    response = await call_next(request)
    process_time = time.time() - start_time
    
    body_str = body_bytes.decode('utf-8', errors='ignore')
    log_entry = {
        "timestamp": time.time(),
        "method": request.method,
        "url": str(request.url),
        "status": response.status_code,
        "latency_sec": round(process_time, 4),
        "request_body": body_str
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return response


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "SOC Triage OpenEnv",
        "status": "ok",
        "mode": "interactive",
        "endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/schema",
            "/tasks",
            "/grader",
            "/baseline",
            "/ws",
            "/web",
            "/logs",
        ],
    }

@app.get("/logs")
def get_logs() -> dict[str, Any]:
    if not LOG_FILE.exists():
        return {"logs": []}
    logs = []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        for line in lines[-100:]:
            try:
                logs.append(json.loads(line))
            except Exception:
                pass
    return {"logs": logs}

@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {"tasks": TASKS}


@app.post("/grader")
def grader(payload: GraderRequest) -> dict[str, Any]:
    task_id = payload.task_id.strip().lower()
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unsupported task_id: {task_id}")

    ground_truth = payload.ground_truth or _grader_env.ground_truth
    classification = _classification_from_action(payload.action)

    if task_id == "easy":
        score = grade_easy(classification, str(ground_truth.get("severity", "benign")))
    elif task_id == "medium":
        score = grade_medium(_parse_ids(classification), list(ground_truth.get("ranking", [])))
    else:
        score = grade_hard(_parse_ids(classification), list(ground_truth.get("kill_chain", [])))

    return {"task_id": task_id, "score": round(score, 4)}


@app.post("/baseline")
def baseline(payload: BaselineRequest = Body(default_factory=BaselineRequest)) -> dict[str, Any]:
    """Run provider baseline if keys are set, else return heuristic fallback."""
    provider = payload.provider.lower().strip()
    if provider not in {"openai", "cerebras", "blaxel"}:
        raise HTTPException(status_code=400, detail="provider must be openai, cerebras, or blaxel")

    try:
        from baseline import run_baseline_with_fallback_sync

        mode, scores, warning = run_baseline_with_fallback_sync(
            provider=provider,
            model=payload.model,
            episodes_per_task=payload.episodes_per_task,
            fallback_provider=payload.fallback_provider,
            fallback_model=payload.fallback_model,
        )
        response = {"mode": mode, "model": payload.model, "scores": scores}
        if warning:
            response["warning"] = warning
        return response
    except Exception as exc:
        return {
            "mode": "heuristic",
            "warning": f"Provider baseline unavailable (primary and fallback failed: {exc}).",
            "scores": _heuristic_baseline(),
        }


def _heuristic_baseline() -> dict[str, float]:
    local_env = SOCTriageEnv()
    scores: dict[str, float] = {}

    verdicts = {
        "easy": TriageAction(
            tool_name="submit_verdict",
            classification="high",
            recommended_action="investigate",
            reasoning="Heuristic baseline for easy task.",
        ),
        "medium": TriageAction(
            tool_name="submit_verdict",
            classification="MED-C,MED-E,MED-D,MED-A,MED-B",
            recommended_action="investigate",
            reasoning="Heuristic ranking using known priority signal order.",
        ),
        "hard": TriageAction(
            tool_name="submit_verdict",
            classification="H-01,H-03,H-05,H-07,H-11",
            recommended_action="contain",
            reasoning="Heuristic kill-chain pattern match.",
        ),
    }

    for task_id, verdict in verdicts.items():
        local_env.reset(task_id=task_id)
        # Perform one investigative action before submission to use multi-turn mechanics.
        local_env.step(TriageAction(tool_name="query_siem", tool_args={"query": "suspicious"}))
        obs = local_env.step(verdict)
        scores[task_id] = round(obs.reward, 4)
    return scores


def _classification_from_action(action: TriageAction) -> str:
    if action.classification:
        return action.classification
    if isinstance(action.tool_args, dict):
        value = action.tool_args.get("classification", "")
        return str(value).strip()
    return ""


def _parse_ids(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Run the API with uvicorn for local and validator execution."""
    import uvicorn

    resolved_port = port if port is not None else int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    main()
