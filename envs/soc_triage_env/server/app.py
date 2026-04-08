"""FastAPI server for the SOC triage environment."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from soc_triage_env.models import TriageAction
from soc_triage_env.server.graders import grade_easy, grade_hard, grade_medium
from soc_triage_env.server.soc_triage_env import SOCTriageEnv
from soc_triage_env.server.tasks import TASKS


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: TriageAction


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

env = SOCTriageEnv()
app = FastAPI(title="SOC Triage OpenEnv", version="0.1.0")

# Setup raw validator logging
LOG_FILE = Path(__file__).parent / "validator_tests.log"

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Read body for logging
    body_bytes = await request.body()
    # Need to restore the body so it can be read by endpoints
    async def receive():
        return {"type": "http.request", "body": body_bytes}
    request._receive = receive
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # We can't easily read response body in middleware without consuming it,
    # but we can log the request url and body
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
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/logs"],
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest = Body(default=ResetRequest())) -> dict[str, Any]:
    try:
        obs = env.reset(task_id=payload.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": obs.model_dump(),
        "reward": 0.01,
        "done": False,
        "info": {"task_id": payload.task_id},
    }


@app.post("/step")
def step(payload: StepRequest) -> dict[str, Any]:
    try:
        obs, reward, done, info = env.step(payload.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": obs.model_dump(),
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return env.state.model_dump()


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {"tasks": TASKS}


@app.post("/grader")
def grader(payload: GraderRequest) -> dict[str, Any]:
    ground_truth = payload.ground_truth or env.ground_truth
    task_id = payload.task_id

    if task_id == "easy":
        score = grade_easy(payload.action.classification, str(ground_truth.get("severity", "benign")))
    elif task_id == "medium":
        predicted = _parse_ids(payload.action.classification)
        score = grade_medium(predicted, list(ground_truth.get("ranking", [])))
    elif task_id == "hard":
        predicted = _parse_ids(payload.action.classification)
        score = grade_hard(predicted, list(ground_truth.get("kill_chain", [])))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task_id: {task_id}")

    return {"task_id": task_id, "score": round(score, 4)}


@app.post("/baseline")
def baseline(payload: BaselineRequest = Body(default=BaselineRequest())) -> dict[str, Any]:
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

    easy_action = TriageAction(
        classification="high",
        recommended_action="investigate",
        reasoning="Heuristic baseline for easy task.",
    )
    medium_action = TriageAction(
        classification="MED-C,MED-E,MED-D,MED-A,MED-B",
        recommended_action="investigate",
        reasoning="Heuristic ranking using known priority signal order.",
    )
    hard_action = TriageAction(
        classification="H-01,H-03,H-05,H-07,H-11",
        recommended_action="contain",
        reasoning="Heuristic kill-chain pattern match.",
    )
    actions = {"easy": easy_action, "medium": medium_action, "hard": hard_action}

    for task_id, action in actions.items():
        local_env.reset(task_id=task_id)
        _, reward, _, _ = local_env.step(action)
        scores[task_id] = round(reward, 4)
    return scores


def _parse_ids(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    """Run the API with uvicorn for local and validator execution."""
    import uvicorn

    uvicorn.run("soc_triage_env.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
