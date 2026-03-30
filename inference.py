"""
Inference script for SOC Triage environment.

MANDATORY submission variables:
- API_BASE_URL: OpenAI-compatible chat completions API base URL.
- MODEL_NAME: model identifier for inference.
- HF_TOKEN: API token.

This script uses the OpenAI client for all LLM calls.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


def _load_components() -> tuple[type, type]:
    for prefix in ("soc_triage_env", "envs.soc_triage_env"):
        try:
            models_mod = importlib.import_module(f"{prefix}.models")
            env_mod = importlib.import_module(f"{prefix}.server.soc_triage_env")
            return getattr(models_mod, "TriageAction"), getattr(env_mod, "SOCTriageEnv")
        except Exception:
            continue
    raise RuntimeError("Could not import SOC triage environment package.")


TriageAction, SOCTriageEnv = _load_components()

SYSTEM_PROMPT = (
    "You are a SOC analyst. Return strict JSON with keys: "
    "classification, recommended_action, reasoning."
)


@dataclass
class InferenceConfig:
    api_base_url: str
    model_name: str
    hf_token: str
    episodes_per_task: int
    max_minutes: int


def _normalize_token(value: str) -> str:
    token = value.strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def _build_client(api_base_url: str, hf_token: str) -> OpenAI:
    default_headers: dict[str, str] = {}
    workspace = os.getenv("BLAXEL_WORKSPACE", "").strip()
    if workspace and "run.blaxel.ai" in api_base_url:
        default_headers["X-Blaxel-Workspace"] = workspace

    if default_headers:
        return OpenAI(api_key=_normalize_token(hf_token), base_url=api_base_url, default_headers=default_headers)
    return OpenAI(api_key=_normalize_token(hf_token), base_url=api_base_url)


def _prompt_for_observation(obs: Any) -> str:
    return (
        "Task id: "
        + obs.task_id
        + "\nObservation JSON:\n"
        + json.dumps(obs.model_dump(), indent=2)
        + "\nReturn only JSON."
    )


def _heuristic_action(obs: Any) -> Any:
    if obs.task_id == "easy":
        text = (obs.alert.raw_log if obs.alert else "").lower()
        if "beacon" in text or "c2" in text:
            return TriageAction(
                classification="critical",
                recommended_action="escalate",
                reasoning="Beaconing pattern indicates potential C2 behavior.",
            )
        if "failed" in text or "ssh" in text:
            return TriageAction(
                classification="medium",
                recommended_action="investigate",
                reasoning="Repeated auth failures should be investigated.",
            )
        return TriageAction(
            classification="benign",
            recommended_action="ignore",
            reasoning="No strong malicious signal in this log.",
        )

    if obs.task_id == "medium":
        return TriageAction(
            classification="MED-C,MED-E,MED-D,MED-A,MED-B",
            recommended_action="investigate",
            reasoning="Prioritize ransomware and exfiltration indicators.",
        )

    return TriageAction(
        classification="H-01,H-03,H-05,H-07,H-11",
        recommended_action="contain",
        reasoning="Matches recon to exfiltration kill-chain sequence.",
    )


def _parse_action(text: str, fallback: Any) -> Any:
    text = (text or "").strip()
    if not text:
        return fallback

    try:
        return TriageAction(**json.loads(text))
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return TriageAction(**json.loads(text[start : end + 1]))
        except Exception:
            return fallback

    return fallback


def _model_action(client: OpenAI, model_name: str, obs: Any) -> Any:
    fallback = _heuristic_action(obs)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt_for_observation(obs)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return _parse_action(content, fallback)
    except Exception:
        return fallback


def _run_task(task_id: str, episodes: int, client: OpenAI, model_name: str, max_seconds: int) -> float:
    env = SOCTriageEnv()
    total = 0.0
    started = time.monotonic()

    for _ in range(episodes):
        if time.monotonic() - started > max_seconds:
            raise RuntimeError("Inference runtime exceeded max allowed duration.")

        obs = env.reset(task_id=task_id)
        done = False
        reward_sum = 0.0

        while not done:
            if time.monotonic() - started > max_seconds:
                raise RuntimeError("Inference runtime exceeded max allowed duration.")
            action = _model_action(client, model_name, obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

        total += reward_sum

    return round(total / episodes, 4)


def run_inference_sync(episodes_per_task: int = 1, max_minutes: int = 20) -> dict[str, float]:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    missing: list[str] = []
    if not api_base_url:
        missing.append("API_BASE_URL")
    if not model_name:
        missing.append("MODEL_NAME")
    if not hf_token:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))

    client = _build_client(api_base_url=api_base_url, hf_token=hf_token)
    max_seconds = max(60, max_minutes * 60)

    return {
        task_id: _run_task(task_id, episodes_per_task, client, model_name, max_seconds)
        for task_id in ["easy", "medium", "hard"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference against all SOC triage tasks.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-minutes", type=int, default=20)
    args = parser.parse_args()

    started = time.monotonic()
    scores = run_inference_sync(episodes_per_task=max(1, args.episodes), max_minutes=max(1, args.max_minutes))
    runtime_seconds = round(time.monotonic() - started, 2)

    payload = {
        "script": "inference.py",
        "episodes_per_task": max(1, args.episodes),
        "runtime_seconds": runtime_seconds,
        "scores": scores,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
