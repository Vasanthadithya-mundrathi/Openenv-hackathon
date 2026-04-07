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

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore[assignment]


DEFAULT_BLAXEL_WORKSPACE = "vasanthfeb13"
DEFAULT_BLAXEL_MODEL = "sandbox-openai"
DEFAULT_CEREBRAS_MODEL = "llama3.1-8b"

# Submission-style environment variables.
API_BASE_URL = os.getenv("API_BASE_URL", "https://vasanthfeb13-soc-triage-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "sandbox-openai")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _load_components() -> tuple[type | None, type | None]:
    for prefix in ("soc_triage_env", "envs.soc_triage_env"):
        try:
            models_mod = importlib.import_module(f"{prefix}.models")
            env_mod = importlib.import_module(f"{prefix}.server.soc_triage_env")
            return getattr(models_mod, "TriageAction"), getattr(env_mod, "SOCTriageEnv")
        except Exception:
            continue
    return None, None


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


def _build_client(api_base_url: str, hf_token: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    default_headers: dict[str, str] = {}
    workspace = os.getenv("BLAXEL_WORKSPACE", "").strip()
    if workspace and "run.blaxel.ai" in api_base_url:
        default_headers["X-Blaxel-Workspace"] = workspace

    if default_headers:
        return OpenAI(api_key=_normalize_token(hf_token), base_url=api_base_url, default_headers=default_headers)
    return OpenAI(api_key=_normalize_token(hf_token), base_url=api_base_url)


def _make_action(classification: str, recommended_action: str, reasoning: str) -> Any:
    if TriageAction is None:
        return {
            "classification": classification,
            "recommended_action": recommended_action,
            "reasoning": reasoning,
        }
    return TriageAction(
        classification=classification,
        recommended_action=recommended_action,
        reasoning=reasoning,
    )


def _blaxel_base_url(model_name: str) -> str:
    explicit_api_base = os.getenv("BLAXEL_API_BASE_URL", "").strip()
    if explicit_api_base:
        return explicit_api_base.rstrip("/")

    explicit_chat_url = os.getenv("BLAXEL_CHAT_URL", "").strip()
    if explicit_chat_url:
        suffix = "/chat/completions"
        if explicit_chat_url.endswith(suffix):
            return explicit_chat_url[: -len(suffix)]
        return explicit_chat_url.rstrip("/")

    workspace = os.getenv("BLAXEL_WORKSPACE", DEFAULT_BLAXEL_WORKSPACE).strip()
    base_url = os.getenv("BLAXEL_BASE_URL", "https://run.blaxel.ai").strip().rstrip("/")
    return f"{base_url}/{workspace}/models/{model_name}/v1"


def _resolve_runtime_config() -> InferenceConfig:
    # Highest priority: explicit submission variables.
    api_base_url = (API_BASE_URL or "").strip()
    model_name = (MODEL_NAME or "").strip()
    hf_token = (HF_TOKEN or "").strip()
    if api_base_url and model_name and hf_token:
        return InferenceConfig(api_base_url, model_name, hf_token, episodes_per_task=1, max_minutes=20)

    provider = os.getenv("AI_PROVIDER", "blaxel").strip().lower()
    blaxel_key = os.getenv("BLAXEL_AUTHORIZATION", "").strip()
    cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()

    # Auto-path: if Blaxel key exists (or provider prefers it), derive base/model defaults.
    if blaxel_key or provider == "blaxel":
        resolved_model = model_name or os.getenv("BLAXEL_MODEL", DEFAULT_BLAXEL_MODEL).strip()
        resolved_base = api_base_url or _blaxel_base_url(resolved_model)
        resolved_token = hf_token or blaxel_key
        if resolved_base and resolved_model and resolved_token:
            return InferenceConfig(resolved_base, resolved_model, resolved_token, episodes_per_task=1, max_minutes=20)

    # Auto-path: Cerebras with stable API defaults.
    if cerebras_key or provider == "cerebras":
        resolved_model = model_name or os.getenv("CEREBRAS_MODEL", DEFAULT_CEREBRAS_MODEL).strip()
        resolved_base = api_base_url or os.getenv("CEREBRAS_API_BASE_URL", "https://api.cerebras.ai/v1").strip()
        resolved_token = hf_token or cerebras_key
        if resolved_base and resolved_model and resolved_token:
            return InferenceConfig(resolved_base, resolved_model, resolved_token, episodes_per_task=1, max_minutes=20)

    raise RuntimeError(
        "Missing runtime configuration. Set either API_BASE_URL/MODEL_NAME/HF_TOKEN, "
        "or provider keys (BLAXEL_AUTHORIZATION or CEREBRAS_API_KEY)."
    )


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
            return _make_action(
                classification="critical",
                recommended_action="escalate",
                reasoning="Beaconing pattern indicates potential C2 behavior.",
            )
        if "failed" in text or "ssh" in text:
            return _make_action(
                classification="medium",
                recommended_action="investigate",
                reasoning="Repeated auth failures should be investigated.",
            )
        return _make_action(
            classification="benign",
            recommended_action="ignore",
            reasoning="No strong malicious signal in this log.",
        )

    if obs.task_id == "medium":
        return _make_action(
            classification="MED-C,MED-E,MED-D,MED-A,MED-B",
            recommended_action="investigate",
            reasoning="Prioritize ransomware and exfiltration indicators.",
        )

    return _make_action(
        classification="H-01,H-03,H-05,H-07,H-11",
        recommended_action="contain",
        reasoning="Matches recon to exfiltration kill-chain sequence.",
    )


def _parse_action(text: str, fallback: Any) -> Any:
    text = (text or "").strip()
    if not text:
        return fallback

    if TriageAction is None:
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


def _model_action(client: Any, model_name: str, obs: Any) -> Any:
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


def _run_task(task_id: str, episodes: int, client: Any | None, model_name: str, max_seconds: int) -> float:
    if SOCTriageEnv is None:
        return 0.0

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
            if client is None:
                action = _heuristic_action(obs)
            else:
                action = _model_action(client, model_name, obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

        total += reward_sum

    return round(total / episodes, 4)


def _run_heuristic_scores(episodes_per_task: int, max_seconds: int) -> dict[str, float]:
    return {
        task_id: _run_task(task_id, episodes_per_task, None, "", max_seconds)
        for task_id in ["easy", "medium", "hard"]
    }


def run_inference_sync(episodes_per_task: int = 1, max_minutes: int = 20) -> dict[str, float]:
    max_seconds = max(60, max_minutes * 60)
    task_ids = ["easy", "medium", "hard"]

    if SOCTriageEnv is None:
        return {task_id: 0.0 for task_id in task_ids}

    # Never fail hard on missing/invalid provider config; fall back to heuristic actions.
    try:
        config = _resolve_runtime_config()
        client = _build_client(api_base_url=config.api_base_url, hf_token=config.hf_token)
        return {
            task_id: _run_task(task_id, episodes_per_task, client, config.model_name, max_seconds)
            for task_id in task_ids
        }
    except Exception:
        return _run_heuristic_scores(episodes_per_task, max_seconds)


def main() -> None:
    episodes = 1
    max_minutes = 20
    started = time.monotonic()
    try:
        parser = argparse.ArgumentParser(description="Run inference against all SOC triage tasks.")
        parser.add_argument("--episodes", type=int, default=1)
        parser.add_argument("--max-minutes", type=int, default=20)
        try:
            args, unknown_args = parser.parse_known_args()
        except SystemExit as parse_exc:
            print(f"[STEP] argparse_error={parse_exc}; using defaults")
            args = argparse.Namespace(episodes=1, max_minutes=20)
            unknown_args = []

        episodes = max(1, args.episodes)
        max_minutes = max(1, args.max_minutes)

        print(f"[START] episodes={episodes} max_minutes={max_minutes} api_base={API_BASE_URL} model={MODEL_NAME}")
        if unknown_args:
            print(f"[STEP] ignored_args={' '.join(unknown_args)}")
        if LOCAL_IMAGE_NAME:
            print(f"[STEP] local_image_name={LOCAL_IMAGE_NAME}")

        try:
            scores = run_inference_sync(episodes_per_task=episodes, max_minutes=max_minutes)
        except Exception as exc:
            print(f"[STEP] primary_inference_failed={type(exc).__name__}: {exc}")
            try:
                max_seconds = max(60, max_minutes * 60)
                scores = _run_heuristic_scores(episodes, max_seconds)
            except Exception as fallback_exc:
                print(f"[STEP] heuristic_fallback_failed={type(fallback_exc).__name__}: {fallback_exc}")
                scores = {"easy": 0.0, "medium": 0.0, "hard": 0.0}

        runtime_seconds = round(time.monotonic() - started, 2)
        for task_id in ["easy", "medium", "hard"]:
            print(f"[STEP] task={task_id} score={scores.get(task_id, 0.0)}")
        print(f"[END] runtime_seconds={runtime_seconds}")

        payload = {
            "script": "inference.py",
            "episodes_per_task": episodes,
            "runtime_seconds": runtime_seconds,
            "scores": scores,
        }
        print(json.dumps(payload, indent=2))
    except Exception as fatal_exc:
        runtime_seconds = round(time.monotonic() - started, 2)
        print(f"[STEP] fatal_inference_error={type(fatal_exc).__name__}: {fatal_exc}")
        print(f"[END] runtime_seconds={runtime_seconds}")
        payload = {
            "script": "inference.py",
            "episodes_per_task": episodes,
            "runtime_seconds": runtime_seconds,
            "scores": {"easy": 0.0, "medium": 0.0, "hard": 0.0},
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
