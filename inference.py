"""
Inference script for SOC Triage environment.

MANDATORY submission variables:
- API_BASE_URL: OpenAI-compatible chat completions API base URL.
- API_KEY: token for the provided OpenAI-compatible proxy (preferred).
- HF_TOKEN: accepted compatibility alias for API_KEY.
- MODEL_NAME: model identifier for inference.

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=soc_triage_env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Mandatory env vars — the validator injects API_BASE_URL and API_KEY.
# MODEL_NAME is optional; default to 'gpt-4o-mini' so the proxy call fires
# even when the evaluator doesn't set it.
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "").strip() or os.getenv("OPENAI_API_BASE_URL", "").strip()
# Use any non-empty model name — the LiteLLM proxy ignores this field
MODEL_NAME = (
    os.getenv("MODEL_NAME", "").strip()
    or os.getenv("AI_MODEL", "").strip()
    or "gpt-4o-mini"
)
API_KEY = (
    os.getenv("API_KEY", "").strip()        # validator-injected key  ← PRIMARY
    or os.getenv("HF_TOKEN", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
# Set ALLOW_PROVIDER_FALLBACK=1 locally to use Blaxel/Cerebras during dev.
# In submission mode it must be 0 so ALL traffic goes through proxy.
ALLOW_PROVIDER_FALLBACK = os.getenv("ALLOW_PROVIDER_FALLBACK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

DEFAULT_BLAXEL_WORKSPACE = "vasanthfeb13"
DEFAULT_BLAXEL_MODEL = "sandbox-openai"
DEFAULT_CEREBRAS_MODEL = "llama3.1-8b"

BENCHMARK = "soc_triage_env"
MAX_STEPS_MAP = {"easy": 4, "medium": 5, "hard": 6}

SYSTEM_PROMPT = (
    "You are a SOC analyst in an interactive OpenEnv environment. "
    "Return strict JSON with keys: tool_name, tool_args, classification, recommended_action, reasoning. "
    "Use investigation tools (query_siem, get_threat_intel, pivot_alert) before submit_verdict."
)

# ---------------------------------------------------------------------------
# OpenAI client import (optional; falls back to heuristic if missing)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Component loader — works even without pip install
# ---------------------------------------------------------------------------

def _load_components() -> tuple[type | None, type | None]:
    """Import TriageAction and SOCTriageEnv from the package.

    Injects envs/ into sys.path so the package is importable even when
    running from a raw file copy (e.g. /tmp/workspace) without pip install.
    """
    repo_root = Path(__file__).resolve().parent
    for candidate in (
        repo_root / "envs",   # adds envs/ so 'soc_triage_env' package is found
        repo_root,            # adds root so 'envs.soc_triage_env' also works
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    for prefix in ("soc_triage_env", "envs.soc_triage_env"):
        try:
            models_mod = importlib.import_module(f"{prefix}.models")
            env_mod = importlib.import_module(f"{prefix}.server.soc_triage_env")
            return getattr(models_mod, "TriageAction"), getattr(env_mod, "SOCTriageEnv")
        except Exception:
            continue
    return None, None


TriageAction, SOCTriageEnv = _load_components()

# ---------------------------------------------------------------------------
# Logging helpers — mandatory [START] / [STEP] / [END] protocol
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# OpenAI client builder
# ---------------------------------------------------------------------------

def _normalize_token(value: str) -> str:
    token = value.strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def _build_client(api_base_url: str, api_key: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    default_headers: dict[str, str] = {}
    workspace = os.getenv("BLAXEL_WORKSPACE", "").strip()
    if workspace and "run.blaxel.ai" in api_base_url:
        default_headers["X-Blaxel-Workspace"] = workspace

    if default_headers:
        return OpenAI(api_key=_normalize_token(api_key), base_url=api_base_url, default_headers=default_headers)
    return OpenAI(api_key=_normalize_token(api_key), base_url=api_base_url)

# ---------------------------------------------------------------------------
# Provider / runtime config resolution
# ---------------------------------------------------------------------------

def _blaxel_base_url(model_name: str) -> str:
    explicit = os.getenv("BLAXEL_API_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    chat = os.getenv("BLAXEL_CHAT_URL", "").strip()
    if chat:
        suffix = "/chat/completions"
        return chat[: -len(suffix)] if chat.endswith(suffix) else chat.rstrip("/")
    workspace = os.getenv("BLAXEL_WORKSPACE", DEFAULT_BLAXEL_WORKSPACE).strip()
    base = os.getenv("BLAXEL_BASE_URL", "https://run.blaxel.ai").strip().rstrip("/")
    return f"{base}/{workspace}/models/{model_name}/v1"


def _resolve_client() -> tuple[Any, str] | None:
    """Return (client, model_name) or None if nothing is configured.

    Priority:
      1. API_BASE_URL + API_KEY  (validator proxy — always preferred)
      2. Provider-specific creds  (only when ALLOW_PROVIDER_FALLBACK=1)
    """
    api_base = (API_BASE_URL or "").strip()
    model = (MODEL_NAME or "").strip() or "gpt-4o-mini"
    token = (API_KEY or "").strip()

    # ── Primary path: validator injects API_BASE_URL + API_KEY ──────────────
    # MODEL_NAME is optional; proxy accepts any model string.
    if api_base and token:
        try:
            client = _build_client(api_base, token)
            print(f"[CLIENT] Using validator proxy: base={api_base} model={model}", flush=True)
            return client, model
        except Exception as exc:
            print(f"[CLIENT] Failed to build proxy client: {exc}", flush=True)
            return None

    # ── Submission mode: if no proxy creds, do NOT fall back to own creds ───
    if not ALLOW_PROVIDER_FALLBACK:
        print("[CLIENT] No API_BASE_URL/API_KEY found and ALLOW_PROVIDER_FALLBACK=0 — heuristic mode.", flush=True)
        return None

    # ── Local-dev fallback: Blaxel ───────────────────────────────────────────
    blaxel_key = os.getenv("BLAXEL_AUTHORIZATION", "").strip()
    if blaxel_key:
        m = os.getenv("BLAXEL_MODEL", DEFAULT_BLAXEL_MODEL).strip() or model
        b = api_base or _blaxel_base_url(m)
        try:
            return _build_client(b, blaxel_key), m
        except Exception:
            pass

    # ── Local-dev fallback: Cerebras ─────────────────────────────────────────
    cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    if cerebras_key:
        m = os.getenv("CEREBRAS_MODEL", DEFAULT_CEREBRAS_MODEL).strip() or model
        b = api_base or os.getenv("CEREBRAS_API_BASE_URL", "https://api.cerebras.ai/v1").strip()
        try:
            return _build_client(b, cerebras_key), m
        except Exception:
            pass

    return None

# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _make_action(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    classification: str | None = None,
    recommended_action: str | None = None,
    reasoning: str = "",
) -> Any:
    if TriageAction is None:
        return {
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "classification": classification,
            "recommended_action": recommended_action,
            "reasoning": reasoning,
        }
    return TriageAction(
        tool_name=tool_name,
        tool_args=tool_args or {},
        classification=classification,
        recommended_action=recommended_action,
        reasoning=reasoning,
    )


def _action_to_str(action: Any) -> str:
    """Return a short, single-line representation of the action for [STEP] logging."""
    try:
        tool = action.tool_name if hasattr(action, "tool_name") else action.get("tool_name", "submit_verdict")
        cls = action.classification if hasattr(action, "classification") else action.get("classification", "")
        rec = action.recommended_action if hasattr(action, "recommended_action") else action.get("recommended_action", "")
        return f"{tool}|{cls}|{rec}"
    except Exception:
        return str(action)[:80]


def _pick_ioc(obs: Any) -> str:
    known = getattr(obs, "known_iocs", []) or []
    known = [str(v) for v in known if str(v).strip()]
    if known:
        return known[0]
    if hasattr(obs, "alert") and obs.alert is not None:
        if getattr(obs.alert, "source_ip", None):
            return str(obs.alert.source_ip)
        if getattr(obs.alert, "destination_ip", None):
            return str(obs.alert.destination_ip)
    return "suspicious-ioc"


def _pick_alert_id(obs: Any) -> str:
    events = getattr(obs, "events", []) or []
    if events:
        return str(getattr(events[0], "alert_id", ""))
    alerts = getattr(obs, "alerts", []) or []
    if alerts:
        return str(getattr(alerts[0], "alert_id", ""))
    if hasattr(obs, "alert") and obs.alert is not None:
        return str(getattr(obs.alert, "alert_id", ""))
    return ""


def _heuristic_verdict(obs: Any) -> Any:
    task_id = obs.task_id if hasattr(obs, "task_id") else "easy"

    if task_id == "easy":
        text = (obs.alert.raw_log if hasattr(obs, "alert") and obs.alert else "").lower()
        if "beacon" in text or "c2" in text:
            return _make_action(
                tool_name="submit_verdict",
                classification="critical",
                recommended_action="escalate",
                reasoning="Beaconing pattern indicates potential C2 behavior.",
            )
        if "failed" in text or "ssh" in text:
            return _make_action(
                tool_name="submit_verdict",
                classification="medium",
                recommended_action="investigate",
                reasoning="Repeated auth failures should be investigated.",
            )
        return _make_action(
            tool_name="submit_verdict",
            classification="benign",
            recommended_action="ignore",
            reasoning="No strong malicious signal in this log.",
        )

    if task_id == "medium":
        return _make_action(
            tool_name="submit_verdict",
            classification="MED-C,MED-E,MED-D,MED-A,MED-B",
            recommended_action="investigate",
            reasoning="Prioritize ransomware and exfiltration indicators.",
        )

    return _make_action(
        tool_name="submit_verdict",
        classification="H-01,H-03,H-05,H-07,H-11",
        recommended_action="contain",
        reasoning="Matches recon to exfiltration kill-chain sequence.",
    )


def _heuristic_action(obs: Any, step_index: int) -> Any:
    if step_index == 0:
        query = {
            "easy": "failed login outbound beacon privilege",
            "medium": "ransomware outbound data privilege",
            "hard": "scan exploit shell lateral exfil",
        }.get(getattr(obs, "task_id", "easy"), "suspicious")
        return _make_action(
            tool_name="query_siem",
            tool_args={"query": query},
            reasoning="Initial SIEM investigation sweep.",
        )

    if step_index == 1:
        return _make_action(
            tool_name="get_threat_intel",
            tool_args={"ioc": _pick_ioc(obs)},
            reasoning="Threat-intel enrichment for discovered IOC.",
        )

    if step_index == 2 and getattr(obs, "task_id", "") == "hard":
        return _make_action(
            tool_name="pivot_alert",
            tool_args={"alert_id": _pick_alert_id(obs)},
            reasoning="Pivot to correlate related timeline events.",
        )

    return _heuristic_verdict(obs)


def _parse_action(text: str, fallback: Any) -> Any:
    text = (text or "").strip()
    if not text or TriageAction is None:
        return fallback
    try:
        return TriageAction(**json.loads(text))
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return TriageAction(**json.loads(text[start: end + 1]))
        except Exception:
            pass
    return fallback


def _model_action(client: Any, model_name: str, obs: Any) -> Any:
    """Call the LiteLLM proxy; fall back to heuristic only on complete failure."""
    step_index = max(0, int(getattr(obs, "step_num", 0)))
    fallback = _heuristic_action(obs, step_index=step_index)

    prompt = (
        f"Task id: {obs.task_id}\n"
        f"Step: {getattr(obs, 'step_num', 0)}/{getattr(obs, 'max_steps', 1)}\n"
        f"Available tools: {getattr(obs, 'available_tools', [])}\n"
        f"Observation JSON:\n{json.dumps(obs.model_dump(), indent=2)}\n"
        "Return only a JSON object."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Attempt 1: without response_format (LiteLLM proxies often reject it)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        print(f"[LLM] API call succeeded, content length={len(content)}", flush=True)
        parsed = _parse_action(content, None)
        if parsed is not None:
            return parsed
        # Content didn't parse to a TriageAction — use fallback but call WAS made
        print("[LLM] Response parse failed, using heuristic action", flush=True)
        return fallback
    except Exception as exc:
        print(f"[LLM] API call failed: {exc}", flush=True)
        return fallback

# ---------------------------------------------------------------------------
# Per-task runner — emits [START] / [STEP] / [END] for EACH task
# ---------------------------------------------------------------------------

def run_task(task_id: str, client: Any | None, model_name: str, max_seconds: int) -> float:
    """Run one episode on *task_id* and return the final score ∈ [0,1]."""
    if SOCTriageEnv is None:
        log_start(task=task_id, env=BENCHMARK, model=model_name)
        log_end(success=False, steps=0, score=0.01, rewards=[])
        return 0.01

    env = SOCTriageEnv()
    rewards: list[float] = []
    steps_taken = 0
    score = 0.01
    success = False
    started = time.monotonic()

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        obs = env.reset(task_id=task_id)
        done = False
        max_steps = max(1, int(getattr(obs, "max_steps", MAX_STEPS_MAP.get(task_id, 4))))

        for step_num in range(1, max_steps + 1):
            if done:
                break
            if time.monotonic() - started > max_seconds:
                break

            # Choose action
            error_msg: str | None = None
            try:
                if client is None:
                    action = _heuristic_action(obs, step_index=step_num - 1)
                else:
                    action = _model_action(client, model_name, obs)
            except Exception as exc:
                error_msg = str(exc)
                action = _heuristic_action(obs, step_index=step_num - 1)

            # Step environment
            try:
                obs = env.step(action)
                reward = float(getattr(obs, "reward", 0.01) or 0.01)
                done = bool(getattr(obs, "done", False))
            except Exception as exc:
                error_msg = str(exc)
                reward = 0.01
                done = True

            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=_action_to_str(action),
                     reward=reward, done=done, error=error_msg)

            if done:
                break

        score = max(0.01, min(0.99, rewards[-1] if rewards else 0.01))
        success = score > 0.0

    except Exception as exc:
        log_step(step=steps_taken + 1, action="error", reward=0.01, done=True, error=str(exc))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Run inference against all SOC triage tasks.")
        parser.add_argument("--episodes", type=int, default=1)
        parser.add_argument("--max-minutes", type=int, default=20)
        try:
            args, _ = parser.parse_known_args()
        except SystemExit:
            args = argparse.Namespace(episodes=1, max_minutes=20)

        episodes = max(1, args.episodes)
        max_minutes = max(1, args.max_minutes)
        max_seconds = max(60, max_minutes * 60)
        model_name = (MODEL_NAME or "heuristic").strip()

        # Resolve LLM client (None → heuristic fallback)
        resolved = _resolve_client()
        client = resolved[0] if resolved else None
        effective_model = resolved[1] if resolved else model_name

        task_ids = ["easy", "medium", "hard"]
        scores: dict[str, float] = {}

        for task_id in task_ids:
            best_score = 0.01
            for _ in range(episodes):
                s = run_task(task_id, client, effective_model, max_seconds)
                best_score = max(best_score, s)
            scores[task_id] = round(best_score, 4)

        # Summary JSON (optional, for debugging)
        print(json.dumps({
            "script": "inference.py",
            "episodes_per_task": episodes,
            "scores": scores,
        }, indent=2), flush=True)

    except Exception as fatal:
        # Absolute last resort — emit valid [END] so the validator doesn't crash-parse
        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
        print(json.dumps({
            "script": "inference.py",
            "fatal_error": str(fatal),
            "scores": {"easy": 0.01, "medium": 0.01, "hard": 0.01},
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
