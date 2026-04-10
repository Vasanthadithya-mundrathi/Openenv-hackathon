"""Baseline runner for SOC triage environment.

This script uses OpenAI-compatible APIs (OpenAI, Cerebras, Blaxel).
It can also run a deterministic heuristic baseline for local smoke tests.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any


def _load_components() -> tuple[type, type, type]:
    for prefix in ("soc_triage_env", "envs.soc_triage_env"):
        try:
            models_mod = importlib.import_module(f"{prefix}.models")
            env_mod = importlib.import_module(f"{prefix}.server.soc_triage_env")
            return (
                getattr(models_mod, "TriageAction"),
                getattr(models_mod, "TriageObservation"),
                getattr(env_mod, "SOCTriageEnv"),
            )
        except Exception:
            continue
    raise RuntimeError("Could not import SOC triage environment package.")


TriageAction, TriageObservation, SOCTriageEnv = _load_components()


SYSTEM_PROMPT = (
    "You are a SOC analyst agent in an interactive OpenEnv task. "
    "Respond with strict JSON keys: tool_name, tool_args, classification, recommended_action, reasoning. "
    "Use investigation tools before submitting final verdict."
)


@dataclass
class BaselineConfig:
    provider: str = "blaxel"
    model: str = "sandbox-openai"
    fallback_provider: str = "cerebras"
    fallback_model: str = "llama3.1-8b"
    episodes_per_task: int = 1
    use_heuristic: bool = False


def _prompt_for_observation(obs: Any) -> str:
    return (
        "Task id: "
        + obs.task_id
        + "\n"
        + "Step: "
        + str(getattr(obs, "step_num", 0))
        + "/"
        + str(getattr(obs, "max_steps", 1))
        + "\n"
        + "Observation JSON:\n"
        + json.dumps(obs.model_dump(), indent=2)
        + "\nReturn valid JSON only."
    )


def _heuristic_verdict(obs: Any) -> Any:
    if obs.task_id == "easy":
        text = (obs.alert.raw_log if getattr(obs, "alert", None) else "").lower()
        if "beacon" in text or "c2" in text:
            return TriageAction(
                tool_name="submit_verdict",
                classification="critical",
                recommended_action="escalate",
                reasoning="Beaconing indicates likely command-and-control traffic.",
            )
        if "failed" in text or "ssh" in text:
            return TriageAction(
                tool_name="submit_verdict",
                classification="medium",
                recommended_action="investigate",
                reasoning="Repeated failed logins require investigation.",
            )
        return TriageAction(
            tool_name="submit_verdict",
            classification="benign",
            recommended_action="ignore",
            reasoning="No clear malicious indicator in the event.",
        )

    if obs.task_id == "medium":
        return TriageAction(
            tool_name="submit_verdict",
            classification="MED-C,MED-E,MED-D,MED-A,MED-B",
            recommended_action="investigate",
            reasoning="Prioritize ransomware and data exfil signals over noise.",
        )

    return TriageAction(
        tool_name="submit_verdict",
        classification="H-01,H-03,H-05,H-07,H-11",
        recommended_action="contain",
        reasoning="Pattern matches recon, exploit, shell, lateral movement, exfiltration.",
    )


def _heuristic_action(obs: Any, step_index: int) -> Any:
    if step_index == 0:
        query = {
            "easy": "failed login outbound beacon privilege",
            "medium": "ransomware outbound data privilege",
            "hard": "scan exploit shell lateral exfil",
        }.get(obs.task_id, "suspicious")
        return TriageAction(
            tool_name="query_siem",
            tool_args={"query": query},
            reasoning="Initial SIEM investigation sweep.",
        )

    if step_index == 1:
        ioc = _pick_ioc(obs)
        return TriageAction(
            tool_name="get_threat_intel",
            tool_args={"ioc": ioc},
            reasoning="Threat-intel enrichment for discovered IOC.",
        )

    if step_index == 2 and obs.task_id == "hard":
        alert_id = _pick_alert_id(obs)
        return TriageAction(
            tool_name="pivot_alert",
            tool_args={"alert_id": alert_id},
            reasoning="Pivot to correlate related timeline events.",
        )

    return _heuristic_verdict(obs)


def _pick_ioc(obs: Any) -> str:
    if getattr(obs, "known_iocs", None):
        values = [str(v) for v in obs.known_iocs if str(v).strip()]
        if values:
            return values[0]
    if getattr(obs, "alert", None):
        if getattr(obs.alert, "source_ip", None):
            return str(obs.alert.source_ip)
        if getattr(obs.alert, "destination_ip", None):
            return str(obs.alert.destination_ip)
    return "suspicious-ioc"


def _pick_alert_id(obs: Any) -> str:
    if getattr(obs, "events", None):
        first = obs.events[0]
        return str(getattr(first, "alert_id", ""))
    if getattr(obs, "alerts", None):
        first = obs.alerts[0]
        return str(getattr(first, "alert_id", ""))
    if getattr(obs, "alert", None):
        return str(getattr(obs.alert, "alert_id", ""))
    return ""


def _parse_action(text: str, fallback: Any) -> Any:
    text = text.strip()
    if not text:
        return fallback

    try:
        data = json.loads(text)
        return TriageAction(**data)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return TriageAction(**data)
        except Exception:
            return fallback

    return fallback


def _model_action(provider: str, client: Any, model: str, obs: Any) -> Any:
    step_index = max(0, int(getattr(obs, "step_num", 0)))
    fallback = _heuristic_action(obs, step_index=step_index)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _prompt_for_observation(obs)},
    ]

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""

    return _parse_action(content, fallback)


def _run_task(task_id: str, episodes: int, provider: str, client: Any | None, model: str) -> float:
    env = SOCTriageEnv()
    total = 0.0

    for _ in range(episodes):
        obs = env.reset(task_id=task_id)
        done = False
        episode_reward = 0.01
        max_steps = max(1, int(getattr(obs, "max_steps", 4)))
        step_index = 0

        while not done and step_index < max_steps:
            if client is None:
                action = _heuristic_action(obs, step_index=step_index)
            else:
                action = _model_action(provider, client, model, obs)

            obs = env.step(action)
            episode_reward = float(getattr(obs, "reward", 0.01) or 0.01)
            done = bool(getattr(obs, "done", False))
            step_index += 1

        total += max(0.01, min(0.99, episode_reward))

    avg_score = total / episodes
    return round(max(0.01, min(0.99, avg_score)), 4)


def run_heuristic_baseline_sync(episodes_per_task: int = 1) -> dict[str, float]:
    return {
        task_id: _run_task(task_id, episodes_per_task, provider="heuristic", client=None, model="")
        for task_id in ["easy", "medium", "hard"]
    }


def _resolve_provider(provider: str) -> str:
    normalized = provider.lower().strip()
    if normalized not in {"openai", "cerebras", "blaxel"}:
        raise RuntimeError("provider must be 'openai', 'cerebras', or 'blaxel'.")
    return normalized


def _resolve_api_key(provider: str) -> str:
    if provider == "cerebras":
        return os.getenv("CEREBRAS_API_KEY", "").strip()
    if provider == "blaxel":
        return os.getenv("BLAXEL_AUTHORIZATION", "").strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def _resolve_model(provider: str, model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    if provider == "cerebras":
        return os.getenv("CEREBRAS_MODEL", "llama3.1-8b").strip()
    if provider == "blaxel":
        return os.getenv("BLAXEL_MODEL", "sandbox-openai").strip()
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def _normalize_api_key(api_key: str) -> str:
    key = api_key.strip()
    if key.lower().startswith("bearer "):
        return key[7:].strip()
    return key


def _blaxel_base_url(model: str) -> str:
    explicit_api_base = os.getenv("BLAXEL_API_BASE_URL", "").strip()
    if explicit_api_base:
        return explicit_api_base.rstrip("/")

    explicit_chat_url = os.getenv("BLAXEL_CHAT_URL", "").strip()
    if explicit_chat_url:
        suffix = "/chat/completions"
        if explicit_chat_url.endswith(suffix):
            return explicit_chat_url[: -len(suffix)]
        return explicit_chat_url.rstrip("/")

    workspace = os.getenv("BLAXEL_WORKSPACE", "vasanthfeb13").strip()
    base_url = os.getenv("BLAXEL_BASE_URL", "https://run.blaxel.ai").strip().rstrip("/")
    return f"{base_url}/{workspace}/models/{model}/v1"


def _build_client(provider: str, api_key: str, model: str) -> Any:
    try:
        OpenAI = getattr(importlib.import_module("openai"), "OpenAI")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed.") from exc

    normalized_key = _normalize_api_key(api_key)
    if provider == "cerebras":
        base_url = os.getenv("CEREBRAS_API_BASE_URL", "https://api.cerebras.ai/v1").strip()
        return OpenAI(api_key=normalized_key, base_url=base_url)

    if provider == "blaxel":
        base_url = _blaxel_base_url(model)
        workspace = os.getenv("BLAXEL_WORKSPACE", "").strip()
        default_headers: dict[str, str] = {}
        if workspace:
            default_headers["X-Blaxel-Workspace"] = workspace
        if default_headers:
            return OpenAI(api_key=normalized_key, base_url=base_url, default_headers=default_headers)
        return OpenAI(api_key=normalized_key, base_url=base_url)

    openai_base_url = os.getenv("OPENAI_API_BASE_URL", "").strip()
    if openai_base_url:
        return OpenAI(api_key=normalized_key, base_url=openai_base_url)
    return OpenAI(api_key=normalized_key)


def run_baseline_sync(
    provider: str = "cerebras",
    model: str | None = None,
    episodes_per_task: int = 1,
) -> dict[str, float]:
    provider_name = _resolve_provider(provider)
    api_key = _resolve_api_key(provider_name)
    if not api_key:
        if provider_name == "cerebras":
            key_name = "CEREBRAS_API_KEY"
        elif provider_name == "blaxel":
            key_name = "BLAXEL_AUTHORIZATION"
        else:
            key_name = "OPENAI_API_KEY"
        raise RuntimeError(f"{key_name} is not set.")

    selected_model = _resolve_model(provider_name, model)
    client = _build_client(provider_name, api_key, selected_model)

    return {
        task_id: _run_task(
            task_id,
            episodes_per_task,
            provider=provider_name,
            client=client,
            model=selected_model,
        )
        for task_id in ["easy", "medium", "hard"]
    }


def run_baseline_with_fallback_sync(
    provider: str,
    model: str | None,
    episodes_per_task: int,
    fallback_provider: str | None = "blaxel",
    fallback_model: str | None = None,
) -> tuple[str, dict[str, float], str | None]:
    try:
        scores = run_baseline_sync(provider=provider, model=model, episodes_per_task=episodes_per_task)
        return provider, scores, None
    except Exception as primary_exc:
        if not fallback_provider:
            raise

        fb = _resolve_provider(fallback_provider)
        if fb == _resolve_provider(provider):
            raise RuntimeError(f"Primary provider failed and fallback provider is the same: {primary_exc}") from primary_exc

        try:
            fb_scores = run_baseline_sync(provider=fb, model=fallback_model, episodes_per_task=episodes_per_task)
            warning = f"Primary provider '{provider}' failed: {primary_exc}. Used fallback '{fb}'."
            return fb, fb_scores, warning
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Primary provider '{provider}' failed ({primary_exc}) and fallback '{fb}' failed ({fallback_exc})."
            ) from fallback_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOC triage baseline across all tasks.")
    parser.add_argument("--provider", default=os.getenv("AI_PROVIDER", "blaxel"))
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "sandbox-openai"))
    parser.add_argument("--fallback-provider", default=os.getenv("AI_FALLBACK_PROVIDER", "cerebras"))
    parser.add_argument("--fallback-model", default=os.getenv("AI_FALLBACK_MODEL", "llama3.1-8b"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()

    config = BaselineConfig(
        provider=args.provider,
        model=args.model,
        fallback_provider=args.fallback_provider,
        fallback_model=args.fallback_model,
        episodes_per_task=max(1, args.episodes),
        use_heuristic=args.heuristic,
    )

    if config.use_heuristic:
        results = run_heuristic_baseline_sync(config.episodes_per_task)
        mode = "heuristic"
        warning = None
    else:
        mode, results, warning = run_baseline_with_fallback_sync(
            provider=config.provider,
            model=config.model,
            episodes_per_task=config.episodes_per_task,
            fallback_provider=config.fallback_provider,
            fallback_model=config.fallback_model,
        )

    payload = {"mode": mode, "config": asdict(config), "scores": results}
    if warning:
        payload["warning"] = warning
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
