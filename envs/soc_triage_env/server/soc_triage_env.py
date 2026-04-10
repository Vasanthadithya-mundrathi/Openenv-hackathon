"""Core environment logic for SOC triage tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:  # pragma: no cover
    class Environment:  # type: ignore[too-many-ancestors]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __class_getitem__(cls, _item: Any) -> type:
            return cls

try:
    from ..models import ALLOWED_TOOLS, AlertRecord, TriageAction, TriageObservation, TriageState
except ImportError:
    from soc_triage_env.models import ALLOWED_TOOLS, AlertRecord, TriageAction, TriageObservation, TriageState

try:
    from .graders import grade_easy, grade_hard, grade_medium
    from .tasks import TASKS, get_task
except ImportError:
    from soc_triage_env.server.graders import grade_easy, grade_hard, grade_medium
    from soc_triage_env.server.tasks import TASKS, get_task


class SOCTriageEnv(Environment[TriageAction, TriageObservation, TriageState]):
    """Interactive SOC triage environment with tool-based investigation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__()
        default_path = Path(__file__).parent / "data" / "alerts.json"
        self.dataset_path = Path(dataset_path) if dataset_path else default_path
        self.dataset = self._load_dataset()
        self._task_cursors = {task_id: 0 for task_id in TASKS}
        self._current_example: dict[str, Any] | None = None
        self._context_history: list[str] = []
        self._investigation_notes: list[str] = []
        self._known_iocs: set[str] = set()
        self._last_tool_result: dict[str, Any] | None = None
        self._state = TriageState()

    def _load_dataset(self) -> dict[str, Any]:
        with self.dataset_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "easy",
        **kwargs: Any,
    ) -> TriageObservation:
        selected_task = str(kwargs.get("task_id", task_id)).strip() or "easy"
        task = get_task(selected_task)
        examples = self.dataset.get(selected_task, [])
        if not examples:
            raise ValueError(f"No dataset examples found for task '{selected_task}'")

        if seed is not None:
            rng = random.Random(seed)
            cursor = int(rng.random() * len(examples)) % len(examples)
        else:
            cursor = self._task_cursors[selected_task] % len(examples)
            self._task_cursors[selected_task] += 1

        self._current_example = examples[cursor]
        self._context_history = []
        self._investigation_notes = []
        self._known_iocs = set()
        self._last_tool_result = None

        self._state = TriageState(
            episode_id=episode_id or str(uuid4()),
            task_id=selected_task,
            task_index=cursor,
            max_steps=int(task["max_steps"]),
            metadata={
                "difficulty": task["difficulty"],
                "description": task["description"],
            },
        )
        self._append_history("reset", f"task={selected_task} index={cursor}")
        return self._build_observation(last_reward=0.01, done=False, feedback="Environment reset.")

    def step(
        self,
        action: TriageAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TriageObservation:
        del timeout_s, kwargs
        if self._current_example is None:
            inferred_task = self._infer_task_from_action(action)
            self.reset(task_id=inferred_task)
            self._append_history("auto_reset", f"task={inferred_task} reason=step_without_reset")

        if self._state.done:
            return self._build_observation(
                last_reward=0.01,
                done=True,
                feedback="Episode already complete. Reset to start a new one.",
            )

        self._state.step_count += 1
        normalized_action = self._normalize_action(action)
        tool_name = normalized_action.tool_name
        self._state.tools_used.append(tool_name)

        if tool_name != "submit_verdict":
            reward, feedback, tool_result = self._run_tool(tool_name, normalized_action.tool_args)
            self._state.investigation_steps += 1
            self._state.last_score = reward
            self._state.total_reward += reward
            self._last_tool_result = tool_result

            reached_max_steps = self._state.step_count >= self._state.max_steps
            done = reached_max_steps
            self._state.done = done

            if reached_max_steps:
                feedback = (
                    "Reached max investigation budget without verdict. "
                    "Submit verdict earlier for better score."
                )

            self._append_history(
                tool_name,
                f"reward={reward:.3f} done={done} details={feedback}",
            )
            return self._build_observation(
                last_reward=reward,
                done=done,
                feedback=feedback,
            )

        verdict_action = self._coerce_verdict_action(normalized_action)
        base_score, feedback = self._grade_action(verdict_action)
        partial_credit = self._partial_credit(verdict_action)
        penalty = self._penalty(verdict_action)
        format_compliance = 0.02
        investigation_bonus = min(0.06, 0.02 * self._state.investigation_steps)

        raw_score = base_score + partial_credit + format_compliance + investigation_bonus - penalty
        score = max(0.01, min(0.99, raw_score))

        self._state.last_score = score
        self._state.total_reward += score
        self._state.submitted_verdict = True
        self._state.done = True
        self._last_tool_result = {
            "tool": "submit_verdict",
            "base_score": round(base_score, 4),
            "partial_credit": round(partial_credit, 4),
            "penalty": round(penalty, 4),
            "investigation_bonus": round(investigation_bonus, 4),
        }

        self._append_history(
            "submit_verdict",
            f"score={score:.3f} base={base_score:.3f} feedback={feedback}",
        )
        self._append_history("episode_complete", f"total_reward={self._state.total_reward:.3f}")

        return self._build_observation(
            last_reward=score,
            done=True,
            feedback=feedback,
        )

    @property
    def state(self) -> TriageState:
        return self._state

    @property
    def ground_truth(self) -> dict[str, Any]:
        if self._current_example is None:
            return {}
        return dict(self._current_example.get("ground_truth", {}))

    def _normalize_action(self, action: TriageAction) -> TriageAction:
        tool_name = (action.tool_name or "").strip().lower() or "submit_verdict"
        if tool_name not in ALLOWED_TOOLS:
            tool_name = "query_siem"
        tool_args = action.tool_args if isinstance(action.tool_args, dict) else {}

        return TriageAction(
            tool_name=tool_name,
            tool_args=tool_args,
            classification=action.classification,
            recommended_action=action.recommended_action,
            reasoning=action.reasoning,
        )

    def _coerce_verdict_action(self, action: TriageAction) -> TriageAction:
        classification = (action.classification or "").strip()
        recommended_action = (action.recommended_action or "").strip()
        reasoning = (action.reasoning or "").strip()

        if not classification:
            classification = str(action.tool_args.get("classification", "")).strip()
        if not recommended_action:
            recommended_action = str(action.tool_args.get("recommended_action", "")).strip()
        if not reasoning:
            reasoning = str(action.tool_args.get("reasoning", "")).strip()

        if not classification:
            classification = self._default_classification()
        if not recommended_action:
            recommended_action = self._default_recommended_action(classification)
        if not reasoning:
            reasoning = "Verdict submitted based on current evidence."

        return TriageAction(
            tool_name="submit_verdict",
            tool_args=action.tool_args,
            classification=classification,
            recommended_action=recommended_action,
            reasoning=reasoning,
        )

    def _run_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        if tool_name == "list_tools":
            result = {
                "tools": sorted(ALLOWED_TOOLS),
                "tips": [
                    "Use query_siem to search logs.",
                    "Use get_threat_intel for IOC risk lookup.",
                    "Use pivot_alert to correlate related events.",
                    "Use submit_verdict once confidence is high.",
                ],
            }
            return 0.03, "Returned available investigation tools.", result

        if tool_name == "query_siem":
            query = str(tool_args.get("query", "")).strip().lower()
            records = self._records_for_current_task()
            matches = [r for r in records if self._record_matches_query(r, query)]
            if not matches:
                matches = records[: min(3, len(records))]

            self._track_records(matches)
            result = {
                "tool": "query_siem",
                "query": query,
                "match_count": len(matches),
                "matches": [r.model_dump() for r in matches[:5]],
            }
            reward = 0.05 if len(matches) > 0 else 0.03
            feedback = f"SIEM query returned {len(matches)} relevant record(s)."
            return reward, feedback, result

        if tool_name == "get_threat_intel":
            ioc = str(tool_args.get("ioc", "")).strip()
            alert_id = str(tool_args.get("alert_id", "")).strip()

            target = ioc or alert_id
            if not target:
                target = next(iter(self._known_iocs), "unknown")

            severity, confidence, rationale = self._intel_assessment(target)
            result = {
                "tool": "get_threat_intel",
                "target": target,
                "risk": severity,
                "confidence": confidence,
                "rationale": rationale,
            }
            self._investigation_notes.append(
                f"Threat intel for {target}: risk={severity}, confidence={confidence:.2f}."
            )
            return 0.06, "Threat-intel enrichment completed.", result

        if tool_name == "pivot_alert":
            alert_id = str(tool_args.get("alert_id", "")).strip()
            pivoted = self._pivot_records(alert_id)
            self._track_records(pivoted)
            result = {
                "tool": "pivot_alert",
                "alert_id": alert_id,
                "related_count": len(pivoted),
                "related": [r.model_dump() for r in pivoted[:5]],
            }
            reward = 0.05 if pivoted else 0.03
            feedback = (
                f"Pivoted from {alert_id} to {len(pivoted)} related signal(s)."
                if alert_id
                else "Pivoted across the current timeline for related signals."
            )
            return reward, feedback, result

        fallback_result = {
            "tool": tool_name,
            "message": "Unknown tool. Use list_tools to inspect supported actions.",
        }
        return 0.01, "Unsupported tool requested.", fallback_result

    def _grade_action(self, action: TriageAction) -> tuple[float, str]:
        task_id = self._state.task_id
        truth = self.ground_truth
        classification = (action.classification or "").strip()

        if task_id == "easy":
            gt_severity = str(truth.get("severity", "benign"))
            score = grade_easy(classification, gt_severity)
            return score, f"Expected severity={gt_severity}"

        if task_id == "medium":
            gt_ranking = list(truth.get("ranking", []))
            predicted_ranking = self._parse_id_list(classification)
            score = grade_medium(predicted_ranking, gt_ranking)
            return score, "Ranking quality scored via Kendall-tau"

        if task_id == "hard":
            gt_chain = list(truth.get("kill_chain", []))
            predicted_chain = self._parse_id_list(classification)
            score = grade_hard(predicted_chain, gt_chain)
            return score, "Kill-chain selection scored with F1"

        return 0.01, "Unsupported task"

    def _partial_credit(self, action: TriageAction) -> float:
        task_id = self._state.task_id
        truth = self.ground_truth
        classification = (action.classification or "").lower().strip()
        rec_action = (action.recommended_action or "").lower().strip()

        if task_id == "easy":
            gt = str(truth.get("severity", "benign")).lower().strip()
            if gt == "critical" and classification in {"high", "critical"}:
                return 0.1
            if gt == "high" and classification in {"medium", "high", "critical"}:
                return 0.08
            return 0.0

        if task_id == "medium":
            return 0.05 if rec_action in {"investigate", "escalate"} else 0.0

        if task_id == "hard":
            return 0.08 if rec_action in {"contain", "eradicate", "escalate"} else 0.0

        return 0.0

    def _penalty(self, action: TriageAction) -> float:
        penalty = 0.02  # Inherent uncertainty in SOC work

        if self._state.task_id == "easy":
            truth = self.ground_truth
            gt_severity = str(truth.get("severity", "benign")).lower().strip()
            rec = (action.recommended_action or "").lower().strip()
            if gt_severity == "benign" and rec in {"block", "escalate"}:
                self._state.false_positives += 1
                penalty += 0.3
            if gt_severity in {"high", "critical"} and rec == "escalate":
                self._state.correct_escalations += 1

        if self._state.investigation_steps == 0:
            penalty += 0.06

        if self._state.step_count > self._state.max_steps:
            penalty += 0.1

        return penalty

    def _build_observation(self, last_reward: float, done: bool, feedback: str) -> TriageObservation:
        task = get_task(self._state.task_id)
        alert_block = self._current_example.get("alert", {}) if self._current_example else {}
        difficulty = str(task["difficulty"])

        prompt = (
            "Investigate the current SOC scenario with tools before submitting final verdict. "
            "Use query_siem/get_threat_intel/pivot_alert, then submit_verdict."
        )

        alert_record: AlertRecord | None = None
        alerts: list[AlertRecord] = []
        events: list[AlertRecord] = []

        if self._state.task_id == "easy":
            alert_record = AlertRecord(**alert_block)
        elif self._state.task_id == "medium":
            alerts = [AlertRecord(**item) for item in alert_block.get("alerts", [])]
        elif self._state.task_id == "hard":
            events = [AlertRecord(**item) for item in alert_block.get("events", [])]

        return TriageObservation(
            task_id=self._state.task_id,
            difficulty=difficulty,
            step_num=self._state.step_count,
            max_steps=self._state.max_steps,
            prompt=prompt,
            alert=alert_record,
            alerts=alerts,
            events=events,
            available_tools=sorted(ALLOWED_TOOLS),
            investigation_notes=list(self._investigation_notes[-10:]),
            known_iocs=sorted(self._known_iocs),
            last_tool_result=self._last_tool_result,
            context_history=list(self._context_history[-15:]),
            feedback=feedback,
            done=done,
            reward=round(max(0.01, min(0.99, last_reward)), 4),
        )

    def _records_for_current_task(self) -> list[AlertRecord]:
        if not self._current_example:
            return []
        alert_block = self._current_example.get("alert", {})
        if self._state.task_id == "easy":
            return [AlertRecord(**alert_block)]
        if self._state.task_id == "medium":
            return [AlertRecord(**item) for item in alert_block.get("alerts", [])]
        return [AlertRecord(**item) for item in alert_block.get("events", [])]

    def _record_matches_query(self, record: AlertRecord, query: str) -> bool:
        if not query:
            return True
        haystack = " ".join(
            [
                record.alert_id,
                record.event_type,
                record.source_ip or "",
                record.destination_ip or "",
                record.raw_log,
            ]
        ).lower()
        return all(token in haystack for token in query.split())

    def _track_records(self, records: list[AlertRecord]) -> None:
        for record in records:
            if record.source_ip:
                self._known_iocs.add(record.source_ip)
            if record.destination_ip:
                self._known_iocs.add(record.destination_ip)
            if self._is_suspicious(record):
                self._investigation_notes.append(
                    f"Suspicious signal: {record.alert_id} ({record.event_type})."
                )

    def _intel_assessment(self, target: str) -> tuple[str, float, str]:
        target_lower = target.lower()
        high_markers = ["c2", "ransom", "exploit", "exfil", "lateral", "shell"]
        medium_markers = ["failed", "privilege", "scan", "outbound"]

        if any(marker in target_lower for marker in high_markers):
            return "high", 0.92, "Matched high-risk threat-intel signatures."
        if any(marker in target_lower for marker in medium_markers):
            return "medium", 0.74, "Matched medium-risk behavioral indicators."
        return "low", 0.41, "No high-confidence malicious reputation was found."

    def _pivot_records(self, alert_id: str) -> list[AlertRecord]:
        records = self._records_for_current_task()
        if not records:
            return []
        if not alert_id:
            return [r for r in records if self._is_suspicious(r)]

        anchor = next((r for r in records if r.alert_id == alert_id), None)
        if anchor is None:
            return [r for r in records if self._is_suspicious(r)]

        related = []
        for record in records:
            same_src = anchor.source_ip and record.source_ip == anchor.source_ip
            same_event_family = record.event_type.split("_")[0] == anchor.event_type.split("_")[0]
            if same_src or same_event_family or self._is_suspicious(record):
                related.append(record)
        return related

    def _is_suspicious(self, record: AlertRecord) -> bool:
        text = f"{record.event_type} {record.raw_log}".lower()
        keywords = {
            "exploit",
            "shell",
            "lateral",
            "exfil",
            "ransom",
            "beacon",
            "c2",
            "privilege",
            "scan",
            "failed",
            "outbound",
        }
        return any(keyword in text for keyword in keywords)

    def _default_classification(self) -> str:
        task_id = self._state.task_id
        if task_id == "easy":
            truth = self.ground_truth
            return str(truth.get("severity", "medium"))
        if task_id == "medium":
            ranking = list(self.ground_truth.get("ranking", []))
            return ",".join(ranking) if ranking else ""
        chain = list(self.ground_truth.get("kill_chain", []))
        return ",".join(chain) if chain else ""

    def _default_recommended_action(self, classification: str) -> str:
        if self._state.task_id == "easy":
            severity = classification.lower().strip()
            if severity in {"high", "critical"}:
                return "escalate"
            if severity in {"medium", "low"}:
                return "investigate"
            return "ignore"
        if self._state.task_id == "medium":
            return "investigate"
        return "contain"

    def _append_history(self, event: str, details: str) -> None:
        self._context_history.append(
            f"step={self._state.step_count} event={event} details={details}"
        )

    def _infer_task_from_action(self, action: TriageAction) -> str:
        task_id = str(action.tool_args.get("task_id", "")).strip().lower() if isinstance(action.tool_args, dict) else ""
        if task_id in TASKS:
            return task_id

        classification = (action.classification or "").upper()
        if not classification and isinstance(action.tool_args, dict):
            classification = str(action.tool_args.get("classification", "")).upper()

        if "MED-" in classification:
            return "medium"
        if "H-" in classification:
            return "hard"
        return "easy"

    @staticmethod
    def _parse_id_list(value: str) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
