"""Core environment logic for SOC triage tasks."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from soc_triage_env.models import AlertRecord, TriageAction, TriageObservation, TriageState
from soc_triage_env.server.graders import grade_easy, grade_hard, grade_medium
from soc_triage_env.server.tasks import TASKS, get_task


class SOCTriageEnv:
    """Deterministic SOC triage environment with three graded tasks."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        default_path = Path(__file__).parent / "data" / "alerts.json"
        self.dataset_path = Path(dataset_path) if dataset_path else default_path
        self.dataset = self._load_dataset()
        self._task_cursors = {task_id: 0 for task_id in TASKS}
        self._current_example: dict | None = None
        self._context_history: list[str] = []
        self._state = TriageState()

    def _load_dataset(self) -> dict:
        with self.dataset_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def reset(self, task_id: str = "easy") -> TriageObservation:
        task = get_task(task_id)
        examples = self.dataset.get(task_id, [])
        if not examples:
            raise ValueError(f"No dataset examples found for task '{task_id}'")

        cursor = self._task_cursors[task_id] % len(examples)
        self._task_cursors[task_id] += 1
        self._current_example = examples[cursor]
        self._context_history = []

        self._state = TriageState(
            episode_id=str(uuid4()),
            task_id=task_id,
            task_index=cursor,
            max_steps=int(task["max_steps"]),
            metadata={"difficulty": task["difficulty"]},
        )
        return self._build_observation(last_reward=0.0, done=False)

    def step(self, action: TriageAction) -> tuple[TriageObservation, float, bool, dict]:
        if self._current_example is None:
            raise RuntimeError("Environment has not been reset. Call reset(task_id=...) first.")
        if self._state.done:
            obs = self._build_observation(last_reward=0.0, done=True)
            return obs, 0.0, True, {"message": "Episode already complete"}

        self._state.step_count += 1
        base_score, feedback = self._grade_action(action)
        partial_credit = self._partial_credit(action)
        penalty = self._penalty(action)

        score = max(0.0, min(1.0, base_score + partial_credit - penalty))
        self._state.last_score = score
        self._state.total_reward += score

        max_steps = self._state.max_steps
        reached_confident_score = score >= 0.95
        reached_max_steps = self._state.step_count >= max_steps
        done = reached_confident_score or reached_max_steps
        self._state.done = done

        self._context_history.append(
            f"step={self._state.step_count} score={score:.3f} base={base_score:.3f} feedback={feedback}"
        )

        if done:
            self._context_history.append(
                f"episode_complete total_reward={self._state.total_reward:.3f}"
            )

        observation = self._build_observation(last_reward=score, done=done)
        info = {
            "base_score": round(base_score, 4),
            "partial_credit": round(partial_credit, 4),
            "penalty": round(penalty, 4),
            "feedback": feedback,
        }
        return observation, score, done, info

    @property
    def state(self) -> TriageState:
        return self._state

    @property
    def ground_truth(self) -> dict:
        if self._current_example is None:
            return {}
        return dict(self._current_example.get("ground_truth", {}))

    def _grade_action(self, action: TriageAction) -> tuple[float, str]:
        task_id = self._state.task_id
        truth = self.ground_truth

        if task_id == "easy":
            gt_severity = truth.get("severity", "benign")
            score = grade_easy(action.classification, gt_severity)
            return score, f"Expected severity={gt_severity}"

        if task_id == "medium":
            gt_ranking = truth.get("ranking", [])
            predicted_ranking = self._parse_id_list(action.classification)
            score = grade_medium(predicted_ranking, gt_ranking)
            return score, "Ranking quality scored via Kendall-tau"

        if task_id == "hard":
            gt_chain = truth.get("kill_chain", [])
            predicted_chain = self._parse_id_list(action.classification)
            score = grade_hard(predicted_chain, gt_chain)
            return score, "Kill-chain selection scored with F1"

        return 0.0, "Unsupported task"

    def _partial_credit(self, action: TriageAction) -> float:
        task_id = self._state.task_id
        truth = self.ground_truth

        if task_id == "easy":
            gt = truth.get("severity", "benign")
            pred = action.classification.lower().strip()
            if gt == "critical" and pred in {"high", "critical"}:
                return 0.1
            if gt == "high" and pred in {"medium", "high", "critical"}:
                return 0.08
            return 0.0

        if task_id == "medium":
            return 0.05 if action.recommended_action.lower().strip() in {"investigate", "escalate"} else 0.0

        if task_id == "hard":
            action_name = action.recommended_action.lower().strip()
            if action_name in {"contain", "eradicate", "escalate"}:
                return 0.08
            return 0.0

        return 0.0

    def _penalty(self, action: TriageAction) -> float:
        penalty = 0.0

        if self._state.step_count > 1:
            penalty += 0.05 * (self._state.step_count - 1)

        if self._state.task_id == "easy":
            truth = self.ground_truth
            gt_severity = truth.get("severity", "benign").lower().strip()
            rec = action.recommended_action.lower().strip()
            if gt_severity == "benign" and rec in {"block", "escalate"}:
                self._state.false_positives += 1
                penalty += 0.3
            if gt_severity in {"high", "critical"} and rec == "escalate":
                self._state.correct_escalations += 1

        return penalty

    def _build_observation(self, last_reward: float, done: bool) -> TriageObservation:
        task = get_task(self._state.task_id)
        alert_block = self._current_example.get("alert", {}) if self._current_example else {}
        difficulty = str(task["difficulty"])

        prompt = (
            "Review the provided security evidence, classify risk, and recommend response. "
            "Use concise reasoning grounded in the logs."
        )

        alert_record = None
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
            context_history=list(self._context_history),
            done=done,
            reward=round(last_reward, 4),
        )

    @staticmethod
    def _parse_id_list(value: str) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
