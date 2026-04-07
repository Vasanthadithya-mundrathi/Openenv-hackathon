"""Deterministic graders for SOC triage tasks."""

from __future__ import annotations

SEVERITY_ORDER = ["benign", "low", "medium", "high", "critical"]

try:
    from scipy.stats import kendalltau as _kendalltau
except Exception:  # pragma: no cover
    _kendalltau = None


def _clamp01(value: float) -> float:
    return max(0.05, min(0.95, value))


def _kendall_tau_fallback(pred: list[int], truth: list[int]) -> float:
    """Simple Kendall-tau fallback when scipy is unavailable."""
    n = len(pred)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_sign = (pred[i] - pred[j])
            true_sign = (truth[i] - truth[j])
            if pred_sign == 0 or true_sign == 0:
                continue
            if (pred_sign > 0 and true_sign > 0) or (pred_sign < 0 and true_sign < 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def grade_easy(action_classification: str, ground_truth_severity: str) -> float:
    """Grade single-alert severity classification."""
    pred = action_classification.lower().strip()
    gt = ground_truth_severity.lower().strip()

    if pred == gt:
        return 0.95
    if pred not in SEVERITY_ORDER or gt not in SEVERITY_ORDER:
        return 0.05

    diff = abs(SEVERITY_ORDER.index(pred) - SEVERITY_ORDER.index(gt))
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.2
    return 0.05


def grade_medium(agent_ranking: list[str], ground_truth_ranking: list[str]) -> float:
    """Grade alert queue ranking with Kendall-tau normalized to [0,1]."""
    if not ground_truth_ranking:
        return 0.0

    n = len(ground_truth_ranking)
    pred_ranks: list[int] = []
    for alert_id in ground_truth_ranking:
        if alert_id in agent_ranking:
            pred_ranks.append(agent_ranking.index(alert_id))
        else:
            pred_ranks.append(n)

    truth_ranks = list(range(n))
    if _kendalltau is not None:
        tau, _ = _kendalltau(pred_ranks, truth_ranks)
        tau = float(tau) if tau is not None else 0.0
    else:
        tau = _kendall_tau_fallback(pred_ranks, truth_ranks)

    return round(_clamp01((tau + 1.0) / 2.0), 4)


def grade_hard(agent_selected: list[str], ground_truth_chain: list[str]) -> float:
    """Grade kill-chain event selection with F1 score."""
    truth = {x.strip() for x in ground_truth_chain if x.strip()}
    pred = {x.strip() for x in agent_selected if x.strip()}

    if not truth:
        return 0.0

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return round(_clamp01(f1), 4)
