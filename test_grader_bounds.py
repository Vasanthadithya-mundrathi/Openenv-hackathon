#!/usr/bin/env python3
"""
Exhaustive boundary test for all graders, env step/reset, and inference output.

Validates the strict (0, 1) exclusive constraint that the OpenEnv validator enforces:
  - Every score/reward must be > 0.0 and < 1.0

Run:  python3 test_grader_bounds.py
"""

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup path so we can import the env even without pip install
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "envs"))

from soc_triage_env.server.graders import grade_easy, grade_medium, grade_hard, _clamp01
from soc_triage_env.server.soc_triage_env import SOCTriageEnv
from soc_triage_env.models import TriageAction

PASS = 0
FAIL = 0

def check(name: str, value: float) -> None:
    global PASS, FAIL
    if not isinstance(value, (int, float)):
        print(f"  FAIL  {name}: not a number ({value!r})")
        FAIL += 1
        return
    if value <= 0.0 or value >= 1.0:
        print(f"  FAIL  {name}: {value} is NOT in (0, 1)")
        FAIL += 1
    else:
        print(f"  PASS  {name}: {value}")
        PASS += 1


# ===== 1. grade_easy =====
print("\n===== grade_easy =====")
SEVERITY = ["benign", "low", "medium", "high", "critical"]
for pred in SEVERITY + ["unknown", "", "GARBAGE"]:
    for gt in SEVERITY + ["unknown", ""]:
        score = grade_easy(pred, gt)
        check(f"grade_easy({pred!r}, {gt!r})", score)

# ===== 2. grade_medium =====
print("\n===== grade_medium =====")
gt_ranking = ["A", "B", "C", "D", "E"]
test_cases_medium = [
    (gt_ranking, gt_ranking, "perfect"),
    (list(reversed(gt_ranking)), gt_ranking, "reversed"),
    ([], gt_ranking, "empty pred"),
    (["X", "Y", "Z"], gt_ranking, "no overlap"),
    (["A"], gt_ranking, "partial"),
    (gt_ranking, [], "empty gt"),
    ([], [], "both empty"),
    (["A"], ["A"], "single match"),
    (["B"], ["A"], "single mismatch"),
]
for pred, gt, label in test_cases_medium:
    score = grade_medium(pred, gt)
    check(f"grade_medium({label})", score)

# ===== 3. grade_hard =====
print("\n===== grade_hard =====")
gt_chain = ["H-01", "H-03", "H-05", "H-07", "H-11"]
test_cases_hard = [
    (gt_chain, gt_chain, "perfect"),
    ([], gt_chain, "empty pred"),
    (["WRONG-01", "WRONG-02"], gt_chain, "no overlap"),
    (["H-01"], gt_chain, "partial pred"),
    (gt_chain + ["EXTRA"], gt_chain, "superset pred"),
    (gt_chain, [], "empty gt"),
    ([], [], "both empty"),
    (["H-01", "H-03"], ["H-01", "H-03"], "perfect 2-item"),
]
for pred, gt, label in test_cases_hard:
    score = grade_hard(pred, gt)
    check(f"grade_hard({label})", score)

# ===== 4. _clamp01 =====
print("\n===== _clamp01 =====")
for val in [-1.0, -0.5, 0.0, 0.001, 0.5, 0.999, 1.0, 1.5, 2.0]:
    clamped = _clamp01(val)
    check(f"_clamp01({val})", clamped)

# ===== 5. SOCTriageEnv.reset() reward =====
print("\n===== SOCTriageEnv reset() =====")
env = SOCTriageEnv()
for task_id in ["easy", "medium", "hard"]:
    obs = env.reset(task_id=task_id)
    check(f"reset({task_id}).reward", obs.reward)

# ===== 6. SOCTriageEnv.step() reward =====
print("\n===== SOCTriageEnv step() rewards =====")
# Perfect actions
perfect_actions = {
    "easy": TriageAction(classification="critical", recommended_action="escalate", reasoning="test"),
    "medium": TriageAction(classification="MED-C,MED-E,MED-D,MED-A,MED-B", recommended_action="investigate", reasoning="test"),
    "hard": TriageAction(classification="H-01,H-03,H-05,H-07,H-11", recommended_action="contain", reasoning="test"),
}
# Worst actions
worst_actions = {
    "easy": TriageAction(classification="GARBAGE", recommended_action="ignore", reasoning="test"),
    "medium": TriageAction(classification="WRONG", recommended_action="ignore", reasoning="test"),
    "hard": TriageAction(classification="WRONG", recommended_action="ignore", reasoning="test"),
}
for label, actions in [("perfect", perfect_actions), ("worst", worst_actions)]:
    for task_id, action in actions.items():
        local_env = SOCTriageEnv()
        local_env.reset(task_id=task_id)
        obs = local_env.step(action)
        reward = obs.reward
        check(f"step({task_id}, {label}).reward", reward)
        check(f"step({task_id}, {label}).obs.reward", obs.reward)

# step() on already-done env
print("\n===== SOCTriageEnv step() already done =====")
for task_id in ["easy", "medium", "hard"]:
    local_env = SOCTriageEnv()
    local_env.reset(task_id=task_id)
    action = perfect_actions[task_id]
    local_env.step(action)  # first step completes it
    obs = local_env.step(action)  # call again after done
    reward = obs.reward
    check(f"step({task_id}, already_done).reward", reward)

# ===== 7. inference.py stdout =====
print("\n===== inference.py stdout =====")
result = subprocess.run(
    [sys.executable, str(ROOT / "inference.py"), "--episodes", "1"],
    capture_output=True, text=True, timeout=60, cwd=str(ROOT)
)
print(f"  inference.py exit code: {result.returncode}")
if result.returncode != 0:
    print(f"  STDERR: {result.stderr[:500]}")
    FAIL += 1
else:
    PASS += 1
    for line in result.stdout.strip().split("\n"):
        # Check [STEP] lines for reward values
        if line.startswith("[STEP]"):
            for part in line.split():
                if part.startswith("reward="):
                    val = float(part.split("=")[1])
                    check(f"stdout STEP reward", val)
        # Check [END] lines for score values
        elif line.startswith("[END]"):
            for part in line.split():
                if part.startswith("score="):
                    val = float(part.split("=")[1])
                    check(f"stdout END score", val)
                if part.startswith("rewards="):
                    rewards_str = part.split("=")[1]
                    if rewards_str:
                        for r in rewards_str.split(","):
                            if r.strip():
                                val = float(r.strip())
                                check(f"stdout END individual reward", val)
        # Check JSON summary
        elif line.startswith("{"):
            try:
                data = json.loads(line + "".join(
                    l for l in result.stdout.strip().split("\n")
                    if not l.startswith("[")
                ))
            except Exception:
                pass

# Parse the final JSON block
json_lines = []
in_json = False
for line in result.stdout.strip().split("\n"):
    if line.startswith("{"):
        in_json = True
    if in_json:
        json_lines.append(line)
if json_lines:
    try:
        summary = json.loads("\n".join(json_lines))
        for task_id, score_val in summary.get("scores", {}).items():
            check(f"JSON scores.{task_id}", score_val)
    except json.JSONDecodeError:
        print("  WARN  Could not parse JSON summary")

# ===== SUMMARY =====
print(f"\n{'='*50}")
print(f"  PASSED: {PASS}")
print(f"  FAILED: {FAIL}")
print(f"{'='*50}")

if FAIL > 0:
    print("\n  *** THERE ARE FAILURES — fix before pushing! ***\n")
    sys.exit(1)
else:
    print("\n  *** ALL TESTS PASS — safe to push! ***\n")
    sys.exit(0)
