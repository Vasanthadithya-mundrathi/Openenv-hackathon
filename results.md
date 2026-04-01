# Results Report

Generated at (UTC): 2026-04-01 03:33:56Z
Target Space: https://vasanthfeb13-soc-triage-env.hf.space

## Judge-Style Checklist

- HF Space deploys and responds: PASS
  - GET /: 200
  - GET /health: 200
  - POST /reset (no body): 200

- OpenEnv spec compliance: PASS
  - Command: openenv validate envs/soc_triage_env
  - Result: [OK] soc_triage: Ready for multi-mode deployment

- Inference script runs and returns scores: PASS
  - Command: python inference.py --episodes 1 --max-minutes 3
  - Runtime: 2.52s
  - Scores: easy=1.0, medium=1.0, hard=1.0

- Dockerfile builds: PASS
  - Build tag: openenv-soc-triage-check:results-md
  - Build result: success
  - Container smoke test: /health=200, /reset(no body)=200

- 3+ tasks with graders and scores in [0.0, 1.0]: PASS
  - Tasks present: easy, medium, hard
  - Grader status: 200 for all
  - Score range check: valid for all

## SOC Task Test Runs (3 tests)

Action set used:
- easy: classification=critical, recommended_action=escalate
- medium: classification=MED-C,MED-E,MED-D,MED-A,MED-B
- hard: classification=H-01,H-03,H-05,H-07,H-11

Results:
- easy
  - reset_status: 200
  - step_status: 200
  - steps_taken: 1
  - done: true
  - total_reward: 1.0
  - grader_status: 200
  - grader_score: 1.0
  - feedback: Expected severity=critical

- medium
  - reset_status: 200
  - step_status: 200
  - steps_taken: 1
  - done: true
  - total_reward: 1.0
  - grader_status: 200
  - grader_score: 1.0
  - feedback: Ranking quality scored via Kendall-tau

- hard
  - reset_status: 200
  - step_status: 200
  - steps_taken: 1
  - done: true
  - total_reward: 1.0
  - grader_status: 200
  - grader_score: 1.0
  - feedback: Kill-chain selection scored with F1

## Backend Provider Status (live baseline endpoint)

- blaxel primary: FALLBACK CURRENTLY ACTIVE
  - mode returned: cerebras
  - warning: insufficient_quota on blaxel key, fallback used
  - scores: easy=0.0, medium=1.0, hard=0.0

- cerebras primary: PASS
  - mode returned: cerebras
  - warning: none
  - scores: easy=0.0, medium=1.0, hard=0.0

## Overall Situation

- Submission-critical platform checks are passing.
- Cerebras backend is working and returning live SOC results.
- Blaxel route is configured but currently falling back due quota limits on the key.
