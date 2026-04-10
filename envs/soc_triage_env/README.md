---
title: SOC Triage OpenEnv
emoji: "🛡️"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - cybersecurity
  - triage
---

## SOC Triage Environment

A real-world cybersecurity SOC analyst triage environment for agent training and evaluation.

## Overview

This environment simulates tier-1 SOC workflow where an agent must:

- investigate evidence with SOC tools
- classify alert severity only after evidence gathering
- prioritize alert queues under uncertainty
- identify multi-stage kill chains in noisy timelines

Architecture now follows OpenEnv reference patterns:

- `SOCTriageEnv` subclasses `Environment`
- server uses `create_app(...)` from `openenv-core`
- standard endpoints (`/reset`, `/step`, `/state`, `/schema`, `/ws`) are auto-generated
- custom endpoints (`/tasks`, `/grader`, `/baseline`) remain available

## Action Space

- `tool_name`: one of `list_tools`, `query_siem`, `get_threat_intel`, `pivot_alert`, `submit_verdict`
- `tool_args`: JSON object for tool parameters (query strings, IOC, alert id, etc.)
- `classification`: required for `submit_verdict`
- `recommended_action`: required for `submit_verdict`
- `reasoning`: analyst rationale for traceability

## Observation Space

- `task_id`, `difficulty`, `step_num`, `max_steps`
- `prompt` and `context_history`
- `available_tools` for guided interaction
- `investigation_notes`, `known_iocs`, `last_tool_result`
- task evidence (`alert` or `alerts` or `events`)
- `reward`, `done`, and `feedback`

## Tasks

- easy: single alert severity classification
- medium: alert queue prioritization
- hard: kill-chain event correlation

## Graders

- easy: severity-distance score
- medium: Kendall-tau rank correlation normalized to [0,1]
- hard: F1 score on selected kill-chain events

## Reward Design

Reward is shaped as:

- investigation step rewards for useful evidence gathering
- base grader score on `submit_verdict`
- partial credit for directionally correct outcomes
- investigation bonus when tool usage improves trace quality
- penalties for false positives and premature/no-investigation verdicts

## Run Locally

```bash
pip install -r requirements.txt
uvicorn soc_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t soc-triage-env:latest -f server/Dockerfile .
docker run -p 8000:8000 soc-triage-env:latest
```

## Required Endpoints

- POST /reset
- POST /step
- GET /state
- GET /tasks
- POST /grader
- POST /baseline
- GET /health

## Latest Verification Snapshot

Validated locally against the current code state:

- diagnostics: no workspace errors
- grader contract/bounds suite: `test_grader_bounds.py` passed (`127/127`)
- OpenEnv validator: ready for multi-mode deployment
- endpoint smoke checks: `/health`, `/schema`, `/reset`, `/step`, `/tasks`, `/grader`, `/logs` all returned HTTP 200

## Reference Alignment Summary

Compared against OpenEnv reference repos and templates (calendar, reasoning_gym, tbench2, carla, repl):

- app creation follows `create_app(...)` factory pattern used by reference env servers
- environment implementation follows `Environment` contract (`reset`, `step`, `state`)
- supports multi-turn progression and deterministic grading-friendly behavior
- custom SOC routes (`/tasks`, `/grader`, `/baseline`) are additive and do not break standard OpenEnv API shape

Intentional environment-specific extension:

- request-log capture middleware and `/logs` endpoint for triage/debug observability
