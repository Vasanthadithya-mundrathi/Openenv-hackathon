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

- classify alert severity
- prioritize alert queues
- identify multi-stage kill chains in noisy timelines

## Action Space

- classification: severity label or comma-separated ids (task dependent)
- recommended_action: operational response recommendation
- reasoning: explanation string

## Observation Space

- task_id, difficulty, step_num, max_steps
- prompt and context_history
- either alert, alerts, or events based on task
- reward and done flags

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

- base grader score
- plus partial credit for directionally correct response
- minus penalties for false positives and extra steps

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
