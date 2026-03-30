# SOC Triage OpenEnv Environment

Real-world cybersecurity SOC triage environment for agent training and evaluation, built for the Meta x PyTorch x Scaler OpenEnv Hackathon.

## Why This Environment

SOC analysts triage large volumes of alerts under time pressure. This environment simulates three high-value workflows:

- single-alert severity classification
- queue prioritization under noise
- multi-event kill-chain correlation

The environment is deterministic, reproducible, and exposes OpenEnv-style reset, step, and state APIs plus required hackathon endpoints.

## Environment Structure

- package root: envs/soc_triage_env
- API server: envs/soc_triage_env/server/app.py
- core env logic: envs/soc_triage_env/server/soc_triage_env.py
- tasks and graders: envs/soc_triage_env/server/tasks.py, envs/soc_triage_env/server/graders.py
- synthetic dataset: envs/soc_triage_env/server/data/alerts.json
- baseline runner: baseline.py

## Action Space

| Field | Type | Description |
| --- | --- | --- |
| classification | string | Severity label or comma-separated alert ids, depending on task |
| recommended_action | string | Operational response decision |
| reasoning | string | Analyst rationale for traceability |

## Observation Space

| Field | Type | Description |
| --- | --- | --- |
| task_id | string | Current task id: easy, medium, hard |
| difficulty | string | Difficulty label |
| step_num | int | Current step count |
| max_steps | int | Episode step budget |
| prompt | string | Task prompt for agent |
| alert | object or null | Single alert payload for easy task |
| alerts | list | Alert queue payload for medium task |
| events | list | Event timeline payload for hard task |
| context_history | list[string] | Previous step summaries |
| done | bool | Episode completion flag |
| reward | float | Last step reward |

## Tasks and Graders

| Task | Difficulty | Goal | Grader |
| --- | --- | --- | --- |
| easy | easy | classify one alert severity | severity-distance score |
| medium | medium | prioritize 5 alerts by urgency | Kendall-tau normalized to [0,1] |
| hard | hard | select kill-chain events in noisy timeline | F1 score on selected ids |

## Reward Design

Reward is shaped per step (not binary):

- base score from grader
- partial credit for directionally correct actions
- false-positive penalty for aggressive wrong actions
- loop penalty for excessive steps

## Required Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| /health | GET | service health |
| /reset | POST | start episode and return initial observation |
| /step | POST | apply action and return reward plus next observation |
| /state | GET | return current internal state |
| /tasks | GET | list task metadata and schemas |
| /grader | POST | evaluate action against provided or active ground truth |
| /baseline | POST | run baseline with provider + fallback chain |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn soc_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

## Baseline Providers

Default strategy:

- primary: blaxel (model: sandbox-openai)
- fallback: cerebras (model: llama3.1-8b)

```bash
# Primary provider (Blaxel)
export BLAXEL_WORKSPACE=your_workspace
export BLAXEL_AUTHORIZATION=your_blaxel_token
# optional: export BLAXEL_AUTHORIZATION="Bearer your_blaxel_token"

# Fallback provider (Cerebras)
export CEREBRAS_API_KEY=your_cerebras_key

# Run baseline with defaults
python baseline.py --episodes 1

# Explicit provider chain
python baseline.py --provider blaxel --model sandbox-openai --episodes 1 \
  --fallback-provider cerebras --fallback-model llama3.1-8b
```

OpenAI-compatible alternative:

```bash
export OPENAI_API_KEY=your_openai_key
python baseline.py --provider openai --model gpt-4o-mini --episodes 1
```

Heuristic smoke baseline:

```bash
python baseline.py --heuristic
```

## Submission Inference Script (Mandatory)

For submission, use the root-level inference script:

- script path: inference.py
- required environment variables:
  - API_BASE_URL
  - MODEL_NAME
  - HF_TOKEN
- all LLM calls are made via OpenAI client

Example with Blaxel:

```bash
export API_BASE_URL="https://run.blaxel.ai/vasanthfeb13/models/sandbox-openai/v1"
export MODEL_NAME="sandbox-openai"
export HF_TOKEN="your_token"
export BLAXEL_WORKSPACE="vasanthfeb13"

python inference.py --episodes 1 --max-minutes 20
```

Example with any OpenAI-compatible endpoint:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-4o-mini"
export HF_TOKEN="your_hf_token"

python inference.py --episodes 1 --max-minutes 20
```

## Baseline Scores (Sample, 28 Mar 2026)

Sample run used: 1 episode per task, provider blaxel, model sandbox-openai.

| Task | Score |
| --- | --- |
| easy | 0.5 |
| medium | 1.0 |
| hard | 0.0 |

These numbers are expected to vary by provider/model and prompt behavior.

## Docker

```bash
docker build -t soc-triage-env:latest -f envs/soc_triage_env/server/Dockerfile .
docker run -d --name soc-test -p 8010:8000 \
  -e BLAXEL_WORKSPACE=your_workspace \
  -e BLAXEL_AUTHORIZATION=your_token \
  soc-triage-env:latest

curl http://127.0.0.1:8010/health
curl -X POST http://127.0.0.1:8010/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
```

## OpenEnv Validation

```bash
openenv validate envs/soc_triage_env/
```

Current local result: pass.

## Hugging Face Deployment

CLI path:

```bash
huggingface-cli login
openenv push --repo-id vasanthfeb13/soc-triage-env
```

After deployment, verify:

```bash
curl -X POST https://vasanthfeb13-soc-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'
```

## Submission Artifacts Checklist

- public GitHub repository
- requirements.txt
- baseline.py
- README.md
- envs/soc_triage_env/openenv.yaml
- deployed HF Space URL
