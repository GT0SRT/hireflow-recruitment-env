# HireFlow Recruitment Simulator (OpenEnv)

HireFlow is a complete, real-world OpenEnv environment for training and evaluating AI recruiter agents.

The agent plays the role of a technical recruiter screening candidates for backend roles under enterprise constraints:
- limited profile API budget
- strict salary caps
- quality-focused shortlist objectives
- deterministic grading for reproducible benchmarks

This is not a toy benchmark. It models an operational hiring workflow used in real recruiting teams.

## Why this environment is useful

Most benchmarks do not evaluate tactical exploration under business constraints. HireFlow forces agents to balance:
- information gathering cost (profile fetch budget)
- role-skill alignment
- compensation feasibility
- precision of final shortlist decisions

This creates a practical evaluation loop for enterprise AI automation in HR operations.

## OpenEnv Spec Compliance

Typed Pydantic models are implemented in [env.py](env.py):
- `RecruitmentAction`
- `RecruitmentObservation`
- `RecruitmentReward`
- `RecruitmentState`

OpenEnv API methods:
- `reset(task_id)` -> initial `RecruitmentObservation`
- `step(action)` -> `(RecruitmentObservation, RecruitmentReward, done, info)`
- `state()` -> current `RecruitmentState`

FastAPI service in [app.py](app.py) exposes:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /health`

Metadata is defined in [openenv.yaml](openenv.yaml).

## Action Space

`RecruitmentAction` fields:
- `action_type`: one of `list_candidates`, `fetch_profile`, `run_interview`, `shortlist_candidate`, `reject_candidate`, `finish`
- `candidate_id`: required for candidate-targeted actions
- `rationale`: optional recruiter note

## Observation Space

`RecruitmentObservation` includes:
- task metadata: `task_id`, `difficulty`, `objective`
- episode metadata: `step_count`, `max_steps`, `done`
- budget state: `api_calls_used`, `api_budget`
- candidate data: `candidate_pool` (preview) and `fetched_profiles` (full profiles)
- interview data: `interview_results` keyed by candidate id (`pass` or `fail`)
- decisions: `shortlisted_ids`, `rejected_ids`
- learning signal: `progress` in `[0, 1]`
- feedback: `last_feedback`, `warnings`

## Reward Function (Dense + Partial Progress)

Per-step reward components include:
- small step penalty (`-0.01`) to discourage loops
- positive reward for useful discovery and first-time profile retrieval
- progress-delta shaping based on improved shortlist state
- penalties for invalid actions, duplicate fetches, and duplicate decisions
- strong penalty for API budget overrun
- penalty for shortlisting candidates violating salary cap

Terminal reward includes deterministic final grader score in `[0.0, 1.0]` plus efficiency bonus.

## Tasks and Deterministic Graders

Task data lives in [mock_data.py](mock_data.py).

1. `easy_task` (easy)
- Goal: shortlist one Python-ready backend candidate from 3 candidates.
- Constraint: API budget 4, salary cap 11 LPA.

2. `medium_task` (medium)
- Goal: precisely identify the strongest Python + FastAPI + SQL candidate.
- Constraint: API budget 5, salary cap 10 LPA.

3. `hard_task` (hard)
- Goal: optimize shortlist quality under strict salary and API constraints in a noisier candidate set.
- Constraint: API budget 6, salary cap 10 LPA.

Grader properties:
- deterministic rule-based logic
- outputs always clipped to `[0.0, 1.0]`
- combines shortlist F1, trajectory progress, budget compliance, salary compliance, efficiency

## Baseline Inference Script

Baseline script is [inference.py](inference.py) (root-level, as required).

Mandatory environment variables before submission:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The script uses the OpenAI client for model calls and evaluates all 3 tasks.

If `HF_TOKEN`/`MODEL_NAME` are missing locally, it falls back to a deterministic policy so you can still reproduce a stable baseline offline.

Run:

```bash
python inference.py
```

Example baseline output shape:

```json
{
  "scores": {
    "easy_task": 0.88,
    "medium_task": 0.86,
    "hard_task": 0.82
  },
  "mean_score": 0.8533
}
```

## Local Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Smoke test:

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy_task"}'
```

## Docker

Build:

```bash
docker build -t hireflow-openenv .
```

Run:

```bash
docker run --rm -p 7860:7860 hireflow-openenv
```

## Hugging Face Spaces (Docker)

1. Create a new Docker Space and push this repository.
2. Set Space tag to include `openenv`.
3. Set variables/secrets:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
4. Ensure the container starts and returns 200 for `/health`, then test `/reset`.

## Pre-Submission Checklist

- [x] 3+ tasks with deterministic graders and score range `[0.0, 1.0]`
- [x] Full OpenEnv interface (`reset`, `step`, `state`) with typed models
- [x] `openenv.yaml` present and aligned with implementation
- [x] Baseline script named [inference.py](inference.py) in repo root
- [x] Containerized with working [Dockerfile](Dockerfile)
- [x] Runtime suitable for low-resource machine targets (2 vCPU / 8 GB)
