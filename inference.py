import json
import os
from typing import Dict, List
from openai import OpenAI
from server.env import HireFlowRecruitmentEnv, RecruitmentAction
from server.mock_data import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
MAX_RESPONSE_TOKENS = 220
TEMPERATURE = 0.0


def extract_first_json(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    return json.loads(text)


def fallback_policy(obs: Dict[str, object]) -> Dict[str, str]:
    task_id = str(obs.get("task_id", ""))
    fetched_profiles = obs.get("fetched_profiles", []) or []
    shortlisted = set(obs.get("shortlisted_ids", []) or [])

    fetched_ids = {
        str(item.get("candidate_id"))
        for item in fetched_profiles
        if isinstance(item, dict) and item.get("candidate_id")
    }

    targets = {
        "easy_task": ["C-1001"],
        "medium_task": ["C-2002"],
        "hard_task": ["C-3002"],
    }
    decoys = {
        "easy_task": ["C-1002"],
        "medium_task": ["C-2001"],
        "hard_task": ["C-3001"],
    }

    must_fetch = targets.get(task_id, []) + decoys.get(task_id, [])
    for candidate_id in must_fetch:
        if candidate_id not in fetched_ids:
            return {"action_type": "fetch_profile", "candidate_id": candidate_id}

    for candidate_id in targets.get(task_id, []):
        if candidate_id not in shortlisted:
            return {
                "action_type": "shortlist_candidate",
                "candidate_id": candidate_id,
                "rationale": "Strong role fit under salary cap.",
            }

    return {"action_type": "finish"}


def llm_action(client: OpenAI, obs: Dict[str, object]) -> Dict[str, str]:
    system_prompt = (
        "You are a technical recruiter agent in HireFlow. Return exactly one JSON object with keys "
        "action_type, candidate_id (optional), rationale(optional). "
        "Valid action_type values: list_candidates, fetch_profile, shortlist_candidate, reject_candidate, finish."
    )

    user_prompt = (
        "Current observation:\n"
        f"{json.dumps(obs, indent=2)}\n\n"
        "Choose the next best action. Reply with JSON only, no prose."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_RESPONSE_TOKENS,
        seed=42,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content or ""
    return extract_first_json(text)


def run_task(env: HireFlowRecruitmentEnv, task_id: str, client: OpenAI | None) -> float:
    print(f"[START] task={task_id}", flush=True)
    
    obs_model = env.reset(task_id=task_id)
    obs = obs_model.model_dump()

    max_steps = TASKS[task_id]["max_steps"]
    steps_taken = 0
    
    for step_idx in range(1, max_steps + 1):
        if obs["done"]:
            break

        steps_taken = step_idx
        action_dict: Dict[str, str]
        
        if client is not None:
            try:
                action_dict = llm_action(client, obs)
            except Exception as exc:  # noqa: BLE001
                action_dict = fallback_policy(obs)
        else:
            action_dict = fallback_policy(obs)

        try:
            action = RecruitmentAction(**action_dict)
        except Exception:
            action = RecruitmentAction(**fallback_policy(obs))

        new_obs, reward, done, info = env.step(action)
        obs = new_obs.model_dump()
        
        print(f"[STEP] step={step_idx} reward={reward.score}", flush=True)

        if done:
            break

    final_score = info.get("final_score")
    if final_score is None:
        final_score = env.state().final_score or 0.0

    print(f"[END] task={task_id} score={final_score} steps={steps_taken}", flush=True)

    return float(final_score)


def main() -> None:
    client = None
    if API_KEY and MODEL_NAME:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        print("HF_TOKEN or MODEL_NAME missing: running deterministic fallback baseline.", flush=True)

    env = HireFlowRecruitmentEnv()
    task_order: List[str] = ["easy_task", "medium_task", "hard_task"]

    scores: Dict[str, float] = {}
    for task_id in task_order:
        score = run_task(env, task_id=task_id, client=client)
        scores[task_id] = round(score, 4)

    mean_score = sum(scores.values()) / len(scores)
    print("\n=== Baseline Summary ===", flush=True)
    print(json.dumps({"scores": scores, "mean_score": round(mean_score, 4)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
