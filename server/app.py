from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.env import HireFlowRecruitmentEnv, RecruitmentAction
from server.mock_data import TASKS


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_task")


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


app = FastAPI(
    title="HireFlow Recruitment Simulator",
    description="Real-world hiring workflow simulation for AI recruiter agents",
    version="1.0.0",
)

env = HireFlowRecruitmentEnv()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "hireflow-recruitment-openenv",
        "status": "ok",
        "tasks": list(TASKS.keys()),
        "api": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": TASKS}


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None) -> Dict[str, Any]:
    task_id = payload.task_id if payload else "easy_task"
    
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    observation = env.reset(task_id=task_id)
    return {"observation": observation.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(action: RecruitmentAction) -> StepResponse:
    observation, reward, done, info = env.step(action)
    return StepResponse(
        observation=observation.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    return {"state": env.state().model_dump()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()