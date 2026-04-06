"""
FastAPI app for Email Triage OpenEnv
Exposes required endpoints: /baseline, /grader, /tasks
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from env import EmailTriageEnv, Action, Observation, Reward, TASKS

app = FastAPI(title="Email Triage OpenEnv")

# Global environment instance
current_env: Optional[EmailTriageEnv] = None
current_task: Optional[str] = None


class TaskInfo(BaseModel):
    id: str
    description: str
    action_schema: dict


class BaselineResult(BaseModel):
    task_id: str
    score: float


@app.post("/reset")
def reset(task_id: str = "easy") -> Observation:
    """Reset the environment for a specific task"""
    global current_env, current_task
    current_env = EmailTriageEnv(task_id=task_id)
    current_task = task_id
    return current_env.reset()


@app.post("/step")
def step(action: Action) -> dict:
    """Take a step in the environment"""
    if current_env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    obs, reward, done, info = current_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    """Get current environment state"""
    if current_env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return current_env.state()


@app.get("/tasks")
def get_tasks() -> list[TaskInfo]:
    """Return list of tasks and action schema"""
    action_schema = Action.model_json_schema()
    return [
        TaskInfo(
            id=task_id,
            description=task_config["description"],
            action_schema=action_schema,
        )
        for task_id, task_config in TASKS.items()
    ]


@app.get("/grader")
def get_grader_score() -> dict:
    """Return grader score after episode completion"""
    if current_env is None:
        raise HTTPException(status_code=400, detail="No episode in progress")

    if current_task is None:
        raise HTTPException(status_code=400, detail="No task selected")

    grader = TASKS[current_task]["grader"]
    score = grader(current_env)

    return {
        "task_id": current_task,
        "score": score,
        "processed_count": len(current_env.processed),
    }


@app.post("/baseline")
def run_baseline() -> list[BaselineResult]:
    """Trigger baseline inference and return scores for all tasks"""
    # In production, this would run the baseline.py script
    # For now, return placeholder that would be replaced with actual run
    from baseline import run_baseline as execute_baseline

    results = execute_baseline()
    return [
        BaselineResult(task_id=task_id, score=score)
        for task_id, score in results.items()
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
