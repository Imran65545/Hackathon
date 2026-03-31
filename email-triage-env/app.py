import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from env.environment import EmailTriageEnv
from env.models import TriageAction

app = FastAPI(
    title="Email Triage Environment API",
    description="OpenEnv-compliant server for evaluating AI agents on an email triage task.",
    version="1.0.0"
)

# Initialize a central global environment instance. 
# Defaults to task 1, but agents can pass a task_id to /reset.
env = EmailTriageEnv(task_id=1)

class ResetRequest(BaseModel):
    task_id: Optional[int] = None

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    """Resets the environment and returns the initial observation."""
    try:
        task_id = req.task_id if req else None
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: TriageAction):
    """Processes an agent's action and returns the next OpenEnv step payload."""
    if env.done:
        raise HTTPException(
            status_code=400, 
            detail="Episode is already done. Please call /reset to start a new episode."
        )
    
    try:
        next_obs, reward, done, info = env.step(action)
        return {
            "observation": next_obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state():
    """Returns the macro state of the current environment session."""
    return env.state()
