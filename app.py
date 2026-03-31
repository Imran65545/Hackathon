import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from env.environment import EmailTriageEnv
from env.models import TriageAction

app = FastAPI(
    title="Email Triage",
    version="1.0.0"
)

env = EmailTriageEnv(task_id=1)

class ResetRequest(BaseModel):
    task_id: Optional[int] = None

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    try:
        task_id = req.task_id if req else None
        return env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: TriageAction):
    if env.done:
        raise HTTPException(status_code=400, detail="Episode done")
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
    return env.state()
