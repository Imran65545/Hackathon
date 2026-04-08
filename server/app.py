import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from env.environment import EmailTriageEnv
from env.models import TriageAction


app = FastAPI(
    title="Email Triage",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>📬 Email Triage AI Environment</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --primary: #6366f1;
                    --secondary: #4f46e5;
                    --background: #0f172a;
                    --card: rgba(255, 255, 255, 0.05);
                    --text: #f8fafc;
                    --text-dim: #94a3b8;
                }
                body {
                    background-color: var(--background);
                    color: var(--text);
                    font-family: 'Inter', sans-serif;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    overflow: hidden;
                    background-image: 
                        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                        radial-gradient(at 100% 100%, rgba(79, 70, 229, 0.15) 0px, transparent 50%);
                }
                .container {
                    background: var(--card);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    padding: 3rem;
                    border-radius: 24px;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                    max-width: 500px;
                    text-align: center;
                    animation: fadeIn 0.8s ease-out;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                h1 {
                    font-family: 'Outfit', sans-serif;
                    font-weight: 800;
                    font-size: 2.5rem;
                    margin-bottom: 0.5rem;
                    background: linear-gradient(to right, #818cf8, #c084fc);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                p {
                    color: var(--text-dim);
                    line-height: 1.6;
                    margin-bottom: 2.5rem;
                }
                .btn-group {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                .btn {
                    text-decoration: none;
                    background: var(--primary);
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 12px;
                    font-weight: 600;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.5rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .btn:hover {
                    transform: translateY(-2px);
                    background: var(--secondary);
                    box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
                }
                .btn-secondary {
                    background: rgba(255, 255, 255, 0.05);
                    color: var(--text);
                }
                .btn-secondary:hover {
                    background: rgba(255, 255, 255, 0.1);
                    box-shadow: none;
                }
                .badge {
                    display: inline-block;
                    padding: 0.25rem 0.75rem;
                    background: rgba(99, 102, 241, 0.1);
                    color: #818cf8;
                    border-radius: 9999px;
                    font-size: 0.875rem;
                    font-weight: 600;
                    margin-bottom: 1rem;
                }
                .status-dot {
                    height: 8px;
                    width: 8px;
                    background-color: #22c55e;
                    border-radius: 50%;
                    display: inline-block;
                    margin-right: 6px;
                    box-shadow: 0 0 8px #22c55e;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="badge"><span class="status-dot"></span> System Live v1.0.0</div>
                <h1>Email Triage AI</h1>
                <p>Welcome to the OpenEnv Email Triage Training Environment. A specialized API for training and testing corporate triage agents.</p>
                <div class="btn-group">
                    <a href="/docs" class="btn">
                        📘 Explore API Documentation
                    </a>
                    <a href="/state" class="btn btn-secondary">
                        📊 Check Environment State
                    </a>
                </div>
            </div>
        </body>
    </html>
    """


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

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
