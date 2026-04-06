import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import TriageAction

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo-1106")
HF_TOKEN = os.getenv("HF_TOKEN")

def call_llm(client: OpenAI, model: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant for email triage. Output JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except:
        return {}

def run_task(client: OpenAI, model: str, task_id: int):
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    done = False
    
    print(f"START Task {task_id}")
    
    while not done:
        if task_id == 1:
            instruction = "Classify priority: 'urgent', 'normal', or 'low'. Resp: {\"priority\": \"...\"}"
        elif task_id == 2:
            instruction = "Classify priority and category ('HR', 'Sales', 'Tech', 'Billing', 'Other'). Resp: {\"priority\": \"...\", \"category\": \"...\"}"
        elif task_id == 3:
            instruction = "Classify priority, category, and write a reply_draft. Resp: {\"priority\": \"...\", \"category\": \"...\", \"reply_draft\": \"...\"}"
            
        prompt = f"{instruction}\n\nSubject: {obs.subject}\nBody: {obs.body}"
        
        result_json = call_llm(client, model, prompt)
        
        raw_pri = result_json.get("priority", "low").lower()
        pri = raw_pri if raw_pri in ["urgent", "normal", "low"] else "low"
        
        raw_cat = result_json.get("category")
        cat = None
        if raw_cat:
            cat_map = {"hr": "HR", "sales": "Sales", "tech": "Tech", "billing": "Billing", "other": "Other"}
            cat = cat_map.get(raw_cat.lower(), "Other")
            
        reply = result_json.get("reply_draft")
        
        try:
            action = TriageAction(
                email_id=obs.email_id,
                priority=pri,
                category=cat,
                reply_draft=reply
            )
        except:
            action = TriageAction(email_id=obs.email_id, priority="low")
            
        obs, reward, done, info = env.step(action)
        print(f"STEP: [{info['progress']}] {action.email_id} | Score: {reward.score}")

    score = env.cumulative_score / env.state()["total_emails"]
    print(f"END Task {task_id} Avg Score: {score:.3f}")

if __name__ == "__main__":
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    for tid in [1, 2, 3]:
        run_task(client, MODEL_NAME, tid)
