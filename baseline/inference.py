import os
import sys
import json
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import TriageAction

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
    
    print(f"Task {task_id} Start")
    
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
        print(f"[{info['progress']}] {action.email_id} | Score: {reward.score}")

    score = env.cumulative_score / env.state()["total_emails"]
    print(f"Task {task_id} Avg Score: {score:.3f}")

if __name__ == "__main__":
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model_name = "llama-3.3-70b-versatile"
    elif openai_key:
        client = OpenAI(api_key=openai_key)
        model_name = "gpt-3.5-turbo-1106"
    else:
        sys.exit(1)
    
    for tid in [1, 2, 3]:
        run_task(client, model_name, tid)
