import os
import sys
import json
from openai import OpenAI

# Add the parent directory to sys.path so 'env' module can be imported properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import TriageAction

def call_llm(client: OpenAI, prompt: str) -> dict:
    """Calls OpenAI API and safely parses the expected JSON response."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106", # Ensure json_object format is supported
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with triaging corporate emails. You must output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}

def run_task(client: OpenAI, task_id: int):
    """Executes a full evaluation pass for the given task level."""
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    done = False
    
    print(f"\n=============================================")
    print(f"--- Running Task {task_id} Evaluation ---")
    print(f"=============================================")
    
    while not done:
        # Build strict prompts so the baseline has a fair chance to succeed
        if task_id == 1:
            instruction = "Classify this email's priority as exactly one of: 'urgent', 'normal', or 'low'. Output JSON strictly matching key: {\"priority\": \"...\"}"
        elif task_id == 2:
            instruction = "Classify priority ('urgent', 'normal', 'low') and department category ('HR', 'Sales', 'Tech', 'Billing', 'Other'). Output JSON strictly matching keys: {\"priority\": \"...\", \"category\": \"...\"}"
        elif task_id == 3:
            instruction = "Classify priority ('urgent', 'normal', 'low'), category ('HR', 'Sales', 'Tech', 'Billing', 'Other'), and write a professional reply strictly under key 'reply_draft'. Output JSON strictly matching keys: {\"priority\": \"...\", \"category\": \"...\", \"reply_draft\": \"...\"}"
            
        prompt = f"""
{instruction}

Email ID: {obs.email_id}
Sender: {obs.sender}
Subject: {obs.subject}
Body: {obs.body}
Timestamp: {obs.timestamp}
"""
        
        # Call LLM and get parsed dictionary
        result_json = call_llm(client, prompt)
        
        # Clean data mapped to literal requirements of Pydantic models
        raw_pri = result_json.get("priority", "low").lower()
        pri = raw_pri if raw_pri in ["urgent", "normal", "low"] else "low"
        
        raw_cat = result_json.get("category")
        cat = None
        if raw_cat:
            cat_map = {"hr": "HR", "sales": "Sales", "tech": "Tech", "billing": "Billing", "other": "Other"}
            cat = cat_map.get(raw_cat.lower(), "Other")
            
        reply = result_json.get("reply_draft")
        
        # Construct validated payload
        try:
            action = TriageAction(
                email_id=obs.email_id,
                priority=pri,
                category=cat,
                reply_draft=reply
            )
        except Exception as e:
            # Absolute fallback if LLM breaks Pydantic validation (e.g. types)
            action = TriageAction(email_id=obs.email_id, priority="low")
            
        # Execute Environment Step
        obs, reward, done, info = env.step(action)
        print(f"[{info['progress']}] Email {action.email_id} | Score: {reward.score:.2f} | Pri: {action.priority} | Feedback: {reward.feedback}")

    # Calculate average score for the full task
    total_emails = env.state()["total_emails"]
    final_score = env.cumulative_score
    average_score = final_score / total_emails
    
    print(f"\n✅ Task {task_id} Completed!")
    print(f"🏆 Final Average Score: {average_score:.3f} / 1.000\n")

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: 'OPENAI_API_KEY' environment variable is missing.")
        print("Please set it before running: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key)
    print("Starting Central Baseline Inference Evaluation...")
    
    for tid in [1, 2, 3]:
        run_task(client, tid)
        
    print("--- All Evaluation Tasks Completed Successfully ---")
