from typing import Literal, Optional, Dict
from pydantic import BaseModel, Field

class EmailObservation(BaseModel):
    """
    Observation provided to the agent representing an incoming email.
    Includes the task_id to tell the agent which triage level it is solving.
    """
    email_id: str = Field(description="Unique identifier for the email.")
    subject: str = Field(description="Subject line of the email.")
    body: str = Field(description="The main text content of the email.")
    sender: str = Field(description="The email address or name of the sender.")
    timestamp: str = Field(description="The time the email was received (e.g., ISO format).")
    task_id: int = Field(description="The task ID (1, 2, or 3) indicating which triage task the agent should solve.")

class TriageAction(BaseModel):
    """
    The action the agent takes to triage the email. 
    Category and reply_draft are optional depending on the task level (Easy, Medium, or Hard).
    """
    email_id: str = Field(description="The ID of the email being triaged.")
    priority: Literal["urgent", "normal", "low"] = Field(description="The priority label assigned to the email.")
    category: Optional[Literal["HR", "Sales", "Tech", "Billing", "Other"]] = Field(default=None, description="The departmental category assigned to the email.")
    reply_draft: Optional[str] = Field(default=None, description="A drafted one-line professional reply.")

class TriageReward(BaseModel):
    """
    The reward and feedback given to the agent after taking a TriageAction.
    Uses partial credit depending on how many fields were correctly predicted.
    """
    score: float = Field(ge=0.0, le=1.0, description="The final numerical score between 0.0 and 1.0 inclusive.")
    breakdown: Dict[str, float] = Field(description="Detailed breakdown of the score (e.g. {'priority': 0.5, 'category': 0.3, 'reply': 0.2}).")
    feedback: str = Field(description="Human-readable explanation of the score and any penalties applied.")

# -----------------------------------------------------------------------------
# Quick test to validate models load correctly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Pydantic models for valid instantiation...\n")
    
    obs = EmailObservation(
        email_id="e001",
        subject="Server is down!",
        body="The main production database is unreachable. All clients are getting 502 errors.",
        sender="sysadmin@tech.hq",
        timestamp="2024-01-15T09:30:00",
        task_id=3
    )
    
    action = TriageAction(
        email_id="e001",
        priority="urgent",
        category="Tech",
        reply_draft="We are aware of the issue and our team is actively investigating the database outage."
    )
    
    reward = TriageReward(
        score=1.0,
        breakdown={"priority": 0.35, "category": 0.35, "reply": 0.30},
        feedback="Perfect score! Priority, category, and reply were all excellent and professional."
    )
    
    print("--- EmailObservation ---")
    print(obs.model_dump_json(indent=2))
    print("\n--- TriageAction ---")
    print(action.model_dump_json(indent=2))
    print("\n--- TriageReward ---")
    print(reward.model_dump_json(indent=2))
    print("\n✅ All models loaded and validated successfully!")
