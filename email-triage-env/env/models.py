from typing import Literal, Optional, Dict
from pydantic import BaseModel, Field

class EmailObservation(BaseModel):
    email_id: str = Field(description="Unique identifier for the email.")
    subject: str = Field(description="Subject line of the email.")
    body: str = Field(description="The main text content of the email.")
    sender: str = Field(description="The email address or name of the sender.")
    timestamp: str = Field(description="The time the email was received (e.g., ISO format).")
    task_id: int = Field(description="The task ID (1, 2, or 3) indicating which triage task the agent should solve.")

class TriageAction(BaseModel):
    email_id: str = Field(description="The ID of the email being triaged.")
    priority: Literal["urgent", "normal", "low"] = Field(description="The priority label assigned to the email.")
    category: Optional[Literal["HR", "Sales", "Tech", "Billing", "Other"]] = Field(default=None, description="The departmental category assigned to the email.")
    reply_draft: Optional[str] = Field(default=None, description="A drafted one-line professional reply.")

class TriageReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="The final numerical score between 0.0 and 1.0 inclusive.")
    breakdown: Dict[str, float] = Field(description="Detailed breakdown of the score.")
    feedback: str = Field(description="Human-readable explanation of the score and any penalties applied.")
