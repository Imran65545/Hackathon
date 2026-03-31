import json
import os
from typing import Tuple, Dict, Any, Optional, List

from .models import EmailObservation, TriageAction, TriageReward

# We import the grade_action function which will be implemented in tasks.py (Step 5)
from .tasks import grade_action

class EmailTriageEnv:
    """
    OpenEnv-compliant environment for Email Triage.
    Loads synthetic emails and manages episode progression.
    """
    def __init__(self, task_id: int = 1, data_path: Optional[str] = None):
        self.task_id = task_id
        
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "emails.json")
            
        with open(data_path, "r", encoding="utf-8") as f:
            self.emails = json.load(f)
            
        self.current_index = 0
        self.cumulative_score = 0.0
        self.actions_taken: List[TriageAction] = []
        self.done = False

    def reset(self, task_id: Optional[int] = None) -> EmailObservation:
        """
        Resets to the first email of the current (or new) task.
        Returns the first observation.
        """
        if task_id is not None:
            if task_id not in [1, 2, 3]:
                raise ValueError("task_id must be 1, 2, or 3")
            self.task_id = task_id
            
        self.current_index = 0
        self.cumulative_score = 0.0
        self.actions_taken = []
        self.done = False
        
        return self._get_observation()

    def _get_observation(self) -> EmailObservation:
        """Helper to create an EmailObservation for the currently active email."""
        # If we've reached the end, return the last observation so we don't go out of bounds
        idx = min(self.current_index, len(self.emails) - 1)
        email = self.emails[idx]
        
        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            timestamp=email["timestamp"],
            task_id=self.task_id
        )

    def step(self, action: TriageAction) -> Tuple[EmailObservation, TriageReward, bool, Dict[str, Any]]:
        """
        Processes the action, computes the reward via tasks.py, and advances one step safely.
        Returns: next_observation, reward, done, info
        """
        if self.done:
            raise RuntimeError("Episode is already done. Please call reset().")
            
        # Get the ground truth to calculate score
        current_email = self.emails[self.current_index]
        ground_truth = current_email["ground_truth"]
        
        # We pass historical actions so the grader can check for identical repeated actions (penalties)
        reward = grade_action(self.task_id, action, ground_truth, self.actions_taken)
        
        # State updates
        self.cumulative_score += reward.score
        self.actions_taken.append(action)
        self.current_index += 1
        
        if self.current_index >= len(self.emails):
            self.done = True
            
        next_obs = self._get_observation()
        info = {
            "cumulative_score": self.cumulative_score,
            "progress": f"{self.current_index}/{len(self.emails)}"
        }
        
        return next_obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """
        Returns the full current state of the environment.
        """
        return {
            "current_task": self.task_id,
            "current_email_index": self.current_index,
            "total_emails": len(self.emails),
            "cumulative_score": self.cumulative_score,
            "actions_taken": [a.model_dump() for a in self.actions_taken],
            "done": self.done
        }
