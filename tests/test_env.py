import pytest
import sys
import os

# Ensure the parent directory is loaded natively so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import TriageAction
from pydantic import ValidationError

@pytest.fixture
def env_t1():
    return EmailTriageEnv(task_id=1)

@pytest.fixture
def env_t2():
    return EmailTriageEnv(task_id=2)

@pytest.fixture
def env_t3():
    return EmailTriageEnv(task_id=3)

def test_reset(env_t1):
    obs = env_t1.reset()
    assert obs is not None
    assert obs.email_id.startswith("e")
    assert obs.task_id == 1
    
    # Test dynamic task switching
    obs2 = env_t1.reset(task_id=2)
    assert env_t1.task_id == 2
    assert obs2.task_id == 2
    assert env_t1.current_index == 0

def test_step_task_1(env_t1):
    obs = env_t1.reset()
    gt_priority = env_t1.emails[0]["ground_truth"]["priority"]
    action = TriageAction(email_id=obs.email_id, priority=gt_priority)
    
    next_obs, reward, done, info = env_t1.step(action)
    assert reward.score == 1.0
    assert reward.breakdown["priority"] == 1.0
    assert done is False
    assert env_t1.current_index == 1

def test_step_task_2(env_t2):
    obs = env_t2.reset()
    gt_priority = env_t2.emails[0]["ground_truth"]["priority"]
    gt_cat = env_t2.emails[0]["ground_truth"]["category"]
    
    action = TriageAction(email_id=obs.email_id, priority=gt_priority, category=gt_cat)
    
    next_obs, reward, done, info = env_t2.step(action)
    assert reward.score == 1.0
    assert reward.breakdown["priority"] == 0.5
    assert reward.breakdown["category"] == 0.5
    assert "correct" in reward.feedback.lower()

def test_step_task_3(env_t3):
    obs = env_t3.reset()
    gt = env_t3.emails[0]["ground_truth"]
    
    # Send an ideal professional action mapping reality exactly
    action = TriageAction(email_id=obs.email_id, priority=gt["priority"], category=gt["category"], reply_draft=gt["ideal_reply"])
    next_obs, reward, done, info = env_t3.step(action)
    
    # Priority (0.35) + Category (0.35) + length overlap & tone (~0.30)
    assert reward.score >= 0.85
    assert done is False

def test_reward_edge_cases(env_t3):
    obs = env_t3.reset()
    
    # 1. Edge Case: Empty task 3 fields (Missing Category and Reply Draft)
    action_1 = TriageAction(email_id=obs.email_id, priority="low") 
    _, reward, _, _ = env_t3.step(action_1)
    
    # Should penalize -0.1 for missing category and -0.1 for missing reply at Task 3
    assert "Missing category" in reward.feedback
    assert "Missing reply draft" in reward.feedback
    assert reward.score < 0.5 
    
    # 2. Edge Case: Repeated identical actions across different emails
    _, reward_dup, _, _ = env_t3.step(action_1)
    
    # Since priority is identical and optional fields are missing exactly, it detects repetition
    assert "Repeated identical action" in reward_dup.feedback

def test_wrong_types_pydantic():
    # Pydantic block catches garbage payload entirely before env is stepped
    with pytest.raises(ValidationError):
        # Priority must be urgent, normal, or low; passing 'super-urgent' fails schema
        action = TriageAction(email_id="x123", priority="super-urgent")

def test_state_and_done(env_t1):
    env_t1.reset()
    emails_count = len(env_t1.emails)
    
    # Step through every email
    for i in range(emails_count):
        # Using a varied action payload to avoid repeated loop penalties
        action = TriageAction(email_id=f"e{i:03d}", priority="urgent" if i % 2 == 0 else "low")
        _, _, done, _ = env_t1.step(action)
        
    assert done is True
    assert env_t1.state()["done"] is True
    assert len(env_t1.state()["actions_taken"]) == emails_count
    
    # Attempting to continue a broken loop errors cleanly
    with pytest.raises(RuntimeError) as exc_info:
        env_t1.step(TriageAction(email_id="too_late", priority="low"))
    assert "Episode is already done" in str(exc_info.value)
