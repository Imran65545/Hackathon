import pytest
import sys
import os
from pydantic import ValidationError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import TriageAction

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

def test_step_task_2(env_t2):
    obs = env_t2.reset()
    gt_priority = env_t2.emails[0]["ground_truth"]["priority"]
    gt_cat = env_t2.emails[0]["ground_truth"]["category"]
    action = TriageAction(email_id=obs.email_id, priority=gt_priority, category=gt_cat)
    next_obs, reward, done, info = env_t2.step(action)
    assert reward.score == 1.0
    assert reward.breakdown["priority"] == 0.5
    assert reward.breakdown["category"] == 0.5

def test_step_task_3(env_t3):
    obs = env_t3.reset()
    gt = env_t3.emails[0]["ground_truth"]
    action = TriageAction(email_id=obs.email_id, priority=gt["priority"], category=gt["category"], reply_draft=gt["ideal_reply"])
    next_obs, reward, done, info = env_t3.step(action)
    assert reward.score >= 0.85

def test_reward_edge_cases(env_t3):
    obs = env_t3.reset()
    action_1 = TriageAction(email_id=obs.email_id, priority="low") 
    _, reward, _, _ = env_t3.step(action_1)
    assert "Missing category" in reward.feedback
    assert "Missing reply draft" in reward.feedback
    assert reward.score < 0.5 
    _, reward_dup, _, _ = env_t3.step(action_1)
    assert "Repeated identical action" in reward_dup.feedback

def test_wrong_types_pydantic():
    with pytest.raises(ValidationError):
        TriageAction(email_id="x123", priority="super-urgent")

def test_state_and_done(env_t1):
    env_t1.reset()
    emails_count = len(env_t1.emails)
    for i in range(emails_count):
        action = TriageAction(email_id=f"e{i:03d}", priority="urgent" if i % 2 == 0 else "low")
        _, _, done, _ = env_t1.step(action)
    assert done is True
    assert env_t1.state()["done"] is True
    with pytest.raises(RuntimeError):
        env_t1.step(TriageAction(email_id="too_late", priority="low"))
