from typing import List, Dict, Any, Tuple
from .models import TriageAction, TriageReward

def _calculate_penalties(task_id: int, action: TriageAction, actions_taken: List[TriageAction]) -> Tuple[float, str]:
    """
    Applies global penalties across all tasks for invalid or degenerative agent behavior.
    """
    penalty = 0.0
    feedback_msgs = []
    
    # 1. Empty/null action penalty (-0.1)
    # Depending on task, missing required fields (even if Pydantic allowed it via Optional) is penalized.
    if task_id >= 2 and not action.category:
        penalty -= 0.10
        feedback_msgs.append("Penalty: Missing category (-0.10).")
    if task_id == 3 and not action.reply_draft:
        penalty -= 0.10
        feedback_msgs.append("Penalty: Missing reply draft (-0.10).")
        
    # 2. Repeated identical actions (-0.05 each)
    # Prevents the agent from outputting the exact same (priority, category, reply) combo repeatedly.
    # Note: In Task 1 and 2, identical overlaps are mathematically guaranteed, which reduces the theoretical max score 
    # slightly, but strictly adheres to the "apply to all tasks" rule.
    repeats = 0
    for past in actions_taken:
        if (past.priority == action.priority and 
            past.category == action.category and 
            past.reply_draft == action.reply_draft):
            repeats += 1
            
    if repeats > 0:
        p = repeats * 0.05
        penalty -= p
        feedback_msgs.append(f"Penalty: Repeated identical action {repeats} times (-{p:.2f}).")
        
    return penalty, " ".join(feedback_msgs)

def grade_task_1(action: TriageAction, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """TASK 1: Easy - Priority Sorting Only"""
    breakdown = {"priority": 0.0}
    feedback = ""
    
    if action.priority.lower() == ground_truth["priority"].lower():
        breakdown["priority"] = 1.0
        feedback = "Priority is correct."
    else:
        feedback = f"Priority is incorrect. Expected {ground_truth['priority']}."
        
    return sum(breakdown.values()), breakdown, feedback

def grade_task_2(action: TriageAction, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """TASK 2: Medium - Full Categorization (Priority + Category)"""
    breakdown = {"priority": 0.0, "category": 0.0}
    feedback_msgs = []
    
    if action.priority.lower() == ground_truth["priority"].lower():
        breakdown["priority"] = 0.5
        feedback_msgs.append("Priority correct.")
    else:
        feedback_msgs.append(f"Priority incorrect (expected {ground_truth['priority']}).")
        
    if action.category and action.category.lower() == ground_truth["category"].lower():
        breakdown["category"] = 0.5
        feedback_msgs.append("Category correct.")
    else:
        feedback_msgs.append(f"Category incorrect (expected {ground_truth['category']}).")
        
    return sum(breakdown.values()), breakdown, " ".join(feedback_msgs)

def grade_task_3(action: TriageAction, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """TASK 3: Hard - Full Triage + Reply Draft"""
    breakdown = {"priority": 0.0, "category": 0.0, "reply": 0.0}
    feedback_msgs = []
    
    # Grade Prioriy
    if action.priority.lower() == ground_truth["priority"].lower():
        breakdown["priority"] = 0.35
        feedback_msgs.append("Priority correct.")
    else:
        feedback_msgs.append(f"Priority incorrect (expected {ground_truth['priority']}).")
        
    # Grade Category
    if action.category and action.category.lower() == ground_truth["category"].lower():
        breakdown["category"] = 0.35
        feedback_msgs.append("Category correct.")
    else:
        feedback_msgs.append(f"Category incorrect (expected {ground_truth['category']}).")
        
    # Grade Reply (Length + Keyword constraints)
    ideal_reply = ground_truth.get("ideal_reply", "")
    draft = action.reply_draft or ""
    
    reply_score = 0.0
    if draft:
        # Brevity & Length check
        if len(draft.split()) >= 3 and len(draft) <= 250:
            reply_score += 0.15
            
        # Relevance & Professionalism check
        ideal_words = set(ideal_reply.lower().split())
        draft_words = set(draft.lower().split())
        stop_words = {"the", "is", "at", "which", "and", "on", "a", "an", "to", "we", "are", "of", "in", "for", "with", "it", "this"}
        
        ideal_key = ideal_words - stop_words
        has_overlap = len(ideal_key.intersection(draft_words)) >= 1
        
        prof_keywords = ["apologi", "sorry", "thank", "pleas", "investigat", "look", "resolv", "updat", "fix", "schedul", "contact"]
        has_professional_tone = any(word in draft.lower() for word in prof_keywords)
        
        if has_overlap or has_professional_tone:
            reply_score += 0.15
            
    breakdown["reply"] = round(reply_score, 2)
    
    if reply_score >= 0.30:
        feedback_msgs.append("Reply is relevant and professional.")
    elif reply_score > 0:
        feedback_msgs.append("Reply is partially relevant or lacks professionalism.")
    else:
        feedback_msgs.append("Reply is missing or highly irrelevant/unprofessional.")
        
    return sum(breakdown.values()), breakdown, " ".join(feedback_msgs)

def grade_action(task_id: int, action: TriageAction, ground_truth: Dict[str, Any], actions_taken: List[TriageAction]) -> TriageReward:
    """
    Main entry point for calculating step reward.
    Routes to specific tasks based on task_id, evaluates raw scores,
    applies global penalties, and clamps the final score between 0.0 and 1.0.
    """
    if task_id == 1:
        base_score, breakdown, feedback = grade_task_1(action, ground_truth)
    elif task_id == 2:
        base_score, breakdown, feedback = grade_task_2(action, ground_truth)
    elif task_id == 3:
        base_score, breakdown, feedback = grade_task_3(action, ground_truth)
    else:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")
        
    # Subtract penalties
    penalty, penalty_feedback = _calculate_penalties(task_id, action, actions_taken)
    
    final_score = max(0.0, min(1.0, base_score + penalty))
    
    # Reassemble feedback string
    full_feedback = feedback
    if penalty_feedback:
        full_feedback += f" | {penalty_feedback}"
        
    return TriageReward(
        score=round(final_score, 2),
        breakdown=breakdown,
        feedback=full_feedback.strip()
    )
