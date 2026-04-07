from typing import List, Dict, Any, Tuple
from .models import TriageAction, TriageReward

def _calculate_penalties(task_id: int, action: TriageAction, actions_taken: List[TriageAction]) -> Tuple[float, str]:
    penalty = 0.0
    feedback_msgs = []
    
    if task_id >= 2 and not action.category:
        penalty -= 0.10
        feedback_msgs.append("Penalty: Missing category (-0.10).")
    if task_id == 3 and not action.reply_draft:
        penalty -= 0.10
        feedback_msgs.append("Penalty: Missing reply draft (-0.10).")
        
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
    breakdown = {"priority": 0.0}
    feedback = ""
    if action.priority.lower() == ground_truth["priority"].lower():
        breakdown["priority"] = 1.0
        feedback = "Priority is correct."
    else:
        feedback = f"Priority is incorrect. Expected {ground_truth['priority']}."
    return sum(breakdown.values()), breakdown, feedback

def grade_task_2(action: TriageAction, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
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
    breakdown = {"priority": 0.0, "category": 0.0, "reply": 0.0}
    feedback_msgs = []
    if action.priority.lower() == ground_truth["priority"].lower():
        breakdown["priority"] = 0.35
        feedback_msgs.append("Priority correct.")
    else:
        feedback_msgs.append(f"Priority incorrect (expected {ground_truth['priority']}).")
    if action.category and action.category.lower() == ground_truth["category"].lower():
        breakdown["category"] = 0.35
        feedback_msgs.append("Category correct.")
    else:
        feedback_msgs.append(f"Category incorrect (expected {ground_truth['category']}).")
    ideal_reply = ground_truth.get("ideal_reply", "")
    draft = action.reply_draft or ""
    reply_score = 0.0
    if draft:
        if len(draft.split()) >= 3 and len(draft) <= 250:
            reply_score += 0.15
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
    if task_id == 1:
        base_score, breakdown, feedback = grade_task_1(action, ground_truth)
    elif task_id == 2:
        base_score, breakdown, feedback = grade_task_2(action, ground_truth)
    elif task_id == 3:
        base_score, breakdown, feedback = grade_task_3(action, ground_truth)
    else:
        raise ValueError(f"Unknown task_id: {task_id}.")
    penalty, penalty_feedback = _calculate_penalties(task_id, action, actions_taken)
    final_score = max(0.0, min(1.0, base_score + penalty))
    full_feedback = feedback
    if penalty_feedback:
        full_feedback += f" | {penalty_feedback}"
    return TriageReward(
        score=round(final_score, 2),
        breakdown=breakdown,
        feedback=full_feedback.strip()
    )
