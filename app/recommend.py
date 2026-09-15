"""
Combines Level 1 (rule-based) and Level 2 (ML) into one recommendation
function.

Decision logic:
  - Fewer than MIN_HISTORY_FOR_ML sessions logged -> use rule-based only.
    This solves the "cold start" problem: an ML model trained on
    3-session-window features is meaningless (and was never trained on)
    a user with 1 logged session.
  - Enough history -> use the ML model's predicted weight as the primary
    number, but still run plateau detection from the rule-based system
    as an overlay, since that's a discrete, actionable flag the regression
    model doesn't explicitly output.
"""
from typing import List, Dict, Optional
from app.rules import rule_based_recommendation, detect_plateau, DELOAD_FACTOR
from app.ml_model import predict_next_weight

MIN_HISTORY_FOR_ML = 5


def recommend(history: List[Dict], exercise_type: str, model=None) -> Dict:
    """
    `history` must be sorted most-recent-first (as returned by database.get_history).
    """
    if not history:
        return {
            "method": "none",
            "suggested_weight": None,
            "suggested_reps": None,
            "plateau": False,
            "reason": "No history yet -- log a workout to get a recommendation.",
        }

    plateau = detect_plateau(history)

    if len(history) < MIN_HISTORY_FOR_ML or model is None:
        result = rule_based_recommendation(history, exercise_type)
        result["method"] = "rule_based"
        return result

    # ML path: build chronological (oldest-first) order for feature building.
    history_oldest_first = list(reversed(history))
    predicted_weight = predict_next_weight(model, history_oldest_first, exercise_type)
    last = history[0]

    if plateau:
        # Even if the model predicts a number, a detected plateau overrides
        # it with an explicit deload -- breaking plateaus is a coaching
        # decision, not something we want a black-box regression silently
        # papering over.
        deload_weight = round(last["weight"] * DELOAD_FACTOR, 1)
        return {
            "method": "ml_with_plateau_override",
            "suggested_weight": deload_weight,
            "suggested_reps": last["reps_target"],
            "plateau": True,
            "reason": (
                f"Model predicted ~{predicted_weight:.1f}kg, but a plateau was "
                f"detected over the last few sessions -- recommending a deload "
                f"to {deload_weight}kg instead."
            ),
        }

    # --- Guardrail: never recommend a weight *below* the last successful
    # weight if the user just hit their target reps. ---
    #
    # This was added after real testing surfaced a case where the raw
    # model predicted a weight noticeably below the user's last lift, even
    # though they had just hit all target reps. That happened because the
    # model regresses toward the average outcome of *similar-looking*
    # historical windows (which included a recent missed session dragging
    # the rolling averages down), not because it malfunctioned -- but it's
    # a recommendation no real coach would make, since "you just succeeded"
    # should never translate into "lift less next time." Rather than trust
    # the model blindly, we apply a simple domain-knowledge floor on top of
    # it: this is the kind of guardrail worth adding to any ML system that
    # makes user-facing recommendations, not just this one.
    final_weight = predicted_weight
    guardrail_applied = False
    if last["reps_achieved"] >= last["reps_target"] and predicted_weight < last["weight"]:
        final_weight = last["weight"]
        guardrail_applied = True

    reason = (
        f"Based on your last {len(history)} sessions, the model predicts "
        f"you're ready for ~{final_weight:.1f}kg for {last['reps_target']} reps."
    )
    if guardrail_applied:
        reason += (
            f" (Model's raw prediction was {predicted_weight:.1f}kg, but since you "
            f"hit all reps last time, we don't recommend going below your last "
            f"successful weight.)"
        )

    return {
        "method": "ml" if not guardrail_applied else "ml_with_guardrail",
        "suggested_weight": round(final_weight, 1),
        "suggested_reps": last["reps_target"],
        "plateau": False,
        "reason": reason,
    }
