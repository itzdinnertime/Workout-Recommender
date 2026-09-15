"""
Level 1: Rule-based progressive overload recommendation.

This encodes the standard heuristic a human coach would use:
  - Hit all target reps last time?  -> increase weight
  - Missed target reps?             -> repeat the same weight
  - Same weight + same reps stuck for several sessions in a row -> plateau,
    recommend a deload (temporary weight reduction) to break out of it

No ML here at all -- this is intentionally simple and deterministic, and
it's also the fallback used when there isn't enough history yet for the
ML model in level 2 to make a sensible prediction (the "cold start" problem).
"""
from typing import List, Dict, Optional

# How much weight to add when a lift is successfully completed.
# Lower-body compound lifts tolerate bigger jumps than upper-body isolation work.
INCREMENT_KG = {
    "lower": 5.0,
    "upper": 2.5,
}

PLATEAU_WINDOW = 3      # how many recent sessions we look at
DELOAD_FACTOR = 0.9     # reduce weight by 10% on a deload


def detect_plateau(history: List[Dict]) -> bool:
    """
    A plateau = the last PLATEAU_WINDOW sessions all used the same weight
    AND the user did not manage to increase reps_achieved either.
    `history` is expected sorted most-recent-first (as returned by get_history).
    """
    if len(history) < PLATEAU_WINDOW:
        return False

    recent = history[:PLATEAU_WINDOW]
    weights = {row["weight"] for row in recent}
    if len(weights) != 1:
        return False  # weight changed recently, not stuck

    reps = [row["reps_achieved"] for row in recent]
    # Stuck if reps never improved across the window (non-increasing sequence,
    # read oldest-to-newest).
    reps_oldest_first = list(reversed(reps))
    improved = any(reps_oldest_first[i] < reps_oldest_first[i + 1]
                   for i in range(len(reps_oldest_first) - 1))
    return not improved


def rule_based_recommendation(history: List[Dict], exercise_type: str) -> Dict:
    """
    Returns a recommendation dict:
      { "suggested_weight": float, "suggested_reps": int,
        "plateau": bool, "reason": str }
    """
    if not history:
        return {
            "suggested_weight": None,
            "suggested_reps": None,
            "plateau": False,
            "reason": "No history yet -- log a workout to get a recommendation.",
        }

    last = history[0]  # most recent session
    increment = INCREMENT_KG.get(exercise_type, 2.5)
    plateau = detect_plateau(history)

    if plateau:
        deload_weight = round(last["weight"] * DELOAD_FACTOR, 1)
        return {
            "suggested_weight": deload_weight,
            "suggested_reps": last["reps_target"],
            "plateau": True,
            "reason": (
                f"No progress over the last {PLATEAU_WINDOW} sessions at "
                f"{last['weight']}kg -- suggesting a deload to {deload_weight}kg "
                f"to break the plateau."
            ),
        }

    if last["reps_achieved"] >= last["reps_target"]:
        new_weight = round(last["weight"] + increment, 1)
        return {
            "suggested_weight": new_weight,
            "suggested_reps": last["reps_target"],
            "plateau": False,
            "reason": (
                f"Hit all {last['reps_target']} reps at {last['weight']}kg last time -- "
                f"try {new_weight}kg next session."
            ),
        }
    else:
        return {
            "suggested_weight": last["weight"],
            "suggested_reps": last["reps_target"],
            "plateau": False,
            "reason": (
                f"Missed target reps last time ({last['reps_achieved']}/"
                f"{last['reps_target']} at {last['weight']}kg) -- repeat the same "
                f"weight and aim to hit all reps."
            ),
        }
