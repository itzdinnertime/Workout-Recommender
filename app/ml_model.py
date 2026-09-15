"""
Level 2: ML model that predicts the weight a lifter will actually be able
to lift for their target reps in their *next* session, based on their
recent training history.

Model choice: Gradient Boosted Trees (scikit-learn's GradientBoostingRegressor).
This is a deliberate choice, not a default -- the dataset is small and
tabular (a handful of numeric features per row), and the relationship
between "recent performance" and "next likely output" is non-linear
(e.g. a plateau doesn't behave like a straight-line trend). A deep neural
network would be overkill here and would need far more data to
outperform a well-tuned tree ensemble on a problem this size.

Target variable: next_weight_achievable -- the weight at which the lifter
will hit (or nearly hit) their target reps in their *next* logged session.
We derive this label directly from the synthetic data's known ground truth
during training (see build_training_table below).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

MODEL_PATH = "app/model.joblib"

FEATURE_COLUMNS = [
    "last_weight",
    "last_reps_achieved",
    "last_reps_target",
    "reps_deficit",          # target - achieved last time (0 if hit target)
    "avg_reps_achieved_3",   # rolling average over last 3 sessions
    "weight_trend_3",        # slope of weight over last 3 sessions
    "sessions_since_last_increase",
    "is_lower_body",
    "exercise_avg_weight",   # this user's average weight on this specific
                             # exercise so far -- lets the model distinguish
                             # e.g. Overhead Press (30-60kg) from Deadlift
                             # (80-160kg) even though both might share a
                             # "upper"/"lower" label
]


def build_feature_row(history_oldest_first: list, exercise_type: str) -> dict:
    """
    Builds one feature row from a chronologically-ordered history
    (oldest first), using only information available *before* the
    session we're predicting. Shared by both training and inference,
    so features can never drift between the two.
    """
    last = history_oldest_first[-1]
    recent3 = history_oldest_first[-3:] if len(history_oldest_first) >= 3 else history_oldest_first

    avg_reps_achieved_3 = np.mean([h["reps_achieved"] for h in recent3])
    exercise_avg_weight = np.mean([h["weight"] for h in history_oldest_first])

    if len(recent3) >= 2:
        weights = [h["weight"] for h in recent3]
        # simple slope: last - first over the window, per-session
        weight_trend_3 = (weights[-1] - weights[0]) / max(1, len(weights) - 1)
    else:
        weight_trend_3 = 0.0

    # How many sessions since the weight last went up.
    sessions_since_last_increase = 0
    for h in reversed(history_oldest_first[:-1]):
        if h["weight"] < last["weight"]:
            break
        sessions_since_last_increase += 1

    return {
        "last_weight": last["weight"],
        "last_reps_achieved": last["reps_achieved"],
        "last_reps_target": last["reps_target"],
        "reps_deficit": max(0, last["reps_target"] - last["reps_achieved"]),
        "avg_reps_achieved_3": avg_reps_achieved_3,
        "weight_trend_3": weight_trend_3,
        "sessions_since_last_increase": sessions_since_last_increase,
        "is_lower_body": 1 if exercise_type == "lower" else 0,
        "exercise_avg_weight": exercise_avg_weight,
    }


def build_training_table(csv_path: str) -> pd.DataFrame:
    """
    Walks each (user, exercise) history in chronological order and, for
    every session i (except the first, which has no history), builds a
    feature row describing sessions [0..i-1] with the label being what
    actually happened at session i: the weight used, and whether the
    lifter hit their reps (we predict the *achievable* weight, defined
    as: the weight they used if they hit target reps, or a slightly
    reduced effective weight if they didn't -- this gives the model a
    continuous, meaningful regression target instead of just echoing
    back the logged weight).
    """
    df = pd.read_csv(csv_path)
    df["log_date"] = pd.to_datetime(df["log_date"])

    rows = []
    for (user_id, exercise), group in df.groupby(["user_id", "exercise"]):
        group = group.sort_values("log_date").to_dict("records")
        for i in range(1, len(group)):
            history_so_far = group[:i]
            next_session = group[i]

            features = build_feature_row(history_so_far, next_session["exercise_type"])

            # Label: the "effective achievable weight" at the next session.
            # If they hit target reps, that's simply the weight they used.
            # If they missed, the weight was slightly too ambitious, so we
            # scale it down proportionally to the rep deficit -- this gives
            # a smooth, informative regression target.
            deficit_ratio = next_session["reps_achieved"] / next_session["reps_target"]
            label = next_session["weight"] * min(1.0, deficit_ratio + 0.05)

            features["label"] = label
            rows.append(features)

    return pd.DataFrame(rows)


def train_model(csv_path: str = "data/synthetic_workouts.csv", model_path: str = MODEL_PATH):
    table = build_training_table(csv_path)
    X = table[FEATURE_COLUMNS]
    y = table["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = float(np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100)

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)

    metrics = {
        "mae_kg": round(mae, 3), "r2": round(r2, 4), "mape_pct": round(mape, 2),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    print(f"Trained model on {len(X_train)} rows, tested on {len(X_test)} rows.")
    print(f"Mean Absolute Error: {mae:.2f} kg")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R^2 score: {r2:.4f}")
    return model, metrics


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def predict_next_weight(model, history_oldest_first: list, exercise_type: str) -> float:
    features = build_feature_row(history_oldest_first, exercise_type)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    return float(model.predict(X)[0])


if __name__ == "__main__":
    train_model()
