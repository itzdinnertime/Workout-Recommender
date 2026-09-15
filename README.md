# Workout Recommendation Engine

A workout progression recommender combining a rule-based system (classic
"progressive overload" coaching logic) with a trained ML model, served
behind a tested REST API.

## What it does

Given a user's logged training history for an exercise (weight, target
reps, reps actually achieved), it recommends the weight and reps for
their *next* session -- the same decision a coach makes: "hit your reps?
go up a bit. missed them? repeat. stuck for weeks? deload."

## Architecture

```
POST /workouts ──────────► SQLite (workout_logs table)
                                  │
GET /recommend/{user}/{exercise}  │
        │                        │
        ▼                        ▼
  load history ────────► enough history (5+ sessions)?
        │                   │              │
        │ no                yes            │
        ▼                   ▼              │
  rule_based_recommendation   ML model predicts next weight
   (rules.py)                 (ml_model.py, GradientBoostingRegressor)
                                     │
                                     ▼
                          domain-knowledge guardrail
                          (never recommend below last
                           weight if reps were hit)
                                     │
                                     ▼
                              final recommendation
```

- `app/database.py` -- SQLite storage for logged workouts
- `app/rules.py` -- Level 1: deterministic progressive-overload rules + plateau detection
- `app/ml_model.py` -- Level 2: feature engineering, training, and inference for the ML model
- `app/recommend.py` -- combines both: picks rule-based vs. ML depending on
  how much history exists, and applies a guardrail on top of the ML output
- `app/main.py` -- FastAPI service wiring it all together
- `data/generate_synthetic_data.py` -- generates realistic synthetic training histories
- `tests/` -- unit tests (rules), integration tests (API), and model-quality tests

## Why a rule-based fallback, not just ML

An ML model trained on rolling-window features (last 3 sessions, trend,
etc.) is meaningless for a brand-new user with 1-2 logged sessions -- this
is the classic "cold start" problem. Below `MIN_HISTORY_FOR_ML` (5)
logged sessions, the system falls back to simple, transparent rules
instead of asking a model to guess from insufficient data.

## The ML model

**Model**: `GradientBoostingRegressor` (scikit-learn). Chosen deliberately,
not by default: the dataset is small and tabular, and the relationship
between recent performance and next achievable weight is non-linear
(plateaus don't behave like a straight trend line) -- a tree ensemble is
well-suited to this size and shape of problem, and a deep neural network
would be overkill and would need far more data to do better.

**Features**: last weight/reps, rep deficit, 3-session rolling average
reps, 3-session weight trend, sessions since last weight increase,
exercise-type flag, and this user's average weight on this specific
exercise (so e.g. Overhead Press and Deadlift aren't confused just
because both are technically "upper"/"lower" body).

**Measured performance** (on held-out synthetic data, 80/20 split):
- R² ≈ 0.90
- MAE ≈ 11.8 kg
- MAPE ≈ 12%

**A real limitation, found through testing, not hidden**: the model
does not confidently extrapolate a linear trend. Given 3 consecutive
sessions of hitting every rep with steadily increasing weight, the model
predicts a next weight *below* naive linear extrapolation, because in
the training data, hot streaks are frequently followed by a stall (this
is intentional in the synthetic data generator). This is a legitimate
characteristic of tree-based regressors -- they predict based on the
average outcome of similar historical windows, they don't extrapolate
patterns the way a human coach (or a linear model) might.

This surfaced a second, more serious issue during manual end-to-end
testing: a user who stumbled once and then had a *successful comeback
session* got a raw model recommendation *below* the weight they'd just
successfully lifted -- which no real coach would ever suggest. Rather
than trust the model blindly, `recommend.py` applies a simple
domain-knowledge guardrail on top of the ML output: if the user just hit
their target reps, the recommendation is never allowed to fall below
their last successful weight. This is a deliberate, tested design
decision (see `tests/test_recommend.py::test_guardrail_prevents_weight_decrease_after_successful_lift`),
and it reflects a common real-world pattern: ML for nuance, explicit
rules for safety/sanity bounds.

## Running it

```bash
pip install -r requirements.txt

# Generate synthetic training data and train the model
python3 data/generate_synthetic_data.py
python3 -m app.ml_model

# Run the API
uvicorn app.main:app --reload
```

Then, for example:
```bash
curl -X POST http://127.0.0.1:8000/workouts -H "Content-Type: application/json" -d '{
  "user_id": "jason", "exercise": "Bench Press", "exercise_type": "upper",
  "log_date": "2026-09-01", "weight": 60, "reps_target": 8, "reps_achieved": 8, "sets": 3
}'

curl http://127.0.0.1:8000/recommend/jason/Bench%20Press
```

## Testing

```bash
python3 -m pytest tests/ -v
```

24 tests covering:
- **Unit tests** for the rule-based logic (progressive overload, plateau detection)
- **Integration tests** for every API endpoint, including input validation
  (rejecting invalid exercise types, negative weights, etc.)
- **Model quality tests** that assert measured accuracy stays above an
  agreed bar (R² > 0.7, MAPE < 20%) -- this is a regression test for model
  quality, not just code correctness: if a future change to features or
  training data silently makes the model worse, this test catches it
- **Directional/sanity tests** on model behavior (consistent progress
  should predict meaningfully higher than the streak's starting weight;
  repeated misses should predict lower)
- **The guardrail regression test**, capturing the exact real scenario
  found during manual testing

## Known limitations / possible extensions

- Synthetic training data, not real user data -- real lifter behavior may
  differ from the simulated patterns.
- Single coarse guardrail; a more thorough version might blend the ML
  prediction and rule-based prediction with a weighted average rather
  than a hard floor.
- No authentication -- `user_id` is just a string, not tied to real accounts.
- The model retrains from a flat CSV each time; a production version would
  version datasets/models and support incremental retraining.
