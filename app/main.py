"""
FastAPI service exposing the workout recommendation engine.

Endpoints:
  GET  /health                          -- liveness check
  POST /workouts                        -- log a completed workout session
  GET  /history/{user_id}/{exercise}     -- view logged history
  GET  /recommend/{user_id}/{exercise}   -- get a recommendation for the next session
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import database
from app.recommend import recommend
from app.ml_model import load_model, MODEL_PATH

app = FastAPI(title="Workout Recommendation Engine")

_model = None  # lazily loaded so tests can run without a trained model present


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH)
    return _model


@app.on_event("startup")
def on_startup():
    database.init_db()
    get_model()  # warm the model cache at startup rather than on first request


class WorkoutLogIn(BaseModel):
    user_id: str
    exercise: str
    exercise_type: str = Field(pattern="^(upper|lower)$")
    log_date: str  # ISO date string, e.g. "2026-09-10"
    weight: float = Field(gt=0)
    reps_target: int = Field(gt=0)
    reps_achieved: int = Field(ge=0)
    sets: int = Field(default=1, gt=0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.post("/workouts")
def log_workout(entry: WorkoutLogIn):
    database.insert_log(
        user_id=entry.user_id,
        exercise=entry.exercise,
        exercise_type=entry.exercise_type,
        log_date=entry.log_date,
        weight=entry.weight,
        reps_target=entry.reps_target,
        reps_achieved=entry.reps_achieved,
        sets=entry.sets,
    )
    return {"status": "logged"}


@app.get("/history/{user_id}/{exercise}")
def get_history(user_id: str, exercise: str, limit: Optional[int] = None):
    history = database.get_history(user_id, exercise, limit=limit)
    return {"user_id": user_id, "exercise": exercise, "history": history}


@app.get("/recommend/{user_id}/{exercise}")
def get_recommendation(user_id: str, exercise: str):
    history = database.get_history(user_id, exercise)
    if not history:
        raise HTTPException(status_code=404, detail="No workout history found for this user/exercise.")
    exercise_type = history[0]["exercise_type"]
    result = recommend(history, exercise_type, model=get_model())
    return result
