"""
POST /api/feedback — the student's rating of an answer.

Body: {"conversation_id": "...", "signal": 1}
      Got it = 1, Not sure = 0, Confused = -1.
      "topic" is optional (document id + 5-page block, e.g. "abc:0").
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services.learner_state import LearnerStateEstimator
from app.storage.signal_store import SQLiteSignalStore

router = APIRouter(prefix="/api", tags=["feedback"])

# Uses the same metadata.db file as the other stores. The signal store creates
# its own table on first use, so no change to dependencies.py is needed.
_estimator: LearnerStateEstimator | None = None


def get_estimator() -> LearnerStateEstimator:
    global _estimator
    if _estimator is None:
        _estimator = LearnerStateEstimator(
            SQLiteSignalStore(str(settings.metadata_db_path))
        )
    return _estimator


class FeedbackRequest(BaseModel):
    conversation_id: str
    signal: Literal[-1, 0, 1]
    topic: str | None = None


class FeedbackResponse(BaseModel):
    learner_state: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    if not body.conversation_id.strip():
        raise HTTPException(status_code=422, detail="conversation_id is required")
    estimator = get_estimator()
    await estimator.add_signal(body.conversation_id, body.signal, body.topic)
    state = await estimator.estimate(body.conversation_id, body.topic)
    return FeedbackResponse(learner_state=state)
