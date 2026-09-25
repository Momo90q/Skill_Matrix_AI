from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class EvaluationResult(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Score from 0 to 10 evaluating the answer")
    factual_correctness: str = Field(..., description="Brief breakdown of factual accuracy and gaps")
    feedback: str = Field(..., description="Constructive feedback for the candidate")
    adjust_difficulty: Literal[-1, 0, 1] = Field(
        ..., 
        description="-1 if candidate struggled, 0 if adequate, 1 if mastered"
    )
    next_action: Literal["follow_up", "next_topic"] = Field(
        ..., 
        description="'follow_up' if answer was partial/vague, 'next_topic' if covered adequately"
    )
    follow_up_question: Optional[str] = Field(
        None, 
        description="Follow-up question if next_action is 'follow_up', otherwise null"
    )

class CandidateSession(BaseModel):
    session_id: str
    role: str
    current_difficulty: int = 1
    current_question: Optional[str] = None
    chat_history: List[dict] = []
    scores: List[int] = []