import sys
import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from engine.schemas import CandidateSession, EvaluationResult
from engine.interview_engine import InterviewEngine
from engine.problem_loader import get_random_problem
from speech.processor import SpeechProcessor

# Make sure we can import ml/predictor.py, which lives at person_2/ml/predictor.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.predictor import predict_success, classify_explanation


class RoundStatus(BaseModel):
    current_round: int = 1
    round_name: str = "Core Technical Concepts"
    is_completed: bool = False
    passed_gate: bool = False
    feedback_summary: str = ""


class CodeEvaluationResult(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Score from 0 to 10 evaluating the code")
    passed_hidden_patterns: bool = Field(..., description="Whether hidden patterns/edge cases were handled")
    hidden_pattern_feedback: str = Field(..., description="Analysis of hidden edge cases (e.g. empty arrays, duplicates, negative numbers)")
    time_complexity: str = Field(..., description="Estimated time complexity (e.g., O(n))")
    space_complexity: str = Field(..., description="Estimated space complexity (e.g., O(1))")
    summary: str = Field(..., description="Short review of candidate implementation")
    ml_success_confidence: float = Field(0.0, description="ML model's predicted probability the candidate would pass, based on skill profile")


class InterviewOrchestrator:
    def __init__(self, passing_score_threshold: float = 6.0):
        self.engine = InterviewEngine()
        self.speech_processor = SpeechProcessor()
        self.passing_threshold = passing_score_threshold

    def start_session(self, session_id: str, role: str) -> CandidateSession:
        return CandidateSession(session_id=session_id, role=role)

    def get_next_challenge(self, difficulty: str = None) -> dict:
        """Pulls a real DSA problem from Person 1's dataset instead of a hardcoded one."""
        return get_random_problem(difficulty=difficulty)

    def evaluate_round_1_transition(self, session: CandidateSession) -> RoundStatus:
        if not session.scores:
            return RoundStatus(
                current_round=1,
                round_name="Core Technical Concepts",
                is_completed=False,
                passed_gate=False,
                feedback_summary="No answers submitted yet."
            )

        avg_score = sum(session.scores) / len(session.scores)
        passed = avg_score >= self.passing_threshold

        return RoundStatus(
            current_round=1 if not passed else 2,
            round_name="Core Technical Concepts" if not passed else "Timed Coding (DSA)",
            is_completed=True,
            passed_gate=passed,
            feedback_summary=f"Round 1 Average Score: {avg_score:.1f}/10. Threshold: {self.passing_threshold}. " +
                             ("Passed to Round 2." if passed else "Did not meet passing criteria.")
        )

    def evaluate_code_solution(self, challenge: dict, code: str, language: str,
                                candidate_elo: float = 1200, candidate_topic_skill: float = 0.0,
                                hints_used: int = 0, time_spent_seconds: int = 600) -> CodeEvaluationResult:
        """
        Evaluates code using the LLM judge, then blends in Person 1's ML success-prediction
        model as a second, independent signal.
        """
        evaluator = self.engine.llm.with_structured_output(CodeEvaluationResult)

        diff_map = {"Easy": 1, "Medium": 2, "Hard": 3}
        diff_level = diff_map.get(challenge.get("difficulty", "Medium"), 2)

        prompt = f"""
You are an automated LeetCode-style code judge evaluating a submitted coding problem.

Problem: {challenge['title']}
Description: {challenge['description']}
Hidden Patterns & Edge-Case Constraints to test:
{challenge.get('hidden_patterns', 'None specified.')}

Candidate Programming Language: {language}
Candidate Submitted Code:
```{language}
{code}
```

Evaluate this submission strictly. Score from 0-10, determine whether hidden edge cases
(negative numbers, duplicates, empty input, large input performance) are handled correctly,
and estimate the time and space complexity of the approach.
"""

        llm_result: CodeEvaluationResult = evaluator.invoke(prompt)

        # --- Blend in Person 1's ML success-prediction model ---
        ml_result = predict_success(
            elo_rating=candidate_elo,
            candidate_topic_skill=candidate_topic_skill,
            diff_level=diff_level,
            hints_used=hints_used,
            time_spent_seconds=time_spent_seconds
        )
        llm_result.ml_success_confidence = ml_result["confidence"]

        return llm_result

    def generate_round_3_defense_prompt(self, session: CandidateSession, design_answer: str) -> str:
        """
        Acts as a devil's advocate, challenging the candidate's system design choice,
        and uses Person 1's explanation classifier to gauge answer quality first.
        """
        quality = classify_explanation(design_answer)

        prompt = f"""
You are a principal engineer conducting a system design defense interview.

The candidate proposed the following architecture:
{design_answer}

An automated quality classifier rated this explanation as: {quality['label']} (confidence: {quality['confidence']}).

Write ONE tough, specific devil's-advocate follow-up question that challenges a weakness,
trade-off, or edge case in their proposed design. Do not soften it. Ask only the question,
no preamble.
"""
        response = self.engine.llm.invoke(prompt)
        return response.content.strip()