from typing import Dict, Any, List
from pydantic import BaseModel, Field
from engine.schemas import CandidateSession, EvaluationResult
from engine.interview_engine import InterviewEngine
from speech.processor import SpeechProcessor

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

class InterviewOrchestrator:
    def __init__(self, passing_score_threshold: float = 6.0):
        self.engine = InterviewEngine()
        self.speech_processor = SpeechProcessor()
        self.passing_threshold = passing_score_threshold

    def start_session(self, session_id: str, role: str) -> CandidateSession:
        return CandidateSession(session_id=session_id, role=role)

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

    def evaluate_code_solution(self, challenge: dict, code: str, language: str) -> CodeEvaluationResult:
        """Evaluates code for hidden patterns, edge cases, and algorithmic complexity."""
        evaluator = self.engine.llm.with_structured_output(CodeEvaluationResult)
        
        prompt = f"""
You are an automated LeetCode-style code judge evaluating a submitted coding problem.

Problem: {challenge['title']}
Description: {challenge['description']}
Hidden Patterns & Edge-Case Constraints to test:
{challenge['hidden_patterns']}

Candidate Programming Language: {language}
Candidate Submitted Code:
```{language}
{code}"""