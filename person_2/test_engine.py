from engine.schemas import CandidateSession
from engine.interview_engine import InterviewEngine

def main():
    engine = InterviewEngine()
    session = CandidateSession(session_id="test_001", role="Machine Learning Engineer")

    # 1. Generate Question
    print("--- 1. GENERATING QUESTION ---")
    topic = "Bias-Variance Tradeoff and Model Evaluation"
    question = engine.generate_question(session, topic=topic)
    print(f"Difficulty ({session.current_difficulty}/5) Question:\n{question}\n")

    # 2. Simulate Candidate Answer
    mock_answer = "High bias causes underfitting because the model is too simple, while high variance leads to overfitting on training noise."
    print("--- 2. SUBMITTING CANDIDATE ANSWER ---")
    print(f"Answer: {mock_answer}\n")

    # 3. Evaluate Answer
    print("--- 3. EVALUATING ANSWER ---")
    eval_result = engine.evaluate_response(session, mock_answer)
    print(f"Score: {eval_result.score}/10")
    print(f"Correctness: {eval_result.factual_correctness}")
    print(f"Feedback: {eval_result.feedback}")
    print(f"Difficulty Adjustment: {eval_result.adjust_difficulty}")
    print(f"Next Action: {eval_result.next_action}")
    if eval_result.follow_up_question:
        print(f"Follow-up: {eval_result.follow_up_question}")
    print(f"New Difficulty Level: {session.current_difficulty}/5")

if __name__ == "__main__":
    main()