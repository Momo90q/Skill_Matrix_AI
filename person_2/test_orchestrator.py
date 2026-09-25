from engine.orchestrator import InterviewOrchestrator

def main():
    orchestrator = InterviewOrchestrator(passing_score_threshold=6.0)
    session = orchestrator.start_session("session_101", "Machine Learning Engineer")

    print("--- TESTING ROUND 1 GATEKEEPER ---")
    # Case A: Low scores (Fails gate)
    session.scores = [4, 5, 4]
    status_fail = orchestrator.evaluate_round_1_transition(session)
    print("Low Score Scenario:")
    print(f"  Passed Gate: {status_fail.passed_gate}")
    print(f"  Next Round: {status_fail.round_name} (Round {status_fail.current_round})")
    print(f"  Summary: {status_fail.feedback_summary}\n")

    # Case B: High scores (Passes gate to Round 2)
    session.scores = [7, 8, 8]
    status_pass = orchestrator.evaluate_round_1_transition(session)
    print("High Score Scenario:")
    print(f"  Passed Gate: {status_pass.passed_gate}")
    print(f"  Next Round: {status_pass.round_name} (Round {status_pass.current_round})")
    print(f"  Summary: {status_pass.feedback_summary}\n")

    print("--- TESTING ROUND 3 ML / SYSTEM DESIGN DEFENSE ---")
    mock_candidate_architecture = "I used a single PostgreSQL database with an unindexed table to store raw vector embeddings for real-time semantic search."
    defense_critique = orchestrator.generate_round_3_defense_prompt(session, mock_candidate_architecture)
    print(f"Candidate Choice: {mock_candidate_architecture}")
    print(f"\nInterviewer Defense Challenge:\n{defense_critique}")

if __name__ == "__main__":
    main()