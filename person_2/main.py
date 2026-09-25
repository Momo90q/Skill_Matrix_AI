import os
import json
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from engine.orchestrator import InterviewOrchestrator
from engine.schemas import CandidateSession

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="AI Interview Platform - Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = InterviewOrchestrator(passing_score_threshold=6.0)

# Multi-language templates & hidden constraints for Round 2
DSA_CHALLENGE = {
    "title": "Two Sum (Target Pair Detection)",
    "description": "Given an array of integers `nums` and an integer `target`, return the indices of the two numbers such that they add up to `target`.",
    "hidden_patterns": """
    - Pattern 1: Negative numbers and zero (e.g., nums=[-3, 4, 3, 90], target=0).
    - Pattern 2: Duplicate values completing target (e.g., nums=[3, 3], target=6 -> must not reuse same index).
    - Pattern 3: Large arrays requiring O(n) hash map time complexity; O(n^2) nested loops should receive lower score.
    - Pattern 4: No valid solution found (return empty list or appropriate null indicator).
    """,
    "templates": {
        "python": "def two_sum(nums: list[int], target: int) -> list[int]:\n    # Write your O(n) solution here\n    pass",
        "javascript": "function twoSum(nums, target) {\n    // Write your solution here\n    return [];\n}",
        "java": "class Solution {\n    public int[] twoSum(int[] nums, int target) {\n        // Write your solution here\n        return new int[]{};\n    }\n}",
        "cpp": "#include <vector>\n#include <unordered_map>\n\nstd::vector<int> twoSum(std::vector<int>& nums, int target) {\n    // Write your solution here\n    return {};\n}"
    }
}

active_sessions: dict[str, dict] = {}

@app.get("/health")
async def health_check():
    return {"status": "online", "active_sessions": len(active_sessions)}

@app.websocket("/ws/interview/{session_id}")
async def interview_ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in active_sessions:
        session = orchestrator.start_session(session_id=session_id, role="Full Stack & ML Engineer")
        active_sessions[session_id] = {
            "session": session,
            "round": 1,
            "round_1_scores": [],
            "round_2_score": 0,
            "round_2_details": {},
            "round_3_scores": []
        }
    
    session_data = active_sessions[session_id]
    session: CandidateSession = session_data["session"]

    # Initial Round 1 question
    first_question = orchestrator.engine.generate_question(session, topic="Database Indexing and Transactions")
    await websocket.send_json({
        "event": "new_question",
        "round": 1,
        "round_name": "Round 1: Core Technical Concepts",
        "difficulty": session.current_difficulty,
        "question": first_question
    })

    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            action = payload.get("action")

            # ---------------- ROUND 1: CORE TECHNICAL CONCEPTS ----------------
            if session_data["round"] == 1 and action == "submit_answer":
                candidate_text = payload.get("text", "")
                eval_result = orchestrator.engine.evaluate_response(session, candidate_text)
                session_data["round_1_scores"].append(eval_result.score)

                await websocket.send_json({
                    "event": "evaluation_result",
                    "score": eval_result.score,
                    "feedback": eval_result.feedback,
                    "difficulty": session.current_difficulty
                })

                if len(session_data["round_1_scores"]) >= 3:
                    gate_status = orchestrator.evaluate_round_1_transition(session)
                    if gate_status.passed_gate:
                        session_data["round"] = 2
                        await websocket.send_json({
                            "event": "round_transition",
                            "gate_status": gate_status.model_dump(),
                            "next_round": 2,
                            "challenge": {
                                "title": DSA_CHALLENGE["title"],
                                "description": DSA_CHALLENGE["description"],
                                "templates": DSA_CHALLENGE["templates"]
                            }
                        })
                else:
                    topic = "Bias-Variance Tradeoff and Model Evaluation" if len(session_data["round_1_scores"]) == 1 else "ACID Properties in Relational Databases"
                    next_q = orchestrator.engine.generate_question(session, topic=topic)
                    await websocket.send_json({
                        "event": "new_question",
                        "round": 1,
                        "difficulty": session.current_difficulty,
                        "question": next_q
                    })

            # ---------------- ROUND 2: MULTI-LANGUAGE CODING + HIDDEN PATTERNS ----------------
            elif session_data["round"] == 2 and action == "submit_code":
                code = payload.get("code", "")
                language = payload.get("language", "python")

                # Judge code against hidden test patterns
                code_eval = orchestrator.evaluate_code_solution(DSA_CHALLENGE, code, language)
                session_data["round_2_score"] = code_eval.score
                session_data["round_2_details"] = code_eval.model_dump()
                session_data["round"] = 3

                # Round 3 starter question
                system_design_challenge = "Design an end-to-end vector search architecture for 10 million embeddings with sub-50ms latency. Describe your database, indexing strategy, and caching layers."
                session.current_question = system_design_challenge

                await websocket.send_json({
                    "event": "round_transition",
                    "gate_status": {
                        "current_round": 3,
                        "round_name": "Round 3: Behavioral & System Design Defense",
                        "passed_gate": True,
                        "feedback_summary": f"Coding evaluated. Score: {code_eval.score}/10. Hidden patterns passed: {code_eval.passed_hidden_patterns}. Advancing to Defense."
                    },
                    "next_round": 3,
                    "code_feedback": code_eval.model_dump(),
                    "question": system_design_challenge
                })

            # ---------------- ROUND 3: ARCHITECTURE DEFENSE & FINAL REPORT ----------------
            elif session_data["round"] == 3 and action == "submit_answer":
                design_answer = payload.get("text", "")
                
                if len(session_data["round_3_scores"]) < 1:
                    # Provide devil's advocate challenge
                    session_data["round_3_scores"].append(8)
                    defense_critique = orchestrator.generate_round_3_defense_prompt(session, design_answer)
                    session.current_question = defense_critique
                    await websocket.send_json({
                        "event": "defense_challenge",
                        "question": defense_critique
                    })
                else:
                    # Final submission; compute individual section scores and overall average
                    session_data["round_3_scores"].append(8)

                    r1_avg = sum(session_data["round_1_scores"]) / len(session_data["round_1_scores"])
                    r2_score = session_data["round_2_score"]
                    r3_avg = sum(session_data["round_3_scores"]) / len(session_data["round_3_scores"])
                    total_avg = (r1_avg + r2_score + r3_avg) / 3

                    await websocket.send_json({
                        "event": "interview_complete",
                        "report": {
                            "session_id": session.session_id,
                            "role": session.role,
                            "overall_score": round(total_avg, 1),
                            "section_scores": {
                                "round_1_concepts": round(r1_avg, 1),
                                "round_2_coding": round(float(r2_score), 1),
                                "round_3_system_defense": round(r3_avg, 1)
                            },
                            "coding_breakdown": session_data["round_2_details"]
                        }
                    })

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected.")

@app.get("/", response_class=HTMLResponse)
async def serve_interview_ui():
    html_file = Path(__file__).resolve().parent / "index.html"
    return html_file.read_text(encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)