"""
api/practice.py
Read-only endpoints for Practice Mode: browse and fetch real problems
from Person 1's dataset. No AI judging, no timers — self-paced practice.
"""

import os
import sqlite3
import json
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import subprocess
import sys
from pydantic import BaseModel
import re

router = APIRouter(prefix="/api/practice", tags=["practice"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # person_2/
DB_PATH = os.path.join(BASE_DIR, "dsa_database.db")


def _row_to_problem(row, include_solution=False):
    (problem_id, title, description, topics_json, difficulty,
     constraints_json, test_cases_json, reference_solutions_json, common_mistakes) = row

    def safe_json(x):
        try:
            return json.loads(x) if x else None
        except Exception:
            return None

    problem = {
        "problem_id": problem_id,
        "title": title,
        "description": description,
        "topics": safe_json(topics_json) or [],
        "difficulty": difficulty,
        "constraints": safe_json(constraints_json) or [],
        "test_cases": safe_json(test_cases_json) or [],
    }
    if include_solution:
        problem["reference_solutions"] = safe_json(reference_solutions_json) or {}
    return problem


@router.get("/problems")
def list_problems(
    topic: Optional[str] = Query(None, description="Filter by topic, e.g. 'Array'"),
    difficulty: Optional[str] = Query(None, description="Easy, Medium, or Hard"),
    limit: int = Query(50, le=200),
    offset: int = Query(0)
):
    """Browse problems with optional filters. Returns lightweight list (no full description)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT problem_id, title, topics, difficulty FROM problems WHERE 1=1"
    params = []

    if difficulty:
        query += " AND difficulty = ?"
        params.append(difficulty)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for problem_id, title, topics_json, diff in rows:
        try:
            topics = json.loads(topics_json) if topics_json else []
        except Exception:
            topics = []

        if topic and topic not in topics:
            continue

        results.append({
            "problem_id": problem_id,
            "title": title,
            "topics": topics,
            "difficulty": diff
        })

    total = len(results)
    paged = results[offset: offset + limit]

    return {"total": total, "count": len(paged), "problems": paged}


@router.get("/problem/{problem_id}")
def get_problem(problem_id: str):
    """Fetch full details for one problem, including test cases (but not the reference solution)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT problem_id, title, description, topics, difficulty,
               constraints, test_cases, reference_solutions, common_mistakes
        FROM problems WHERE problem_id = ?
    """, (problem_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")

    return _row_to_problem(row, include_solution=False)
import subprocess
import sys
from pydantic import BaseModel


class RunCodeRequest(BaseModel):
    problem_id: str
    code: str
    language: str = "python"


@router.post("/run")
def run_code(req: RunCodeRequest):
    """
    Runs candidate's Python code against each test case.
    Candidate's code must PRINT the final answer (not return it) — the test
    case's input variables are injected before their code runs.
    Only Python is supported for now.
    """
    if req.language != "python":
        return {"error": "Automated checking currently only supports Python. You can still write and read your solution for other languages manually."}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT test_cases FROM problems WHERE problem_id = ?", (req.problem_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")

    try:
        test_cases = json.loads(row[0]) if row[0] else []
    except Exception:
        test_cases = []

    if not test_cases:
        return {"results": [], "message": "No test cases available for this problem."}

    results = []
    for tc in test_cases:
        test_input = tc.get("input", "")
        expected_output = str(tc.get("output", "")).strip()

         # Split "nums = [...], target = 9" into separate valid Python lines,
         # without breaking on commas that are inside lists/tuples.
        assignments = re.split(r',\s*(?=[A-Za-z_]\w*\s*=)', test_input)
        input_lines = "\n".join(a.strip() for a in assignments)
        full_script = f"{input_lines}\n{req.code}"

        try:
            proc = subprocess.run(
                [sys.executable, "-c", full_script],
                capture_output=True,
                text=True,
                timeout=5
            )
            actual_output = proc.stdout.strip()
            error_output = proc.stderr.strip()
            
            def normalize(s):
                return re.sub(r'\s+', '', s)  # strip all whitespace for comparison

            passed = (normalize(actual_output) == normalize(expected_output))

            results.append({
                "input": test_input,
                "expected": expected_output,
                "actual": actual_output if not error_output else f"ERROR: {error_output[-300:]}",
                "passed": passed
            })
        except subprocess.TimeoutExpired:
            results.append({
                "input": test_input,
                "expected": expected_output,
                "actual": "TIMEOUT (exceeded 5 seconds)",
                "passed": False
            })

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "total_tests": len(results),
        "passed_tests": passed_count,
        "all_passed": passed_count == len(results),
        "results": results
    }



