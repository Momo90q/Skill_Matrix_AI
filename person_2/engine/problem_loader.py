"""
problem_loader.py
Pulls a real DSA problem from the shared dsa_database.db (built by Person 1's pipeline)
and formats it to match the shape main.py expects (like the old hardcoded DSA_CHALLENGE).
"""

import os
import sqlite3
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "dsa_database.db")  # person_2/dsa_database.db

# Basic starter templates per language (problems table doesn't store these, so we generate generically)
GENERIC_TEMPLATES = {
    "python": "def solve(*args):\n    # Write your solution here\n    pass",
    "javascript": "function solve() {\n    // Write your solution here\n    return null;\n}",
    "java": "class Solution {\n    public Object solve() {\n        // Write your solution here\n        return null;\n    }\n}",
    "cpp": "// Write your solution here\n"
}


def get_random_problem(difficulty: str = None) -> dict:
    """
    Fetches a random problem from the problems table.
    difficulty: optional filter - 'Easy', 'Medium', or 'Hard'
    Returns a dict shaped like the old DSA_CHALLENGE constant.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if difficulty:
        cursor.execute(
            "SELECT problem_id, title, description, topics, difficulty, constraints, test_cases, common_mistakes FROM problems WHERE difficulty = ?",
            (difficulty,)
        )
    else:
        cursor.execute(
            "SELECT problem_id, title, description, topics, difficulty, constraints, test_cases, common_mistakes FROM problems"
        )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No problems found in database (difficulty filter: {difficulty})")

    row = random.choice(rows)
    problem_id, title, description, topics_json, diff, constraints_json, test_cases_json, common_mistakes = row

    try:
        topics = json.loads(topics_json) if topics_json else []
    except Exception:
        topics = []

    try:
        test_cases = json.loads(test_cases_json) if test_cases_json else []
    except Exception:
        test_cases = []

    hidden_patterns_text = common_mistakes if common_mistakes else "No specific hidden patterns recorded for this problem."

    return {
        "problem_id": problem_id,
        "title": title,
        "description": description,
        "topics": topics,
        "difficulty": diff,
        "hidden_patterns": hidden_patterns_text,
        "test_cases": test_cases,
        "templates": GENERIC_TEMPLATES
    }


if __name__ == "__main__":
    problem = get_random_problem()
    print("Random problem loaded:")
    print(json.dumps(problem, indent=2)[:1000])  # print first 1000 chars so it's not overwhelming