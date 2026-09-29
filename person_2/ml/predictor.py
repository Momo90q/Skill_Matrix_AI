"""
predictor.py
Loads Person 1's trained ML models and exposes simple functions
that Person 2's orchestrator.py can call.
"""

import os
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- Paths (relative to this file's location) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SUCCESS_MODEL_PATH = os.path.join(MODELS_DIR, "success_predictor.joblib")
SUCCESS_FEATURES_PATH = os.path.join(MODELS_DIR, "success_predictor_features.joblib")
EXPLANATION_MODEL_PATH = os.path.join(MODELS_DIR, "explanation_classifier")

# --- Load models once, at import time ---
_success_model = joblib.load(SUCCESS_MODEL_PATH)
_success_features = joblib.load(SUCCESS_FEATURES_PATH)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_explanation_tokenizer = DistilBertTokenizer.from_pretrained(EXPLANATION_MODEL_PATH)
_explanation_model = DistilBertForSequenceClassification.from_pretrained(EXPLANATION_MODEL_PATH).to(_device)
_explanation_model.eval()

_LABEL_MAP = {0: "incorrect", 1: "partially_correct", 2: "correct"}


def predict_success(elo_rating: float, candidate_topic_skill: float, diff_level: int,
                     hints_used: int, time_spent_seconds: int) -> dict:
    """
    Predicts probability that a candidate will pass a given problem attempt.
    diff_level: 1=Easy, 2=Medium, 3=Hard
    """
    row = {
        "elo_rating": elo_rating,
        "candidate_topic_skill": candidate_topic_skill,
        "diff_level": diff_level,
        "hints_used": hints_used,
        "time_spent_seconds": time_spent_seconds,
    }
    X = [[row[feat] for feat in _success_features]]

    prob = _success_model.predict_proba(X)[0][1]
    prediction = bool(prob >= 0.5)

    return {
        "passed_prediction": prediction,
        "confidence": round(float(prob), 4)
    }


def classify_explanation(text: str) -> dict:
    """
    Classifies a candidate's verbal/written explanation quality.
    """
    inputs = _explanation_tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=64
    ).to(_device)

    with torch.no_grad():
        outputs = _explanation_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {
        "label": _LABEL_MAP.get(pred_idx, "unknown"),
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    print("Testing predict_success...")
    print(predict_success(elo_rating=1200, candidate_topic_skill=0.5, diff_level=2, hints_used=0, time_spent_seconds=400))

    print("\nTesting classify_explanation...")
    print(classify_explanation("I used a hash map to store the complements, allowing O(N) lookup time."))