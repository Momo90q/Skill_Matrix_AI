import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 1. Load data from SQLite database (ABSOLUTE PATH)
conn = sqlite3.connect(r'D:\skillmatrix\skill_matrix_ai\person_1\dsa_database.db')

df_attempts = pd.read_sql_query("""
    SELECT candidate_id, problem_id, topic, difficulty, 
           hints_used, time_spent_seconds, passed 
    FROM candidate_attempts
""", conn)

df_skills = pd.read_sql_query("SELECT * FROM candidate_skill_profiles", conn)
conn.close()

# 2. Merge candidate skills with attempt features
df_dataset = df_attempts.merge(df_skills, on='candidate_id', how='left')

difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
df_dataset['diff_level'] = df_dataset['difficulty'].map(difficulty_map)

def extract_matching_topic_skill(row):
    topic = row['topic']
    if topic in row and not pd.isna(row[topic]):
        return row[topic]
    return 0.0

df_dataset['candidate_topic_skill'] = df_dataset.apply(extract_matching_topic_skill, axis=1)

features = ['elo_rating', 'candidate_topic_skill', 'diff_level', 'hints_used', 'time_spent_seconds']
X = df_dataset[features].fillna(0)
y = df_dataset['passed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

df_results = pd.DataFrame(results)
print("Module 4 Complete: Problem-Success Prediction Evaluation Metrics:")
print(df_results)

# --- SAVE BEST MODEL ---
best_model_name = df_results.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, r'D:\skillmatrix\skill_matrix_ai\person_2\ml\models\success_predictor.joblib')
joblib.dump(features, r'D:\skillmatrix\skill_matrix_ai\person_2\ml\models\success_predictor_features.joblib')

print(f"\nSaved best model ({best_model_name}) to person_2/ml/models/success_predictor.joblib")