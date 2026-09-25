import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag.retriever import KnowledgeRetriever
from engine.schemas import EvaluationResult, CandidateSession

# Look for .env in current folder or parent folder
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError(f"GROQ_API_KEY not found! Please check {env_path}")

class InterviewEngine:
    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            api_key=groq_key
        )
        self.evaluator_llm = self.llm.with_structured_output(EvaluationResult)

    def generate_question(self, session: CandidateSession, topic: str) -> str:
        """Pulls domain context from RAG and generates a difficulty-adjusted question."""
        context = self.retriever.retrieve_context(topic, top_k=2)

        prompt = f"""
You are an expert technical interviewer conducting a live interview for a {session.role} position.
Current Question Difficulty Level: {session.current_difficulty} out of 5 (1=Basic, 3=Intermediate, 5=Architect/Advanced).

Domain Reference Material:
{context}

Generate ONE clear, challenging technical interview question on the topic of '{topic}' calibrated precisely to difficulty level {session.current_difficulty}.
Do not include answers, greetings, or explanations. Ask only the question.
"""
        response = self.llm.invoke(prompt)
        session.current_question = response.content.strip()
        return session.current_question

    def evaluate_response(self, session: CandidateSession, candidate_answer: str) -> EvaluationResult:
        """Evaluates candidate response, produces feedback, and determines state transitions."""
        context = self.retriever.retrieve_context(session.current_question, top_k=2)

        prompt = f"""
You are an objective technical interviewer evaluating a candidate's answer.

Role: {session.role}
Current Difficulty Level: {session.current_difficulty} / 5
Question Asked: {session.current_question}

Ground-Truth Knowledge Base Context:
{context}

Candidate's Answer:
{candidate_answer}

Evaluate the response strictly against technical accuracy, completeness, and clarity.
"""
        result: EvaluationResult = self.evaluator_llm.invoke(prompt)

        session.scores.append(result.score)
        session.current_difficulty = max(1, min(5, session.current_difficulty + result.adjust_difficulty))
        session.chat_history.append({
            "question": session.current_question,
            "answer": candidate_answer,
            "score": result.score,
            "feedback": result.feedback
        })

        return result