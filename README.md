# AI Technical Virtual Interview Platform (Intelligence Engine)

An end-to-end, multi-stage virtual technical interview engine built with FastAPI, LangChain, Groq, ChromaDB, and WebSockets[cite: 1, 2]. The platform automates adaptive technical evaluations across core computer science concepts, multi-language coding assessments with hidden test patterns, and real-time verbal system design defense with client-side anti-cheat face proctoring[cite: 1, 2].

---

## Key Features

- **Domain-Specific RAG Knowledge Pipeline:** Ingests and chunks technical documentation (DBMS, OS, Networks, ML) into a local Chroma vector database to ground questions and factual evaluations.
- **Dynamic Interview Engine & State Evaluator:** Evaluates candidate technical responses against ground-truth context, generates dynamic follow-ups, and adapts question difficulty on a scale from 1 to 5 in real time.
- **Three-Stage Multi-Round Orchestrator:**
  - **Round 1 (Core Concepts & Aptitude):** Automated RAG-grounded conceptual questioning backed by an average score gatekeeper[cite: 1, 2].
  - **Round 2 (Timed Coding & Pattern Verification):** Interactive coding environment supporting Python, JavaScript, Java, and C++, evaluated against hidden edge-case constraints and complexity targets.
  - **Round 3 (System Architecture & ML Defense):** Devil's advocate architectural challenges where the interviewer questions design choices, scalability bottlenecks, and failure modes.
- **Real-Time Speech & Virtual Conversational Interface:** 
  - Browser-native speech synthesis (TTS) for the interviewer[cite: 1, 2].
  - Speech-to-Text transcription via Groq Whisper (`whisper-large-v3-turbo`) with automated communication analytics (filler words, word count, speaking structure).
- **Client-Side Face Proctoring:** MediaPipe Face Detection tracks webcam feeds at 30 FPS to identify unauthorized secondary individuals or unmonitored sessions, terminating sessions if multiple people are detected.
- **Granular Performance Reporting:** Generates a comprehensive final score card detailing per-round scores, algorithmic time/space complexity, and hidden pattern compliance.

---

## System Architecture
Candidate Browser (Webcam + Audio + Code + Chat)
│
▼ (Bidirectional WebSocket)
FastAPI Backend Orchestrator (person_2/main.py)
├── RAG Engine (ChromaDB + HuggingFace all-MiniLM-L6-v2)
├── LLM Interview Agent (Groq / Llama 3 / OpenAI OSS models)
├── Speech Pipeline (Groq Whisper-large-v3-turbo)
└── MediaPipe Vision Detector (Client-Side Face Tracking)



## Directory Structure

```text
person_2/
├── api/                   # Modular API routers
├── data/
│   ├── chroma_db/         # Persistent vector database store
│   ├── docs/              # Raw technical documents for RAG ingestion
│   └── round1_questions.json
├── engine/
│   ├── __init__.py
│   ├── interview_engine.py# Dynamic prompt logic & answer grading
│   ├── orchestrator.py    # Multi-round state machine & gatekeeper
│   └── schemas.py         # Pydantic data schemas for state & evaluations
├── rag/
│   ├── __init__.py
│   ├── ingest.py          # Document loader, text chunker & vector builder
│   └── retriever.py       # Similarity search interface
├── speech/
│   ├── __init__.py
│   └── processor.py       # Audio transcription & filler word metrics
├── .env                   # Environment keys & configuration
├── index.html             # Real-time WebSocket interview dashboard
├── main.py                # Core FastAPI & WebSocket server
├── requirements.txt       # Project dependencies
└── test_engine.py         # Test verification script
