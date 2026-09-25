import os
import pytest
from rag.retriever import KnowledgeRetriever

def test_chroma_db_exists():
    assert os.path.exists("./data/chroma_db"), "ChromaDB folder was not generated."

def test_retriever_returns_results():
    retriever = KnowledgeRetriever()
    query = "database indexing and transactions"
    result = retriever.retrieve_context(query, top_k=2)
    
    assert isinstance(result, str)
    assert len(result.strip()) > 0, "Retriever returned an empty string."
    assert "B-Tree" in result or "ACID" in result, "Retrieved text does not match ingested topics."