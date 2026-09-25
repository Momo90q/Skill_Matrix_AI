from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./data/chroma_db"

class KnowledgeRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        results = self.vector_db.similarity_search(query, k=top_k)
        if not results:
            return ""
        return "\n\n".join([doc.page_content for doc in results])

# Quick verification test
if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    test_query = "What is the difference between precision and recall?"
    print(f"Query: {test_query}\n")
    print("Retrieved Context:")
    print(retriever.retrieve_context(test_query))