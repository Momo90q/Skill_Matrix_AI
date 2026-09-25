import os
from glob import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "./data/docs"
CHROMA_DIR = "./data/chroma_db"

def load_documents(directory: str):
    documents = []
    # Load TXT files
    for file_path in glob(f"{directory}/*.txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    # Load PDF files
    for file_path in glob(f"{directory}/*.pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    return documents

def build_vector_store():
    print("Loading documents...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No documents found in data/docs/")
        return

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating embeddings and saving to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Vector store successfully built at {CHROMA_DIR}")

if __name__ == "__main__":
    build_vector_store()