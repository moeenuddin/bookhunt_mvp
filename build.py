import os
import glob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

BOOKS_DIR = "books"
INDEX_DIR = "faiss_index"
CHUNK_SIZE = 500  # words per chunk

def split_into_chunks(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def build_documents():
    documents = []
    if not os.path.exists(BOOKS_DIR):
        print(f"❌ Folder '{BOOKS_DIR}' does not exist.")
        return []

    files = glob.glob(os.path.join(BOOKS_DIR, "*.txt"))
    if not files:
        print(f"⚠️ No .txt files found in '{BOOKS_DIR}'")
        return []

    for file_path in files:
        bookname = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_into_chunks(text, CHUNK_SIZE)
        for i, chunk in enumerate(chunks):
            metadata = {
                "bookname": bookname,
                "chunk_number": i + 1,
                "page": (i // 2) + 1
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    print(f"✅ Built {len(documents)} chunks from {len(files)} books.")
    return documents

def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding=embeddings)
    db.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved to '{INDEX_DIR}'")

if __name__ == "__main__":
    docs = build_documents()
    if docs:
        create_faiss_index(docs)
