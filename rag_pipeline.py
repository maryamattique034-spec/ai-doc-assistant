"""
RAG pipeline: PDF → chunks → Gemini embeddings → Chroma (per user).
"""

import os
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import chromadb

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_documents")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


def ingest_pdf(pdf_path, user_id: int):
    """
    Index a PDF for one user.
    Metadata stores user_id + source so search stays private per account.
    """
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        print("No extractable text in this PDF (maybe a scan — needs OCR).")
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    print(f"Created {len(chunks)} chunks for user_id={user_id}.")

    file_stem = os.path.basename(pdf_path)
    user_key = str(user_id)

    print("Generating embeddings and saving to ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        # Unique id per user so two users can upload the same filename
        collection.upsert(
            ids=[f"u{user_key}_{file_stem}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": file_stem, "user_id": user_key}],
        )
    print("PDF Ingestion Complete!")
    return len(chunks)


def ask_pdf(question, user_id: int, source: str | None = None, n_results: int = 3):
    """
    Answer using only this user's chunks (and optional filename filter).
    """
    query_embedding = get_embedding(question)
    user_key = str(user_id)

    # Always filter by user. Optionally also filter by PDF name.
    if source:
        where = {"$and": [{"user_id": user_key}, {"source": source}]}
    else:
        where = {"user_id": user_key}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )

    if not results["documents"] or not results["documents"][0]:
        if source:
            return (
                f"No relevant context found in '{source}' for your account. "
                "Index that PDF first (text PDFs only — scans need OCR)."
            )
        return "No relevant context found. Upload and index a PDF first."

    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are an AI assistant. Answer the user's question ONLY using the provided document context below.
Answer in 1–2 short sentences. No preamble.
If the answer is not contained in the context, say "I cannot find this information in the PDF."

Context:
{context}

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    pdf = "sample_data/samplextra2.pdf"
    test_user = 1
    ingest_pdf(pdf, user_id=test_user)
    answer = ask_pdf(
        "What is the somatosensory system?",
        user_id=test_user,
        source=os.path.basename(pdf),
    )
    print(answer)
