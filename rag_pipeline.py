import os
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import chromadb

load_dotenv()

# 1. Initialize Gemini Client & persistent ChromaDB (survives restarts)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_documents")


# 2. Helper: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# 3. Helper: Generate Embeddings using Gemini API
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


# 4. Process PDF & Save into Vector DB
def ingest_pdf(pdf_path):
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    print(f"Created {len(chunks)} chunks.")

    file_stem = os.path.basename(pdf_path)
    print("Generating embeddings and saving to ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.upsert(
            ids=[f"{file_stem}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": file_stem}],
        )
    print("PDF Ingestion Complete!")
    return len(chunks)


# 5. Question Query Function (optionally limit to one PDF via source filename)
def ask_pdf(question, n_results=3, source=None):
    query_embedding = get_embedding(question)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if source:
        query_kwargs["where"] = {"source": source}

    results = collection.query(**query_kwargs)

    if not results["documents"] or not results["documents"][0]:
        if source:
            return f"No relevant context found in '{source}'. Index that PDF first."
        return "No relevant context found in indexed PDFs."

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
    # Re-run ingest only when you change the PDF; data stays in ./chroma_db
    pdf = "sample_data/samplextra2.pdf"
    ingest_pdf(pdf)
    answer = ask_pdf(
        "What is the somatosensory system?",
        source=os.path.basename(pdf),
    )
    print(answer)
