import os
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import chromadb

load_dotenv()

# 1. Initialize Gemini Client & ChromaDB (Local Vector DB)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.Client()
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
        contents=text
    )
    return response.embeddings[0].values

# 4. Process PDF & Save into Vector DB
def ingest_pdf(pdf_path):
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Text ko chunks mein split karein
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings and saving to ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    print("PDF Ingestion Complete!")



# 5. Question Query Function
def ask_pdf(question):
    # Step A: Question's embedding
    query_embedding = get_embedding(question)
    
    # Step B: retreive top 2 matching chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)
    
    # Step C: Strict Context-based Prompt
    prompt = f"""
    You are an AI assistant. Answer the user's question ONLY using the provided document context below.
    Answer in 1–2 short sentences. No preamble.
    If the answer is not contained in the context, say "I cannot find this information in the PDF."

    Context:
    {context}

    Question: {question}
    """
    
    # Step D: Generate Gemini response 
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt
    )
    
    return response.text


# Testing Code:
# ingest_pdf("sample.pdf")  
# answer = ask_pdf("What is the last line of this document?")
# print(answer)







if __name__ == "__main__":
    ingest_pdf("sample_data/samplextra2.pdf")
    answer = ask_pdf("What is the somatosensory system?")
    print(answer)

#The somatosensory system consists of sensors in the skin and sensors in our muscles, tendons, and joints.