import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_pipeline import ingest_pdf, ask_pdf

app = FastAPI(title="AI Document Assistant API")

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Last successfully indexed PDF — used when client does not send source
active_source: str | None = None


class QueryRequest(BaseModel):
    question: str
    source: str | None = None


@app.get("/")
def health():
    return {"status": "ok", "message": "RAG API is running", "active_source": active_source}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global active_source

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = ingest_pdf(file_path)
    active_source = file.filename
    return {
        "message": "PDF indexed successfully",
        "chunks": num_chunks,
        "file": file.filename,
        "active_source": active_source,
    }


@app.post("/chat")
async def chat_with_pdf(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    source = request.source or active_source
    if not source:
        raise HTTPException(
            status_code=400,
            detail="No active PDF. Upload and click Process & Index PDF first.",
        )

    answer = ask_pdf(request.question, source=source)
    return {"response": answer, "source": source}
