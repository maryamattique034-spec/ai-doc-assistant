"""
FastAPI server: auth, upload, chat, and chat history.
"""

import os
import shutil

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, EmailStr, Field

from auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from database import (
    create_user,
    get_document_for_user,
    get_user_by_email,
    init_db,
    list_documents,
    list_messages,
    save_document,
    save_message,
)
from rag_pipeline import ask_pdf, ingest_pdf

app = FastAPI(title="AI Document Assistant API")

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create SQLite tables on startup
init_db()


# ---------- request bodies ----------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class QueryRequest(BaseModel):
    question: str
    source: str | None = None
    document_id: int | None = None


# ---------- public routes ----------

@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "RAG API with user accounts is running",
        "db": "SQLAlchemy (users, documents, messages) + chroma (embeddings)",
    }


@app.post("/register")
def register(body: RegisterRequest):
    try:
        user = create_user(body.email, hash_password(body.password))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = create_access_token(user["id"], user["email"])
    return {
        "message": "Registered successfully",
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@app.post("/login")
def login(body: LoginRequest):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user["id"], user["email"])
    return {
        "message": "Logged in",
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"]},
    }


# ---------- protected routes (need Authorization: Bearer <token>) ----------

@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return current_user


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    user_id = current_user["id"]
    # Keep each user's files in their own folder
    user_dir = os.path.join(UPLOAD_DIR, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    file_path = os.path.join(user_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = ingest_pdf(file_path, user_id=user_id)
    if num_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted from this PDF. "
                "It may be a scanned image — OCR is not supported yet."
            ),
        )

    doc = save_document(user_id, file.filename, num_chunks)
    return {
        "message": "PDF indexed successfully",
        "chunks": num_chunks,
        "file": file.filename,
        "document_id": doc["id"],
        "active_source": file.filename,
    }


@app.post("/chat")
def chat_with_pdf(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    user_id = current_user["id"]
    source = request.source
    document_id = request.document_id

    # If client sent document_id, make sure it belongs to this user
    if document_id is not None:
        doc = get_document_for_user(document_id, user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        source = doc["filename"]

    if not source:
        raise HTTPException(
            status_code=400,
            detail="No PDF selected. Upload/index a PDF first.",
        )

    # Save user question
    save_message(user_id, "user", request.question, document_id=document_id)

    answer = ask_pdf(request.question, user_id=user_id, source=source)

    # Save assistant answer
    save_message(user_id, "assistant", answer, document_id=document_id)

    return {"response": answer, "source": source, "document_id": document_id}


@app.get("/documents")
def my_documents(current_user: dict = Depends(get_current_user)):
    return {"documents": list_documents(current_user["id"])}


@app.get("/chats")
def my_chats(
    document_id: int | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Return saved chat history for the logged-in user."""
    messages = list_messages(current_user["id"], document_id=document_id)
    return {"messages": messages}
