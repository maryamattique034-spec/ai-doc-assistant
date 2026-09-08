"""
Structured database with SQLAlchemy (users, documents, chat history).

Default: SQLite file for local learning.
Deploy: set DATABASE_URL to Postgres, e.g.
  postgresql+psycopg2://user:pass@host:5432/dbname

ChromaDB stays separate — it only stores embeddings for RAG search.
"""

import os
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# Local default. For Postgres on Render/etc, set DATABASE_URL in .env
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app_data.db")

# SQLite needs check_same_thread=False for FastAPI
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def _user_to_dict(user: User, include_password: bool = False) -> dict:
    data = {
        "id": user.id,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }
    if include_password:
        data["password_hash"] = user.password_hash
    return data


def _doc_to_dict(doc: Document) -> dict:
    return {
        "id": doc.id,
        "user_id": doc.user_id,
        "filename": doc.filename,
        "chunks": doc.chunks,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


def _msg_to_dict(msg: Message) -> dict:
    return {
        "id": msg.id,
        "user_id": msg.user_id,
        "document_id": msg.document_id,
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


def init_db():
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


def create_user(email: str, password_hash: str) -> dict:
    db = SessionLocal()
    try:
        user = User(email=email.lower().strip(), password_hash=password_hash)
        db.add(user)
        db.commit()
        db.refresh(user)
        return _user_to_dict(user)
    except IntegrityError:
        db.rollback()
        raise ValueError("Email already registered")
    finally:
        db.close()


def get_user_by_email(email: str):
    db = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.email == email.lower().strip()))
        return _user_to_dict(user, include_password=True) if user else None
    finally:
        db.close()


def get_user_by_id(user_id: int):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        return _user_to_dict(user) if user else None
    finally:
        db.close()


def save_document(user_id: int, filename: str, chunks: int) -> dict:
    db = SessionLocal()
    try:
        doc = Document(user_id=user_id, filename=filename, chunks=chunks)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return _doc_to_dict(doc)
    finally:
        db.close()


def list_documents(user_id: int) -> list:
    db = SessionLocal()
    try:
        docs = db.scalars(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        ).all()
        return [_doc_to_dict(d) for d in docs]
    finally:
        db.close()


def get_document_for_user(document_id: int, user_id: int):
    db = SessionLocal()
    try:
        doc = db.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )
        return _doc_to_dict(doc) if doc else None
    finally:
        db.close()


def save_message(user_id: int, role: str, content: str, document_id: int | None = None):
    db = SessionLocal()
    try:
        msg = Message(
            user_id=user_id,
            document_id=document_id,
            role=role,
            content=content,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg.id
    finally:
        db.close()


def list_messages(user_id: int, document_id: int | None = None, limit: int = 100) -> list:
    """Load chat history for this user (optionally for one document)."""
    db = SessionLocal()
    try:
        stmt = select(Message).where(Message.user_id == user_id)
        if document_id is not None:
            stmt = stmt.where(Message.document_id == document_id)
        stmt = stmt.order_by(Message.created_at.asc()).limit(limit)
        messages = db.scalars(stmt).all()
        return [_msg_to_dict(m) for m in messages]
    finally:
        db.close()
