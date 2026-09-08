# AI Document Assistant

RAG app with **user accounts**, **chat history** (SQLite), and **per-user embeddings** (ChromaDB + Gemini).

## Stack

- FastAPI + Streamlit
- SQLAlchemy — users, documents, chat messages (SQLite local / Postgres via `DATABASE_URL`)
- ChromaDB — embeddings for search
- Gemini — embeddings + answers
- JWT auth (register / login)

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
JWT_SECRET=change-me-to-a-long-random-string
# Optional — default is sqlite:///./app_data.db
# DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
```

## Run

Terminal 1 (API):

```bash
uvicorn server:app --reload --port 8000
```

Terminal 2 (UI):

```bash
streamlit run app.py
```

Open http://localhost:8501

## Usage

1. **Register** or **Login**
2. Upload a PDF → **Process & Index PDF**
3. Ask questions (saved to your chat history)
4. Later: pick a document in the sidebar → **Load this document + chat history**

## API overview

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/register` | no | Create account |
| POST | `/login` | no | Get JWT |
| POST | `/upload` | yes | Index PDF for your user |
| POST | `/chat` | yes | Ask + save messages |
| GET | `/documents` | yes | Your PDFs |
| GET | `/chats` | yes | Your chat history |

## Notes

- Each user only searches their own Chroma chunks (`user_id` filter)
- Text PDFs work; scanned/image PDFs need OCR (not included yet)
- Do not commit `.env`, `venv/`, or `app_data.db`
