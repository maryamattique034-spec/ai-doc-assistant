# AI Document Assistant

Simple RAG app: upload a PDF, index it into local ChromaDB, and ask questions with Gemini.

## Stack

- FastAPI + Streamlit
- ChromaDB (persistent)
- Gemini embeddings + chat
- pypdf

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

## Run

Terminal 1:

```bash
uvicorn server:app --reload --port 8000
```

Terminal 2:

```bash
streamlit run app.py
```

Open the Streamlit URL (usually http://localhost:8501).

## Usage

1. Upload a PDF in the sidebar
2. Click **Process & Index PDF**
3. Ask questions in the chat (answers use only the active PDF)

## Notes

- Works best with PDFs that have selectable text
- Scanned/image PDFs need OCR (not included yet)
- Do not commit `.env` or `venv/`
