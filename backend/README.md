# 3GPP RAG Backend

FastAPI backend for the 3GPP Retrieval-Augmented Generation system.

## Architecture

Next.js / Vercel
        |
        | HTTPS POST /api/chat
        v
FastAPI / Render
        |
        v
Existing RAG graph
        |
        +--> Hybrid retrieval
        +--> Qdrant
        +--> BM25
        +--> Reranker
        +--> Evidence gate
        +--> Groq / GPT-OSS-120B
        +--> Citation validation
        +--> Groundedness / abstention
        |
        v
Structured JSON response

## Run locally

From this directory:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set:

```text
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
FRONTEND_URL=http://localhost:3000
```

The existing `src/` and `data/` directories must be present beside `api/`.

Run:

```bash
python -m uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### GET /api/health

Backend health check.

### GET /api/info

Returns the model, corpus documents, and RAG capabilities.

### POST /api/chat

Request:

```json
{
  "question": "What is the role of the AMF?",
  "conversation_id": "optional-id"
}
```

Response contains:

- answer
- grounded
- abstained
- reason
- citations
- sources
- conversation_id

## Important

The backend expects the existing RAG implementation under:

```text
src/
```

and its indexed data under:

```text
data/
```

The API layer does not duplicate or replace the retrieval/generation implementation.
