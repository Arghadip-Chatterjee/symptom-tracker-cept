# Symptom Tracker RAG MVP

Educational tool: describe symptoms → retrieve knowledge from your medical PDFs and Kaggle datasets → gpt-4o-mini suggests possible diseases and medicines with citations.

**Not medical advice.** Always consult a qualified clinician.

## Stack

- **Frontend:** Next.js 14 (App Router) + TypeScript + Tailwind
- **Backend:** FastAPI + LangChain + ChromaDB + OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **Data:** PDF books (admin upload) + Kaggle CSV ingest script

## Project layout

```
Symptom_Tracker/
  frontend/     # Next.js 14 UI
  backend/      # FastAPI RAG API
  data/
    uploads/    # PDF storage
    kaggle/     # CSV datasets
    chroma/     # vector store
  scripts/      # ingest_kaggle.py
```

## Prerequisites

- Node.js 18+
- Python **3.12** (recommended; 3.14 is too new for some ML deps)
- OpenAI API key

## Setup

### 1. Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and ADMIN_API_KEY
```

Start the API:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Health check: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Frontend

```bash
cd frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### 3. Ingest sample Kaggle-style CSV

With the backend venv active and `OPENAI_API_KEY` set:

```bash
cd backend
source .venv/bin/activate
python ../scripts/ingest_kaggle.py \
  --csv ../data/kaggle/sample_disease_symptoms.csv \
  --dataset-name disease_symptom_sample \
  --disease-col Disease \
  --symptoms-col Symptoms \
  --medicines-col Medicines
```

For your own Kaggle CSV, map column names with `--disease-col`, `--symptoms-col`, and `--medicines-col`. Wide `Symptom_1`, `Symptom_2`, … columns are auto-joined if the symptoms column is empty.

### 4. Upload PDF books

1. Open [http://localhost:3000/admin](http://localhost:3000/admin)
2. Enter the same `ADMIN_API_KEY` from `backend/.env`
3. Upload a PDF — it is chunked, embedded, and stored in Chroma

## API

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Status |
| POST | `/api/predict` | Hybrid BM25 + vectors, cross-encoder rerank, then generate |
| POST | `/api/ask-direct` | `{ "question": "..." }` — gpt-4o-mini with **no** retrieval |
| GET | `/api/history` | list recent consultations |
| GET | `/api/history/{id}` | load one saved consultation |
| DELETE | `/api/history/{id}` | delete a consultation |
| POST | `/api/admin/upload` | multipart PDF; header `X-Admin-Key` |
| GET | `/api/admin/sources` | list ingested sources; header `X-Admin-Key` |

## Environment

`backend/.env`:

```
OPENAI_API_KEY=
ADMIN_API_KEY=change-me-admin-secret
CHROMA_PATH=../data/chroma
UPLOAD_DIR=../data/uploads
CORS_ORIGINS=http://localhost:3000
```

`frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Smoke test

With the API running on port 8000:

```bash
cd backend
source .venv/bin/activate
python ../scripts/smoke_test.py
```

A sample medical PDF lives at `data/samples/sample_medical_notes.pdf` for upload testing. A sample CSV is at `data/kaggle/sample_disease_symptoms.csv`.

Live RAG (ingest + predict) requires a real `OPENAI_API_KEY` in `backend/.env`.