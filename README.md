# Symptom Tracker

Educational symptom checker. You describe how you feel, the app searches your own medical PDFs and disease–symptom datasets, and `gpt-4o-mini` suggests possible conditions and medicines **only from those sources**, with citations and retrieval scores.

**This is not medical advice.** It does not diagnose, prescribe, or invent drug dosages. Always consult a qualified clinician.

## Features

### Symptom analysis

- Free-text symptom description (3–4000 characters).
- Structured results: possible conditions, a likelihood (`high`, `moderate`, or `low`), a short rationale grounded in retrieved text, and medicines mentioned in the sources.
- A medical disclaimer on every answer.

### Grounded answers (RAG)

The model is instructed to use only retrieved passages. If the knowledge base does not support an answer, it returns no conditions instead of guessing.

Two cases skip the knowledge base and offer a separate path:

- The message is not a health question (sports, coding, trivia, and similar).
- No passages are similar enough to the symptoms.

In those cases the UI shows why RAG did not answer and a button to **ask the language model directly**. That reply is labeled as generated without retrieval.

### Hybrid retrieval

Each analysis shows how many passages survived each stage:

1. **Vector search** — semantic similarity with OpenAI `text-embedding-3-small` in Chroma (`vector_candidate_k`, default 24).
2. **BM25** — keyword match over the same corpus (`bm25_candidate_k`, default 24).
3. **Reciprocal rank fusion** — merges the two ranked lists (`rrf_k`, default 60).
4. **Cross-encoder rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores the fused pool. If the reranker cannot load, ranking falls back to vector distance and BM25.
5. **Final context** — up to `top_k` passages (default 6) are sent to `gpt-4o-mini`. Weak rerank scores below `min_rerank_score` (default `-2.0`) are dropped.

Each cited source shows:

- **Match confidence** (0–100), derived from the cross-encoder score.
- **Vector L2 distance** (lower means closer).
- **BM25 score** (higher means more keyword overlap).
- **Rerank score** (cross-encoder logit).

### Consultation history

Every `/api/predict` call is stored in SQLite. The home page lists recent consultations, opens one to restore the original symptoms and result, and can delete an entry. History is local to this server; there is no user login.

### Knowledge sources

Two ways to fill the knowledge base. Both write into the same Chroma collection (`medical_knowledge`) and a `sources` table:

- **PDF upload** on `/admin`. Pages are extracted, split into chunks of about 800 characters with 150 characters of overlap, embedded, and stored. Requires the admin key.
- **CSV ingest** via `scripts/ingest_kaggle.py` for Kaggle-style disease / symptoms / medicines tables. Wide `Symptom_1`, `Symptom_2`, … columns are joined when the symptoms cell is empty.

The admin page lists every ingested source with type, chunk count, and time.

### Health check and smoke test

`GET /health` reports API status and whether a real OpenAI key is set. `scripts/smoke_test.py` checks health, admin auth, predict, and sample PDF extraction without requiring a live model call.

## How a query moves through the system

```
Symptoms
  → health-topic classifier (gpt-4o-mini)
  → if not health, or no good passages: stop and offer direct LLM
  → hybrid retrieve (vector + BM25 → RRF → cross-encoder)
  → gpt-4o-mini answers from those passages only
  → JSON: conditions, likelihood, rationale, medicines
  → saved consultation + citations shown in the UI
```

## Stack

| Layer | Choice |
| --- | --- |
| UI | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS |
| API | FastAPI, Pydantic, Uvicorn |
| Retrieval | LangChain, ChromaDB, rank-BM25, sentence-transformers cross-encoder |
| Models | OpenAI `gpt-4o-mini` (chat), `text-embedding-3-small` (embeddings) |
| Storage | SQLite for consultations and source metadata; Chroma on disk for vectors; uploaded PDFs on disk |

## Project layout

```
symptom-tracker-cept/
  frontend/          Next.js UI (home analyzer + /admin)
  backend/           FastAPI app
    app/routers/     predict, history, admin upload
    app/rag/         ingest, hybrid retriever, generation
    app/models/      SQLite models and API schemas
  data/
    uploads/         saved PDFs
    kaggle/          CSV datasets (includes a sample)
    chroma/          vector store (created on first run)
    samples/         sample_medical_notes.pdf
  scripts/
    ingest_kaggle.py
    smoke_test.py
```

SQLite defaults to `data/symptom_tracker.db`.

## Prerequisites

- Node.js 18+
- Python **3.12** (3.14 is too new for some of the ML dependencies)
- An OpenAI API key (required for embeddings, classification, and answers)

## Setup

### 1. Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env` and set `OPENAI_API_KEY` and a real `ADMIN_API_KEY`.

Start the API:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Health check: [http://localhost:8000/health](http://localhost:8000/health)

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The admin screen is [http://localhost:3000/admin](http://localhost:3000/admin).

`NEXT_PUBLIC_API_URL` must point at the API (default `http://localhost:8000`).

### 3. Load the sample dataset

With the backend virtualenv active and a real `OPENAI_API_KEY` in `backend/.env`:

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

The sample CSV includes short rows for conditions such as common cold, influenza, migraine, and asthma exacerbation. `--limit N` ingests only the first N rows.

For another CSV, map columns with `--disease-col`, `--symptoms-col`, and `--medicines-col`.

### 4. Upload a PDF

1. Open [http://localhost:3000/admin](http://localhost:3000/admin).
2. Enter the same `ADMIN_API_KEY` from `backend/.env`. The key is kept in `sessionStorage` for this browser tab.
3. Upload a PDF. It is chunked, embedded, and added to Chroma.

A small sample PDF is at `data/samples/sample_medical_notes.pdf`.

## Using the app

1. Start the API and the frontend.
2. Ingest at least one CSV or PDF. Without sources, analysis will say no similar passages were found.
3. On the home page, describe symptoms (for example: fever, body aches, dry cough, and fatigue for 2 days) and choose **Analyze symptoms**.
4. Read the disclaimer, the retrieval counts, possible conditions, and the source cards.
5. Reopen or delete past consultations from the list at the bottom of the page.
6. If RAG refuses the question, use **Ask LLM directly** only when you want an ungrounded model reply.

## API

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | Status, collection name, whether OpenAI is configured |
| POST | `/api/predict` | Body `{ "symptoms": "..." }`. Runs the full RAG flow and saves a consultation |
| POST | `/api/ask-direct` | Body `{ "question": "..." }`. `gpt-4o-mini` with no retrieval |
| GET | `/api/history?limit=20` | Recent consultations (`limit` 1–100) |
| GET | `/api/history/{id}` | One saved consultation and its full result |
| DELETE | `/api/history/{id}` | Delete a consultation |
| POST | `/api/admin/upload` | Multipart PDF. Header `X-Admin-Key` |
| GET | `/api/admin/sources` | Ingested sources. Header `X-Admin-Key` |

Predict returns `503` when `OPENAI_API_KEY` is missing or still a placeholder, and `502` when the model or embedding call fails.

A predict response includes `diseases`, `sources` (with scores), `disclaimer`, `consultation_id`, `pipeline` counts, and flags `used_rag`, `out_of_scope`, and `allow_direct_llm`.

## Environment

`backend/.env`:

```
OPENAI_API_KEY=
ADMIN_API_KEY=change-me-admin-secret
CHROMA_PATH=../data/chroma
UPLOAD_DIR=../data/uploads
CORS_ORIGINS=http://localhost:3000
```

Optional backend settings (defaults live in `backend/app/config.py`):

| Variable | Default | Role |
| --- | --- | --- |
| `DATABASE_URL` | SQLite file under `data/` | Consultation and source metadata |
| `COLLECTION_NAME` | `medical_knowledge` | Chroma collection |
| `TOP_K` | `6` | Passages sent to the model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat and health classifier |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings |

Retrieval knobs `retrieve_k` (12), `vector_candidate_k` (24), `bm25_candidate_k` (24), `rrf_k` (60), `min_rerank_score` (-2.0), `max_l2_distance` (1.25), and `reranker_model` are code defaults in the same settings class.

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

Live ingest and predict need a real `OPENAI_API_KEY`. The smoke test still passes without one: predict is expected to return `503` until the key is set.

## Limits

- Educational prototype. No accounts, no access control on consultation history, and the admin key is a shared secret.
- Answers depend entirely on what you upload. The sample CSV is tiny and not a clinical reference.
- PDFs must contain extractable text. Scanned image-only PDFs produce no chunks.
- The first cross-encoder run downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Medicines are names mentioned in your sources, not dosing instructions.
