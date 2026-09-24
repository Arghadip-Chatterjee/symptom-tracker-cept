#!/usr/bin/env python3
"""Local smoke checks for Symptom Tracker (no OpenAI calls unless key is real)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.rag.ingest import extract_pdf_documents, chunk_documents  # noqa: E402


def main() -> None:
    settings = get_settings()
    base = os.environ.get("API_BASE", "http://127.0.0.1:8000")
    failures = 0

    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{base}/health")
        print("GET /health", health.status_code, health.json())
        if health.status_code != 200:
            failures += 1

        denied = client.get(f"{base}/api/admin/sources")
        print("GET /api/admin/sources (no key)", denied.status_code)
        if denied.status_code != 401:
            failures += 1

        ok = client.get(
            f"{base}/api/admin/sources",
            headers={"X-Admin-Key": settings.admin_api_key},
        )
        print("GET /api/admin/sources (admin)", ok.status_code, ok.json())
        if ok.status_code != 200:
            failures += 1

        predict = client.post(f"{base}/api/predict", json={"symptoms": "fever and cough"})
        print("POST /api/predict", predict.status_code, predict.text[:200])
        # 503 without real key; 200 when configured; 502 on upstream OpenAI failures
        if predict.status_code not in (200, 502, 503):
            failures += 1

    pdf = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_medical_notes.pdf"
    if pdf.exists():
        docs = extract_pdf_documents(pdf, pdf.name)
        chunks = chunk_documents(docs)
        print(f"PDF extract: {len(docs)} pages -> {len(chunks)} chunks")
        if not docs:
            failures += 1
    else:
        print("Sample PDF missing")
        failures += 1

    if not settings.openai_configured:
        print("NOTE: Set a real OPENAI_API_KEY in backend/.env to run ingest + live predict.")
    else:
        print("OPENAI_API_KEY looks configured — you can run ingest_kaggle.py and upload PDFs.")

    if failures:
        raise SystemExit(f"Smoke test failed with {failures} issue(s)")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
