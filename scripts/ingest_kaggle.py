#!/usr/bin/env python3
"""Ingest a Kaggle-style CSV into the shared Chroma medical_knowledge collection.

Example:
  cd backend
  python ../scripts/ingest_kaggle.py \\
    --csv ../data/kaggle/sample_disease_symptoms.csv \\
    --dataset-name disease_symptom_sample \\
    --disease-col Disease \\
    --symptoms-col Symptoms \\
    --medicines-col Medicines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow importing backend app package when run from repo root or scripts/
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.models.db import SessionLocal, SourceRecord, init_db  # noqa: E402
from app.rag.ingest import ingest_text_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Kaggle CSV into Chroma RAG store")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--dataset-name", required=True, help="Label stored in metadata/source table")
    parser.add_argument("--disease-col", default="Disease", help="Column name for disease")
    parser.add_argument("--symptoms-col", default="Symptoms", help="Column name for symptoms")
    parser.add_argument("--medicines-col", default="Medicines", help="Column name for medicines")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    settings = get_settings()
    if not settings.openai_configured:
        raise SystemExit("OPENAI_API_KEY is required. Set it in backend/.env")

    df = pd.read_csv(csv_path)
    for col in (args.disease_col, args.symptoms_col):
        if col not in df.columns:
            raise SystemExit(
                f"Missing column '{col}'. Available columns: {list(df.columns)}"
            )

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    rows = []
    for _, record in df.iterrows():
        medicines = ""
        if args.medicines_col in df.columns and pd.notna(record.get(args.medicines_col)):
            medicines = str(record[args.medicines_col])
        symptoms_val = record.get(args.symptoms_col, "")
        # Support wide symptom columns like Symptom_1, Symptom_2... by joining if needed
        if pd.isna(symptoms_val):
            symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
            if symptom_cols:
                parts = [
                    str(record[c]).strip()
                    for c in symptom_cols
                    if pd.notna(record[c]) and str(record[c]).strip()
                ]
                symptoms_val = ", ".join(parts)
            else:
                symptoms_val = ""
        rows.append(
            {
                "disease": "" if pd.isna(record[args.disease_col]) else str(record[args.disease_col]),
                "symptoms": str(symptoms_val),
                "medicines": medicines,
            }
        )

    init_db()
    chunk_count = ingest_text_rows(rows, dataset_name=args.dataset_name, settings=settings)

    db = SessionLocal()
    try:
        db.add(
            SourceRecord(
                name=args.dataset_name,
                source_type="kaggle",
                path=str(csv_path),
                chunk_count=chunk_count,
            )
        )
        db.commit()
    finally:
        db.close()

    print(f"Ingested {chunk_count} chunks from {csv_path.name} as '{args.dataset_name}'.")


if __name__ == "__main__":
    main()
