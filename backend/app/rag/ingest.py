from __future__ import annotations

import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings, get_settings
from app.rag.store import get_vectorstore
from app.rag.retriever import invalidate_lexical_index


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_documents(pdf_path: Path, source_name: str) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _clean_text(raw)
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "type": "pdf",
                    "page": page_idx,
                    "path": str(pdf_path),
                },
            )
        )
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def upsert_documents(
    documents: list[Document],
    settings: Settings | None = None,
) -> int:
    if not documents:
        return 0
    settings = settings or get_settings()
    store = get_vectorstore(settings)
    ids = [str(uuid.uuid4()) for _ in documents]
    store.add_documents(documents=documents, ids=ids)
    invalidate_lexical_index()
    return len(documents)


def ingest_pdf(pdf_path: Path, source_name: str | None = None) -> int:
    settings = get_settings()
    name = source_name or pdf_path.name
    pages = extract_pdf_documents(pdf_path, name)
    chunks = chunk_documents(pages)
    return upsert_documents(chunks, settings)


def ingest_text_rows(
    rows: list[dict],
    dataset_name: str,
    settings: Settings | None = None,
) -> int:
    """
    Each row dict should include keys used to build a prose document.
    Expected keys: disease, symptoms, medicines (optional extras allowed).
    """
    settings = settings or get_settings()
    documents: list[Document] = []
    for idx, row in enumerate(rows):
        disease = str(row.get("disease", "")).strip()
        symptoms = str(row.get("symptoms", "")).strip()
        medicines = str(row.get("medicines", "")).strip()
        if not disease and not symptoms:
            continue
        parts = []
        if disease:
            parts.append(f"Disease: {disease}.")
        if symptoms:
            parts.append(f"Symptoms: {symptoms}.")
        if medicines:
            parts.append(f"Common medicines: {medicines}.")
        content = " ".join(parts)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": dataset_name,
                    "type": "kaggle",
                    "row": idx,
                    "disease": disease or "unknown",
                },
            )
        )
    chunks = chunk_documents(documents) if documents else []
    # Dataset rows are already short; chunking still normalizes size.
    return upsert_documents(chunks or documents, settings)
