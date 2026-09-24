from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.db import SourceRecord, get_db
from app.models.schemas import SourceListItem, UploadResponse
from app.rag.ingest import ingest_pdf

router = APIRouter(prefix="/api/admin", tags=["admin"])


def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> None:
    settings = get_settings()
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
        )


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\-]+", "_", base)
    return base or "upload.pdf"


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_admin)])
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    settings = get_settings()
    if not settings.openai_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is required to embed uploaded PDFs.",
        )

    safe_name = _safe_filename(file.filename)
    dest = settings.upload_path / safe_name
    # Avoid overwrite collisions
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 1
        while True:
            candidate = settings.upload_path / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                safe_name = candidate.name
                break
            i += 1

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    dest.write_bytes(content)

    try:
        chunk_count = ingest_pdf(dest, source_name=safe_name)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF: {exc}",
        ) from exc

    record = SourceRecord(
        name=safe_name,
        source_type="pdf",
        path=str(dest),
        chunk_count=chunk_count,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return UploadResponse(
        id=record.id,
        name=record.name,
        chunk_count=record.chunk_count,
        message=f"Ingested {chunk_count} chunks from {safe_name}.",
    )


@router.get("/sources", response_model=list[SourceListItem], dependencies=[Depends(require_admin)])
def list_sources(db: Session = Depends(get_db)) -> list[SourceListItem]:
    rows = db.query(SourceRecord).order_by(SourceRecord.id.desc()).all()
    return [
        SourceListItem(
            id=r.id,
            name=r.name,
            source_type=r.source_type,
            path=r.path,
            chunk_count=r.chunk_count,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]
