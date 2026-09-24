import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.db import ConsultationRecord, get_db
from app.models.schemas import (
    ConsultationDetail,
    ConsultationSummary,
    DirectAskRequest,
    DirectAskResponse,
    PredictRequest,
    PredictResponse,
)
from app.rag.chain import ask_llm_direct, predict_from_symptoms

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    settings = get_settings()
    if not body.symptoms.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symptoms text is required.",
        )
    if not settings.openai_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured on the server.",
        )
    try:
        result = predict_from_symptoms(body.symptoms, settings)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Prediction failed: {exc}",
        ) from exc

    record = ConsultationRecord(
        symptoms=body.symptoms.strip(),
        response_json=result.model_dump_json(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    result.consultation_id = record.id
    return result


@router.post("/ask-direct", response_model=DirectAskResponse)
def ask_direct(body: DirectAskRequest) -> DirectAskResponse:
    settings = get_settings()
    if not body.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question text is required.",
        )
    if not settings.openai_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured on the server.",
        )
    try:
        return ask_llm_direct(body.question, settings)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Direct LLM request failed: {exc}",
        ) from exc


@router.get("/history", response_model=list[ConsultationSummary])
def list_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[ConsultationSummary]:
    rows = (
        db.query(ConsultationRecord)
        .order_by(ConsultationRecord.id.desc())
        .limit(limit)
        .all()
    )
    items: list[ConsultationSummary] = []
    for row in rows:
        disease_names: list[str] = []
        try:
            payload = json.loads(row.response_json or "{}")
            disease_names = [
                d.get("name", "")
                for d in payload.get("diseases", [])
                if isinstance(d, dict) and d.get("name")
            ]
        except json.JSONDecodeError:
            disease_names = []
        items.append(
            ConsultationSummary(
                id=row.id,
                symptoms=row.symptoms,
                created_at=row.created_at.isoformat() if row.created_at else "",
                disease_names=disease_names,
            )
        )
    return items


@router.get("/history/{consultation_id}", response_model=ConsultationDetail)
def get_history_item(
    consultation_id: int,
    db: Session = Depends(get_db),
) -> ConsultationDetail:
    row = db.query(ConsultationRecord).filter(ConsultationRecord.id == consultation_id).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Consultation not found.")
    try:
        result = PredictResponse.model_validate_json(row.response_json)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stored consultation is invalid: {exc}",
        ) from exc
    result.consultation_id = row.id
    return ConsultationDetail(
        id=row.id,
        symptoms=row.symptoms,
        created_at=row.created_at.isoformat() if row.created_at else "",
        result=result,
    )


@router.delete("/history/{consultation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(
    consultation_id: int,
    db: Session = Depends(get_db),
) -> None:
    row = db.query(ConsultationRecord).filter(ConsultationRecord.id == consultation_id).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Consultation not found.")
    db.delete(row)
    db.commit()
