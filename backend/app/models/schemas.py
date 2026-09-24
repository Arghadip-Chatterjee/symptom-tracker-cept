from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, max_length=4000)


class DirectAskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)


class MedicineInfo(BaseModel):
    name: str
    notes: str | None = None


class DiseasePrediction(BaseModel):
    name: str
    likelihood: str
    rationale: str
    medicines: list[MedicineInfo] = Field(default_factory=list)


class SourceCitation(BaseModel):
    title: str
    snippet: str
    source_type: str
    vector_distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    match_confidence: float | None = None


class RetrievalPipeline(BaseModel):
    vector_asked: int = 0
    vector_kept: int = 0
    bm25_asked: int = 0
    bm25_kept: int = 0
    rrf_merged: int = 0
    rerank_input: int = 0
    final_kept: int = 0
    reranker_used: bool = False


class PredictResponse(BaseModel):
    diseases: list[DiseasePrediction]
    sources: list[SourceCitation]
    disclaimer: str
    raw_answer: str | None = None
    consultation_id: int | None = None
    used_rag: bool = True
    out_of_scope: bool = False
    allow_direct_llm: bool = False
    scope_reason: str | None = None
    pipeline: RetrievalPipeline | None = None


class DirectAskResponse(BaseModel):
    answer: str
    disclaimer: str
    used_rag: bool = False


class SourceListItem(BaseModel):
    id: int
    name: str
    source_type: str
    path: str
    chunk_count: int
    created_at: str


class UploadResponse(BaseModel):
    id: int
    name: str
    chunk_count: int
    message: str


class ConsultationSummary(BaseModel):
    id: int
    symptoms: str
    created_at: str
    disease_names: list[str] = Field(default_factory=list)


class ConsultationDetail(BaseModel):
    id: int
    symptoms: str
    created_at: str
    result: PredictResponse
