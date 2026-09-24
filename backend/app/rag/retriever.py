from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from threading import Lock

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.rag.store import get_vectorstore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_index_lock = Lock()
_bm25_cache: dict[str, object] = {"count": -1, "docs": [], "bm25": None}
_reranker = None
_reranker_lock = Lock()


@dataclass
class RetrievalTrace:
    vector_asked: int = 0
    vector_kept: int = 0
    bm25_asked: int = 0
    bm25_kept: int = 0
    rrf_merged: int = 0
    rerank_input: int = 0
    final_kept: int = 0
    reranker_used: bool = False


@dataclass
class RetrievedHit:
    doc: Document
    vector_distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    match_confidence: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _doc_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('source','')}|{meta.get('page','')}|{meta.get('row','')}|{hash(doc.page_content)}"


def invalidate_lexical_index() -> None:
    with _index_lock:
        _bm25_cache["count"] = -1
        _bm25_cache["docs"] = []
        _bm25_cache["bm25"] = None


def _load_corpus(settings: Settings):
    from rank_bm25 import BM25Okapi

    store = get_vectorstore(settings)
    raw = store.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    docs: list[Document] = []
    tokenized: list[list[str]] = []
    for i, text in enumerate(texts):
        if not text or not str(text).strip():
            continue
        meta = metas[i] if i < len(metas) and metas[i] else {}
        if ids and i < len(ids):
            meta = {**meta, "chroma_id": ids[i]}
        docs.append(Document(page_content=str(text), metadata=meta))
        tokenized.append(_tokenize(str(text)))
    if not tokenized:
        tokenized = [[""]]
        docs = []
    return docs, BM25Okapi(tokenized)


def _get_bm25(settings: Settings):
    store = get_vectorstore(settings)
    try:
        count = store._collection.count()
    except Exception:
        count = 0
    with _index_lock:
        if _bm25_cache["bm25"] is not None and _bm25_cache["count"] == count:
            return _bm25_cache["docs"], _bm25_cache["bm25"]  # type: ignore[return-value]
        docs, bm25 = _load_corpus(settings)
        _bm25_cache["count"] = count
        _bm25_cache["docs"] = docs
        _bm25_cache["bm25"] = bm25
        return docs, bm25


def _get_reranker(settings: Settings):
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.reranker_model)
        return _reranker


def retrieve_context(query: str, settings: Settings | None = None) -> list[Document]:
    hits, _trace = hybrid_retrieve(query, settings)
    return [hit.doc for hit in hits]


def retrieve_context_with_scores(
    query: str,
    settings: Settings | None = None,
) -> list[tuple[Document, float]]:
    hits, _trace = hybrid_retrieve(query, settings)
    return [(h.doc, h.vector_distance if h.vector_distance is not None else 999.0) for h in hits]


def filter_relevant(
    scored: list[tuple[Document, float]],
    settings: Settings | None = None,
) -> list[tuple[Document, float]]:
    settings = settings or get_settings()
    relevant = [(doc, score) for doc, score in scored if score <= settings.max_l2_distance]
    return relevant[: settings.top_k]


def _vector_search(query: str, settings: Settings) -> list[tuple[Document, float]]:
    store = get_vectorstore(settings)
    try:
        results = store.similarity_search_with_score(query, k=settings.vector_candidate_k)
    except Exception:
        return []
    return [(doc, float(score)) for doc, score in results]


def _bm25_search(query: str, settings: Settings) -> list[tuple[Document, float]]:
    try:
        docs, bm25 = _get_bm25(settings)
    except ImportError:
        return []
    if not docs:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    out: list[tuple[Document, float]] = []
    for idx, score in ranked[: settings.bm25_candidate_k]:
        if score <= 0:
            break
        out.append((docs[idx], float(score)))
    return out


def _rrf_fuse(
    vector_hits: list[tuple[Document, float]],
    bm25_hits: list[tuple[Document, float]],
    settings: Settings,
) -> list[RetrievedHit]:
    merged: dict[str, RetrievedHit] = {}
    rrf_k = settings.rrf_k

    for rank, (doc, dist) in enumerate(vector_hits, start=1):
        key = _doc_key(doc)
        hit = merged.get(key)
        if hit is None:
            hit = RetrievedHit(doc=doc)
            merged[key] = hit
        hit.vector_distance = dist
        hit.ranks["vector"] = rank
        hit.rrf_score += 1.0 / (rrf_k + rank)

    for rank, (doc, score) in enumerate(bm25_hits, start=1):
        key = _doc_key(doc)
        hit = merged.get(key)
        if hit is None:
            hit = RetrievedHit(doc=doc)
            merged[key] = hit
        hit.bm25_score = score
        hit.ranks["bm25"] = rank
        hit.rrf_score += 1.0 / (rrf_k + rank)

    return sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)


def _confidence_from_rerank(score: float) -> float:
    # MiniLM MS MARCO logits → 0–100 via sigmoid
    return round(100.0 / (1.0 + math.exp(-score)), 1)


def _attach_metadata(hit: RetrievedHit) -> Document:
    meta = dict(hit.doc.metadata or {})
    meta["vector_distance"] = hit.vector_distance
    meta["bm25_score"] = hit.bm25_score
    meta["rrf_score"] = round(hit.rrf_score, 6)
    meta["rerank_score"] = hit.rerank_score
    meta["match_confidence"] = hit.match_confidence
    return Document(page_content=hit.doc.page_content, metadata=meta)


def hybrid_retrieve(
    query: str, settings: Settings | None = None
) -> tuple[list[RetrievedHit], RetrievalTrace]:
    """Vector + BM25 (RRF) → cross-encoder rerank → top_k."""
    settings = settings or get_settings()
    trace = RetrievalTrace(
        vector_asked=settings.vector_candidate_k,
        bm25_asked=settings.bm25_candidate_k,
    )
    vector_hits = _vector_search(query, settings)
    bm25_hits = _bm25_search(query, settings)
    trace.vector_kept = len(vector_hits)
    trace.bm25_kept = len(bm25_hits)
    fused = _rrf_fuse(vector_hits, bm25_hits, settings)
    trace.rrf_merged = len(fused)
    if not fused:
        return [], trace

    pool = fused[: max(settings.retrieve_k, settings.top_k)]
    trace.rerank_input = len(pool)
    reranker_used = False
    try:
        reranker = _get_reranker(settings)
        pairs = [(query, hit.doc.page_content[:4000]) for hit in pool]
        scores = reranker.predict(pairs)
        for hit, score in zip(pool, scores):
            hit.rerank_score = float(score)
            hit.match_confidence = _confidence_from_rerank(hit.rerank_score)
        pool.sort(key=lambda h: h.rerank_score or -999.0, reverse=True)
        reranker_used = True
    except Exception:
        for hit in pool:
            if hit.vector_distance is not None:
                sim = max(0.0, 1.0 - (hit.vector_distance / max(settings.max_l2_distance * 1.5, 0.01)))
                hit.match_confidence = round(min(100.0, sim * 100.0), 1)
            elif hit.bm25_score:
                hit.match_confidence = round(min(100.0, hit.bm25_score * 5), 1)
    trace.reranker_used = reranker_used

    kept: list[RetrievedHit] = []
    for hit in pool:
        if hit.rerank_score is not None and hit.rerank_score < settings.min_rerank_score:
            continue
        if (
            hit.rerank_score is None
            and hit.vector_distance is not None
            and hit.vector_distance > settings.max_l2_distance
            and not hit.bm25_score
        ):
            continue
        hit.doc = _attach_metadata(hit)
        kept.append(hit)
        if len(kept) >= settings.top_k:
            break
    trace.final_kept = len(kept)
    return kept, trace
