export type MedicineInfo = {
  name: string;
  notes?: string | null;
};

export type DiseasePrediction = {
  name: string;
  likelihood: string;
  rationale: string;
  medicines: MedicineInfo[];
};

export type SourceCitation = {
  title: string;
  snippet: string;
  source_type: string;
  vector_distance?: number | null;
  bm25_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  match_confidence?: number | null;
};

export type RetrievalPipeline = {
  vector_asked: number;
  vector_kept: number;
  bm25_asked: number;
  bm25_kept: number;
  rrf_merged: number;
  rerank_input: number;
  final_kept: number;
  reranker_used: boolean;
};

export type PredictResponse = {
  diseases: DiseasePrediction[];
  sources: SourceCitation[];
  disclaimer: string;
  raw_answer?: string | null;
  consultation_id?: number | null;
  used_rag?: boolean;
  out_of_scope?: boolean;
  allow_direct_llm?: boolean;
  scope_reason?: string | null;
  pipeline?: RetrievalPipeline | null;
};

export type DirectAskResponse = {
  answer: string;
  disclaimer: string;
  used_rag: boolean;
};

export type SourceListItem = {
  id: number;
  name: string;
  source_type: string;
  path: string;
  chunk_count: number;
  created_at: string;
};

export type ConsultationSummary = {
  id: number;
  symptoms: string;
  created_at: string;
  disease_names: string[];
};

export type ConsultationDetail = {
  id: number;
  symptoms: string;
  created_at: string;
  result: PredictResponse;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function errorMessage(err: unknown, fallback: string): string {
  const detail = (err as { detail?: unknown })?.detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join("; ");
  }
  if (typeof detail === "string") return detail;
  return fallback;
}

export async function predictSymptoms(symptoms: string): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Prediction failed"));
  }
  return res.json();
}

export async function askLlmDirect(question: string): Promise<DirectAskResponse> {
  const res = await fetch(`${API_URL}/api/ask-direct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Direct LLM request failed"));
  }
  return res.json();
}

export async function listHistory(limit = 20): Promise<ConsultationSummary[]> {
  const res = await fetch(`${API_URL}/api/history?limit=${limit}`, { cache: "no-store" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Failed to load history"));
  }
  return res.json();
}

export async function getConsultation(id: number): Promise<ConsultationDetail> {
  const res = await fetch(`${API_URL}/api/history/${id}`, { cache: "no-store" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Failed to load consultation"));
  }
  return res.json();
}

export async function deleteConsultation(id: number): Promise<void> {
  const res = await fetch(`${API_URL}/api/history/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Failed to delete consultation"));
  }
}

export async function listSources(adminKey: string): Promise<SourceListItem[]> {
  const res = await fetch(`${API_URL}/api/admin/sources`, {
    headers: { "X-Admin-Key": adminKey },
    cache: "no-store",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Failed to load sources"));
  }
  return res.json();
}

export async function uploadPdf(file: File, adminKey: string) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/admin/upload`, {
    method: "POST",
    headers: { "X-Admin-Key": adminKey },
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorMessage(err, "Upload failed"));
  }
  return res.json();
}
