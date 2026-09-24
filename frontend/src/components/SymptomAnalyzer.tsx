"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  ConsultationSummary,
  DirectAskResponse,
  RetrievalPipeline,
  SourceCitation,
  deleteConsultation,
  getConsultation,
  askLlmDirect,
  listHistory,
  predictSymptoms,
  PredictResponse,
} from "@/lib/api";

function likelihoodClass(value: string) {
  const v = value.toLowerCase();
  if (v === "high") return "likelihood-high";
  if (v === "moderate") return "likelihood-moderate";
  return "likelihood-low";
}

function formatWhen(iso: string) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function confidenceTone(value: number) {
  if (value >= 55) return "text-emerald-800 bg-emerald-50 border-emerald-200";
  if (value >= 35) return "text-amber-900 bg-amber-50 border-amber-200";
  return "text-stone-700 bg-stone-100 border-stone-200";
}

function RetrievalFlow({ pipeline }: { pipeline: RetrievalPipeline }) {
  const steps = [
    {
      label: "1. Vector search",
      detail: `Asked ${pipeline.vector_asked}`,
      count: pipeline.vector_kept,
      unit: "chunks",
    },
    {
      label: "2. BM25 keywords",
      detail: `Asked ${pipeline.bm25_asked}`,
      count: pipeline.bm25_kept,
      unit: "chunks",
    },
    {
      label: "3. RRF fusion",
      detail: "Merged unique ranks",
      count: pipeline.rrf_merged,
      unit: "unique",
    },
    {
      label: "4. Cross-encoder",
      detail: pipeline.reranker_used ? "MiniLM rerank" : "Reranker skipped",
      count: pipeline.rerank_input,
      unit: "scored",
    },
    {
      label: "5. Final context",
      detail: "Sent to gpt-4o-mini",
      count: pipeline.final_kept,
      unit: "chunks",
    },
  ];

  return (
    <section className="rounded-md border border-[var(--line)] bg-white/80 px-4 py-4">
      <h2 className="font-display text-2xl text-[var(--ink)]">Retrieval pipeline</h2>
      <p className="mt-1 text-sm text-[var(--ink-muted)]">
        How many passages were kept at each stage for this query.
      </p>
      <ol className="mt-4 grid gap-3 sm:grid-cols-5">
        {steps.map((step) => (
          <li
            key={step.label}
            className="rounded-md border border-[var(--line)] bg-[var(--background)] px-3 py-3"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-[var(--accent-dark)]">
              {step.label}
            </p>
            <p className="mt-2 text-3xl font-semibold tabular-nums text-[var(--ink)]">{step.count}</p>
            <p className="text-xs text-[var(--ink-muted)]">{step.unit}</p>
            <p className="mt-2 text-xs text-[var(--ink-muted)]">{step.detail}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function SourceScore({ source }: { source: SourceCitation }) {
  const conf = source.match_confidence;
  return (
    <div className="mt-3 space-y-2">
      {conf != null && (
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="font-semibold text-[var(--ink)]">Match confidence</span>
            <span className="tabular-nums font-semibold text-[var(--ink)]">{conf.toFixed(0)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-stone-200">
            <div
              className="h-full rounded-full bg-[var(--accent)]"
              style={{ width: `${Math.max(4, Math.min(100, conf))}%` }}
            />
          </div>
        </div>
      )}
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <div className={`rounded-md border px-2 py-2 ${conf != null ? confidenceTone(conf) : "border-[var(--line)]"}`}>
          <dt className="text-[10px] font-semibold uppercase tracking-wide">Confidence</dt>
          <dd className="mt-0.5 text-lg font-semibold tabular-nums">
            {conf != null ? `${conf.toFixed(0)}%` : "—"}
          </dd>
        </div>
        <div className="rounded-md border border-[var(--line)] bg-white px-2 py-2">
          <dt className="text-[10px] font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
            Vector L2
          </dt>
          <dd className="mt-0.5 text-lg font-semibold tabular-nums text-[var(--ink)]">
            {source.vector_distance != null ? source.vector_distance.toFixed(3) : "—"}
          </dd>
          <p className="text-[10px] text-[var(--ink-muted)]">Lower is closer</p>
        </div>
        <div className="rounded-md border border-[var(--line)] bg-white px-2 py-2">
          <dt className="text-[10px] font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
            BM25
          </dt>
          <dd className="mt-0.5 text-lg font-semibold tabular-nums text-[var(--ink)]">
            {source.bm25_score != null ? source.bm25_score.toFixed(2) : "—"}
          </dd>
          <p className="text-[10px] text-[var(--ink-muted)]">Higher is more keywords</p>
        </div>
        <div className="rounded-md border border-[var(--line)] bg-white px-2 py-2">
          <dt className="text-[10px] font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
            Rerank
          </dt>
          <dd className="mt-0.5 text-lg font-semibold tabular-nums text-[var(--ink)]">
            {source.rerank_score != null ? source.rerank_score.toFixed(2) : "—"}
          </dd>
          <p className="text-[10px] text-[var(--ink-muted)]">Cross-encoder logit</p>
        </div>
      </dl>
    </div>
  );
}

export default function SymptomAnalyzer() {
  const [symptoms, setSymptoms] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [history, setHistory] = useState<ConsultationSummary[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [directAnswer, setDirectAnswer] = useState<DirectAskResponse | null>(null);
  const [directLoading, setDirectLoading] = useState(false);

  const refreshHistory = useCallback(async () => {
    try {
      const rows = await listHistory(30);
      setHistory(rows);
      setHistoryError(null);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Could not load history");
    }
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setDirectAnswer(null);
    setLoading(true);
    try {
      const data = await predictSymptoms(symptoms.trim());
      setResult(data);
      setActiveId(data.consultation_id ?? null);
      await refreshHistory();
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function openConsultation(id: number) {
    setError(null);
    setLoading(true);
    try {
      const detail = await getConsultation(id);
      setSymptoms(detail.symptoms);
      setResult(detail.result);
      setDirectAnswer(null);
      setActiveId(detail.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open consultation");
    } finally {
      setLoading(false);
    }
  }

  async function removeConsultation(id: number) {
    setError(null);
    try {
      await deleteConsultation(id);
      if (activeId === id) {
        setActiveId(null);
        setResult(null);
        setDirectAnswer(null);
      }
      await refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete consultation");
    }
  }

  async function onAskDirect() {
    setError(null);
    setDirectLoading(true);
    try {
      const data = await askLlmDirect(symptoms.trim());
      setDirectAnswer(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Direct LLM request failed");
    } finally {
      setDirectLoading(false);
    }
  }

  return (
    <div className="space-y-10">
      <form onSubmit={onSubmit} className="space-y-4">
        <label htmlFor="symptoms" className="block text-sm font-medium text-[var(--ink-muted)]">
          Describe your symptoms
        </label>
        <textarea
          id="symptoms"
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          rows={5}
          placeholder="Example: fever, body aches, dry cough, and fatigue for 2 days"
          className="w-full resize-y rounded-md border border-[var(--line)] bg-white/80 px-4 py-3 text-[var(--ink)] shadow-sm outline-none ring-[var(--accent)] transition focus:ring-2"
          required
          minLength={3}
        />
        <button
          type="submit"
          disabled={loading || symptoms.trim().length < 3}
          className="inline-flex items-center justify-center rounded-md bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--accent-dark)] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Analyzing…" : "Analyze symptoms"}
        </button>
      </form>

      {error && (
        <p className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </p>
      )}

      {result && (
        <section className="space-y-6 animate-fade-up">
          {activeId != null && (
            <p className="text-xs text-[var(--ink-muted)]">Saved consultation #{activeId}</p>
          )}
          <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
            {result.disclaimer}
          </div>

          {result.pipeline && <RetrievalFlow pipeline={result.pipeline} />}

          {result.out_of_scope ? (
            <div className="space-y-4 rounded-md border border-[var(--line)] bg-white/70 px-4 py-4">
              <h2 className="font-display text-2xl text-[var(--ink)]">RAG did not answer</h2>
              <p className="text-sm leading-relaxed text-[var(--ink-muted)]">
                {result.scope_reason || result.raw_answer}
              </p>
              {result.allow_direct_llm && (
                <button
                  type="button"
                  onClick={onAskDirect}
                  disabled={directLoading || symptoms.trim().length < 3}
                  className="inline-flex items-center justify-center rounded-md border border-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-[var(--accent-dark)] transition hover:bg-[var(--wash)] disabled:opacity-60"
                >
                  {directLoading ? "Asking LLM…" : "Ask LLM directly (without RAG)"}
                </button>
              )}
            </div>
          ) : (
            <>
              {result.raw_answer && (
                <p className="text-sm text-[var(--ink-muted)]">{result.raw_answer}</p>
              )}

              <div className="space-y-4">
                <h2 className="font-display text-2xl text-[var(--ink)]">Possible conditions</h2>
                {result.diseases.length === 0 ? (
                  <p className="text-sm text-[var(--ink-muted)]">
                    No confident matches from the knowledge base. Try adding more detail or upload
                    relevant medical sources.
                  </p>
                ) : (
                  <ul className="space-y-4">
                    {result.diseases.map((disease) => (
                      <li
                        key={disease.name}
                        className="border-b border-[var(--line)] pb-4 last:border-0"
                      >
                        <div className="flex flex-wrap items-baseline gap-3">
                          <h3 className="text-lg font-semibold text-[var(--ink)]">{disease.name}</h3>
                          <span
                            className={`text-xs font-semibold uppercase tracking-wide ${likelihoodClass(disease.likelihood)}`}
                          >
                            {disease.likelihood}
                          </span>
                        </div>
                        <p className="mt-2 text-sm leading-relaxed text-[var(--ink-muted)]">
                          {disease.rationale}
                        </p>
                        {disease.medicines.length > 0 && (
                          <div className="mt-3">
                            <p className="text-xs font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
                              Medicines mentioned in sources
                            </p>
                            <ul className="mt-1 space-y-1 text-sm text-[var(--ink)]">
                              {disease.medicines.map((med) => (
                                <li key={`${disease.name}-${med.name}`}>
                                  <span className="font-medium">{med.name}</span>
                                  {med.notes ? (
                                    <span className="text-[var(--ink-muted)]"> — {med.notes}</span>
                                  ) : null}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="space-y-3">
                <h2 className="font-display text-2xl text-[var(--ink)]">Sources</h2>
                {result.sources.length === 0 ? (
                  <p className="text-sm text-[var(--ink-muted)]">
                    No retrieved chunks. Ingest a PDF or Kaggle CSV first.
                  </p>
                ) : (
                  <ul className="space-y-4">
                    {result.sources.map((source, idx) => (
                      <li
                        key={`${source.title}-${idx}`}
                        className="rounded-md border border-[var(--line)] bg-white/80 px-4 py-4"
                      >
                        <p className="font-medium text-[var(--ink)]">
                          {source.title}{" "}
                          <span className="text-xs font-normal uppercase text-[var(--ink-muted)]">
                            ({source.source_type})
                          </span>
                        </p>
                        <SourceScore source={source} />
                        <p className="mt-3 text-sm leading-relaxed text-[var(--ink-muted)]">
                          {source.snippet}
                        </p>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}

          {directAnswer && (
            <div className="space-y-2 rounded-md border border-[var(--line)] px-4 py-4">
              <h2 className="font-display text-2xl text-[var(--ink)]">LLM answer (no RAG)</h2>
              <p className="text-xs text-[var(--ink-muted)]">{directAnswer.disclaimer}</p>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-[var(--ink)]">
                {directAnswer.answer}
              </p>
            </div>
          )}
        </section>
      )}

      <section className="space-y-4 border-t border-[var(--line)] pt-8">
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="font-display text-2xl text-[var(--ink)]">Past consultations</h2>
          <button
            type="button"
            onClick={() => refreshHistory()}
            className="text-sm font-medium text-[var(--accent-dark)] underline-offset-4 hover:underline"
          >
            Refresh
          </button>
        </div>
        {historyError && <p className="text-sm text-red-700">{historyError}</p>}
        {history.length === 0 ? (
          <p className="text-sm text-[var(--ink-muted)]">No saved consultations yet.</p>
        ) : (
          <ul className="divide-y divide-[var(--line)] border-y border-[var(--line)]">
            {history.map((item) => (
              <li key={item.id} className="flex flex-wrap items-start justify-between gap-3 py-3">
                <button
                  type="button"
                  onClick={() => openConsultation(item.id)}
                  className="min-w-0 flex-1 text-left"
                >
                  <p className="truncate text-sm font-medium text-[var(--ink)]">
                    {item.symptoms}
                  </p>
                  <p className="mt-1 text-xs text-[var(--ink-muted)]">
                    {formatWhen(item.created_at)}
                    {item.disease_names.length > 0
                      ? ` · ${item.disease_names.slice(0, 3).join(", ")}`
                      : ""}
                    {activeId === item.id ? " · viewing" : ""}
                  </p>
                </button>
                <button
                  type="button"
                  onClick={() => removeConsultation(item.id)}
                  className="text-xs font-medium text-red-700 underline-offset-4 hover:underline"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
