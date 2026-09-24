"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { listSources, SourceListItem, uploadPdf } from "@/lib/api";

const STORAGE_KEY = "symptom_tracker_admin_key";

export default function AdminPanel() {
  const [adminKey, setAdminKey] = useState("");
  const [unlocked, setUnlocked] = useState(false);
  const [sources, setSources] = useState<SourceListItem[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshSources = useCallback(async (key: string) => {
    const data = await listSources(key);
    setSources(data);
  }, []);

  useEffect(() => {
    const saved = sessionStorage.getItem(STORAGE_KEY);
    if (saved) {
      setAdminKey(saved);
      setUnlocked(true);
      refreshSources(saved).catch(() => {
        sessionStorage.removeItem(STORAGE_KEY);
        setUnlocked(false);
      });
    }
  }, [refreshSources]);

  async function unlock(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await refreshSources(adminKey);
      sessionStorage.setItem(STORAGE_KEY, adminKey);
      setUnlocked(true);
      setMessage("Admin key accepted.");
    } catch (err) {
      setUnlocked(false);
      setError(err instanceof Error ? err.message : "Invalid admin key");
    }
  }

  async function onUpload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const result = await uploadPdf(file, adminKey);
      setMessage(result.message || "Upload complete.");
      setFile(null);
      await refreshSources(adminKey);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  if (!unlocked) {
    return (
      <form onSubmit={unlock} className="mx-auto max-w-md space-y-4">
        <label className="block text-sm font-medium text-[var(--ink-muted)]" htmlFor="adminKey">
          Admin key
        </label>
        <input
          id="adminKey"
          type="password"
          value={adminKey}
          onChange={(e) => setAdminKey(e.target.value)}
          className="w-full rounded-md border border-[var(--line)] bg-white/80 px-4 py-2.5 outline-none ring-[var(--accent)] focus:ring-2"
          required
        />
        <button
          type="submit"
          className="rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--accent-dark)]"
        >
          Unlock admin
        </button>
        {error && <p className="text-sm text-red-700">{error}</p>}
      </form>
    );
  }

  return (
    <div className="space-y-8">
      <form onSubmit={onUpload} className="space-y-4">
        <div>
          <label htmlFor="pdf" className="block text-sm font-medium text-[var(--ink-muted)]">
            Upload medical PDF book
          </label>
          <input
            id="pdf"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mt-2 block w-full text-sm text-[var(--ink-muted)] file:mr-4 file:rounded-md file:border-0 file:bg-[var(--accent)] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white"
          />
        </div>
        <button
          type="submit"
          disabled={!file || loading}
          className="rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--accent-dark)] disabled:opacity-60"
        >
          {loading ? "Uploading & embedding…" : "Upload PDF"}
        </button>
      </form>

      {message && (
        <p className="rounded-md border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          {message}
        </p>
      )}
      {error && (
        <p className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </p>
      )}

      <section>
        <h2 className="font-display text-2xl text-[var(--ink)]">Ingested sources</h2>
        {sources.length === 0 ? (
          <p className="mt-2 text-sm text-[var(--ink-muted)]">No sources yet.</p>
        ) : (
          <ul className="mt-4 divide-y divide-[var(--line)] border-y border-[var(--line)]">
            {sources.map((s) => (
              <li key={s.id} className="flex flex-wrap items-baseline justify-between gap-2 py-3 text-sm">
                <div>
                  <p className="font-medium text-[var(--ink)]">{s.name}</p>
                  <p className="text-[var(--ink-muted)]">
                    {s.source_type} · {s.chunk_count} chunks
                  </p>
                </div>
                <time className="text-xs text-[var(--ink-muted)]">{s.created_at}</time>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
