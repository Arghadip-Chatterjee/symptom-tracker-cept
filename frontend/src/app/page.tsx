import Link from "next/link";
import SymptomAnalyzer from "@/components/SymptomAnalyzer";

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-40" />
      <div className="pointer-events-none absolute -left-24 top-10 h-72 w-72 rounded-full bg-[var(--wash)] blur-3xl" />
      <div className="pointer-events-none absolute -right-16 bottom-10 h-64 w-64 rounded-full bg-teal-100/70 blur-3xl" />

      <div className="relative mx-auto max-w-3xl px-6 pb-16 pt-10 sm:pt-14">
        <header className="mb-10 flex items-end justify-between gap-4 animate-fade-up">
          <div>
            <p className="font-display text-4xl tracking-tight text-[var(--ink)] sm:text-5xl">
              Symptom Tracker
            </p>
            <p className="mt-3 max-w-xl text-base leading-relaxed text-[var(--ink-muted)]">
              Describe how you feel. The assistant retrieves your medical PDFs and datasets, then
              suggests possible conditions and medicines with source citations.
            </p>
          </div>
          <Link
            href="/admin"
            className="shrink-0 text-sm font-medium text-[var(--accent-dark)] underline-offset-4 hover:underline"
          >
            Admin
          </Link>
        </header>

        <div className="animate-fade-up-delay">
          <SymptomAnalyzer />
        </div>
      </div>
    </main>
  );
}
