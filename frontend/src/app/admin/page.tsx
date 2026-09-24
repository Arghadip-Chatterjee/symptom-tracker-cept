import Link from "next/link";
import AdminPanel from "@/components/AdminPanel";

export default function AdminPage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-40" />
      <div className="relative mx-auto max-w-3xl px-6 pb-16 pt-10 sm:pt-14">
        <header className="mb-10 flex items-end justify-between gap-4">
          <div>
            <p className="font-display text-4xl tracking-tight text-[var(--ink)]">Admin</p>
            <p className="mt-2 text-sm text-[var(--ink-muted)]">
              Upload PDF medical books to ground predictions in your knowledge base.
            </p>
          </div>
          <Link
            href="/"
            className="text-sm font-medium text-[var(--accent-dark)] underline-offset-4 hover:underline"
          >
            Back
          </Link>
        </header>
        <AdminPanel />
      </div>
    </main>
  );
}
