import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, Trophy } from "lucide-react";
import { api } from "../../lib/api";
import { useContributor } from "../../lib/contributor";
import { langLabel, useLanguages } from "../../lib/languages";
import { formatNumber } from "../../lib/utils";
import { Skeleton } from "../../components/ui";
import { QueryError } from "../../components/states";

export function StatsTab({ sessionCount }: { sessionCount: number }) {
  const { languages } = useLanguages();
  const { name } = useContributor();
  const stats = useQuery({
    queryKey: ["contributions", "stats"],
    queryFn: ({ signal }) => api.contributionStats(signal),
    staleTime: 30_000,
  });
  // Leaderboard: use the server's list when present, otherwise derive from recent approved items.
  const approved = useQuery({
    queryKey: ["contributions", "approved-recent"],
    queryFn: ({ signal }) => api.contributions({ status: "approved", limit: 200 }, signal),
    enabled: stats.isSuccess && !stats.data?.leaderboard,
    staleTime: 60_000,
  });

  const board = useMemo(() => {
    if (stats.data?.leaderboard) return stats.data.leaderboard.slice(0, 10);
    const counts = new Map<string, number>();
    for (const c of approved.data?.items ?? []) {
      const who = c.contributor?.trim();
      if (!who) continue;
      counts.set(who, (counts.get(who) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([contributor, count]) => ({ contributor, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [stats.data, approved.data]);

  if (stats.isError) return <QueryError error={stats.error} onRetry={() => void stats.refetch()} />;
  const s = stats.data;
  const byLang = s ? Object.entries(s.by_lang ?? {}) : [];
  const maxLang = Math.max(1, ...byLang.map(([, v]) => v.text + v.audio));

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <div className="flex flex-col gap-6">
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl bg-line ring-1 ring-line md:grid-cols-4">
          {[
            { label: "Contributions", value: s?.total },
            { label: "Approved", value: s?.approved },
            { label: "Awaiting review", value: s?.pending },
            { label: "Contributors", value: s?.contributors },
          ].map((it) => (
            <div key={it.label} className="bg-raised p-5">
              <dt className="text-[14px] text-subtle">{it.label}</dt>
              <dd className="mt-1 font-display text-3xl font-semibold tracking-tight">
                {s ? formatNumber(it.value) : <Skeleton className="mt-1 h-8 w-16" />}
              </dd>
            </div>
          ))}
        </dl>

        <section className="rounded-2xl border border-line p-6" aria-labelledby="by-lang">
          <div className="flex items-baseline justify-between">
            <h3 id="by-lang" className="text-lg font-semibold tracking-tight">
              By language
            </h3>
            {s && <span className="text-[14px] text-subtle">{s.hours_audio.toFixed(1)} hours of voice</span>}
          </div>
          {s && byLang.length === 0 && <p className="mt-4 text-[16px] text-muted">No contributions yet. Yours could be the first.</p>}
          <ul className="mt-5 space-y-5">
            {byLang.map(([lang, v]) => (
              <li key={lang}>
                <div className="flex items-baseline justify-between text-[15px]">
                  <span className="font-medium">{langLabel(lang, languages)}</span>
                  <span className="font-mono text-[13px] text-subtle">
                    {formatNumber(v.text)} text / {formatNumber(v.audio)} audio / {formatNumber(v.approved)} approved
                  </span>
                </div>
                <div className="mt-2 flex h-2 overflow-hidden rounded-full" aria-hidden>
                  <span className="bg-fg/70" style={{ width: `${(v.text / maxLang) * 100}%` }} />
                  <span className="bg-accent" style={{ width: `${(v.audio / maxLang) * 100}%` }} />
                </div>
              </li>
            ))}
          </ul>
          {byLang.length > 0 && (
            <div className="mt-5 flex gap-4 text-[13px] text-subtle">
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-fg/70" /> Text
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-accent" /> Audio
              </span>
            </div>
          )}
        </section>
      </div>

      <aside className="flex flex-col gap-6">
        <section className="rounded-xl border border-line bg-raised p-6 text-fg" aria-labelledby="board-title">
          <h3 id="board-title" className="flex items-center gap-2 text-lg font-semibold tracking-tight">
            <Trophy className="size-5 text-accent-text" strokeWidth={1.6} /> Leaderboard
          </h3>
          {board.length === 0 ? (
            <p className="mt-4 text-[15px] text-subtle">Approved contributors appear here.</p>
          ) : (
            <ol className="mt-4 divide-y divide-line">
              {board.map((b, i) => (
                <li key={b.contributor} className="flex items-center gap-3 py-2.5">
                  <span className="w-6 font-mono text-[13px] text-subtle">{i + 1}</span>
                  <span className={`flex-1 truncate text-[16px] ${b.contributor === name ? "font-semibold text-accent-text" : ""}`}>
                    {b.contributor}
                    {b.contributor === name && " (you)"}
                  </span>
                  <span className="font-mono text-[14px] text-subtle">{b.count}</span>
                </li>
              ))}
            </ol>
          )}
        </section>
        <section className="rounded-2xl border border-line p-6">
          <p className="text-[14px] text-subtle">This session, on this device</p>
          <p className="mt-1 font-display text-3xl font-semibold">{sessionCount}</p>
          <p className="mt-1 text-[15px] text-muted">contributions and reviews</p>
          <a
            href={api.contributionsExportUrl({ format: "jsonl" })}
            className="mt-5 inline-flex items-center gap-2 text-[15px] font-medium text-accent-text underline-offset-4 hover:underline"
          >
            <Download className="size-4" strokeWidth={1.75} /> Download approved data (JSONL)
          </a>
        </section>
      </aside>
    </div>
  );
}
