import type { TMMatch } from "../lib/api";
import { cn } from "../lib/utils";
import { Button, OriginBadge, Pill } from "./ui";

export function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  return (
    <span className="inline-flex items-center gap-2" title={`Similarity ${pct}%`}>
      <span className="relative h-1.5 w-16 overflow-hidden rounded-full bg-sunken" aria-hidden>
        <span
          className={cn("absolute inset-y-0 left-0 rounded-full", pct >= 95 ? "bg-ok" : pct >= 75 ? "bg-fg/60" : "bg-warn")}
          style={{ width: `${pct}%` }}
        />
      </span>
      <span className="font-mono text-[13px] text-muted">
        <span className="sr-only">Similarity </span>
        {pct}%
      </span>
    </span>
  );
}

export function MatchList({
  matches,
  onUse,
  lang,
  actions,
}: {
  matches: TMMatch[];
  onUse?: (m: TMMatch) => void;
  lang?: string;
  actions?: (m: TMMatch) => React.ReactNode;
}) {
  return (
    <ul className="divide-y divide-line">
      {matches.map((m) => (
        <li key={String(m.id)} className="grid gap-2 py-4 sm:grid-cols-[1fr_1fr_auto] sm:items-start sm:gap-6">
          <p className="text-[16px] text-muted">{m.source}</p>
          <p lang={lang} className="text-[16px] font-medium text-fg">
            {m.target}
          </p>
          <div className="flex flex-wrap items-center gap-2 sm:justify-end">
            <ScoreBar score={m.score} />
            {m.approved ? <Pill tone="ok">Approved</Pill> : <Pill>Unreviewed</Pill>}
            {m.origin && m.origin !== "manual" && <OriginBadge origin={m.origin} className="hidden md:inline-flex" />}
            {onUse && (
              <Button size="sm" variant="ghost" onClick={() => onUse(m)}>
                Use
              </Button>
            )}
            {actions?.(m)}
          </div>
        </li>
      ))}
    </ul>
  );
}
