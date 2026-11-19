import { useRef, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeftRight, Check, FileUp, Plus, Search, Trash2, X } from "lucide-react";
import { api, errorMessage, isOffline } from "../lib/api";
import { useDebounced } from "../lib/hooks";
import { langLabel, useLanguages } from "../lib/languages";
import { useToast } from "../lib/toast";
import { cn, formatBytes, formatNumber } from "../lib/utils";
import { Button, Field, IconButton, PageHeader, Skeleton } from "../components/ui";
import { LanguageSelect } from "../components/LanguageSelect";
import { EmptyState, QueryError } from "../components/states";
import { MatchList } from "../components/TmMatches";

export default function Memory() {
  const { languages } = useLanguages();
  const toast = useToast();
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [src, setSrc] = useState("english");
  const [tgt, setTgt] = useState("yoruba");
  const [k, setK] = useState(10);
  const debounced = useDebounced(q.trim(), 120);

  const stats = useQuery({
    queryKey: ["tm", "stats"],
    queryFn: ({ signal }) => api.tmStats(signal),
    staleTime: 30_000,
    retry: false,
  });

  const search = useQuery({
    queryKey: ["tm", "search", debounced, src, tgt, k],
    queryFn: ({ signal }) => api.tmSearch({ q: debounced, src_lang: src, tgt_lang: tgt, k }, signal),
    enabled: debounced.length > 0,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  });

  const approve = useMutation({
    mutationFn: (id: string | number) => api.tmApprove(id),
    onSuccess: () => {
      toast.success("Entry approved");
      void qc.invalidateQueries({ queryKey: ["tm"] });
    },
    onError: (e) => toast.error("Not approved", errorMessage(e)),
  });
  const remove = useMutation({
    mutationFn: (id: string | number) => api.tmDelete(id),
    onSuccess: () => {
      toast.success("Entry deleted");
      void qc.invalidateQueries({ queryKey: ["tm"] });
    },
    onError: (e) => toast.error("Not deleted", errorMessage(e)),
  });

  const pairKey = `${src}→${tgt}`;
  const tgtSpec = languages.find((l) => l.name === tgt);

  return (
    <div className="mx-auto max-w-[1240px] px-5 py-12 md:px-8 md:py-16">
      <PageHeader
        title="Translation memory"
        description="Every approved sentence pair, searchable as you type. Exact matches are served instantly during live sessions."
      />

      <dl className="mb-8 flex flex-wrap gap-x-10 gap-y-4">
        {[
          { label: "Entries", value: stats.data?.entries },
          { label: "Approved", value: stats.data?.approved },
          { label: langLabel(src, languages) + " to " + langLabel(tgt, languages), value: stats.data?.by_pair?.[pairKey] ?? (stats.data ? 0 : undefined) },
        ].map((s) => (
          <div key={s.label}>
            <dt className="text-[13px] text-subtle">{s.label}</dt>
            <dd className="mt-0.5 font-display text-2xl font-semibold tabular-nums">
              {stats.isPending ? <Skeleton className="h-7 w-14" /> : stats.isError ? "-" : formatNumber(s.value)}
            </dd>
          </div>
        ))}
      </dl>

      <div className="grid gap-8 lg:grid-cols-[1fr_22rem]">
        <section aria-label="Search">
          <div className="shell">
            <div className="core p-4 md:p-5">
              <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr]">
                <LanguageSelect label="Source" value={src} onChange={setSrc} languages={languages} hideLabel />
                <IconButton
                  label="Swap languages"
                  onClick={() => {
                    setSrc(tgt);
                    setTgt(src);
                  }}
                  className="mx-auto"
                >
                  <ArrowLeftRight className="size-4" strokeWidth={1.75} />
                </IconButton>
                <LanguageSelect label="Target" value={tgt} onChange={setTgt} languages={languages} exclude={[src]} hideLabel />
              </div>
              <div className="relative mt-3">
                <Search className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-subtle" strokeWidth={1.6} aria-hidden />
                <label htmlFor="tm-q" className="sr-only">
                  Search translation memory
                </label>
                <input
                  id="tm-q"
                  type="search"
                  autoFocus
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="Search a phrase, for example: good morning"
                  className="field-input h-14 pl-12 pr-28 text-[17px]"
                />
                <span className="absolute right-4 top-1/2 -translate-y-1/2 font-mono text-[12px] text-subtle" aria-live="polite">
                  {search.data && debounced ? `${search.data.matches.length} in ${search.data.took_ms.toFixed(1)} ms` : ""}
                </span>
              </div>
            </div>
          </div>

          <div className={cn("mt-6 transition-opacity", search.isFetching && search.isPlaceholderData && "opacity-60")}>
            {!debounced ? (
              <EmptyState
                icon={<Search className="size-5" strokeWidth={1.6} />}
                title="Start typing to search"
                description="Fuzzy matching finds close phrases too, scored by similarity."
              />
            ) : search.isError ? (
              <QueryError error={search.error} onRetry={() => void search.refetch()} />
            ) : search.isPending ? (
              <div className="space-y-3">
                {[0, 1, 2].map((i) => (
                  <Skeleton key={i} className="h-14" />
                ))}
              </div>
            ) : search.data && search.data.matches.length === 0 ? (
              <EmptyState
                title="No matches yet"
                description="Add this phrase on the right, or import an existing glossary."
              />
            ) : search.data ? (
              <>
                <MatchList
                  matches={search.data.matches}
                  lang={tgtSpec?.iso639_1 ?? undefined}
                  actions={(m) => (
                    <>
                      {!m.approved && (
                        <IconButton label="Approve entry" className="size-8" onClick={() => approve.mutate(m.id)}>
                          <Check className="size-4" strokeWidth={1.75} />
                        </IconButton>
                      )}
                      <IconButton
                        label="Delete entry"
                        className="size-8 hover:text-danger"
                        onClick={() => {
                          if (window.confirm(`Delete "${m.source}"?`)) remove.mutate(m.id);
                        }}
                      >
                        <Trash2 className="size-4" strokeWidth={1.6} />
                      </IconButton>
                    </>
                  )}
                />
                {search.data.matches.length >= k && (
                  <Button variant="ghost" size="sm" className="mt-3" onClick={() => setK((v) => v + 10)}>
                    Show more
                  </Button>
                )}
              </>
            ) : null}
          </div>
        </section>

        <aside className="flex flex-col gap-6">
          <AddEntry src={src} tgt={tgt} />
          <ImportFile src={src} tgt={tgt} />
        </aside>
      </div>
    </div>
  );
}

function AddEntry({ src, tgt }: { src: string; tgt: string }) {
  const { languages } = useLanguages();
  const toast = useToast();
  const qc = useQueryClient();
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [approved, setApproved] = useState(true);
  const add = useMutation({
    mutationFn: () => api.tmAdd({ source: source.trim(), target: target.trim(), src_lang: src, tgt_lang: tgt, approved, origin: "manual" }),
    onSuccess: () => {
      toast.success("Added to memory");
      setSource("");
      setTarget("");
      void qc.invalidateQueries({ queryKey: ["tm"] });
    },
    onError: (e) => toast.error("Not added", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });
  return (
    <form
      className="rounded-[1.75rem] border border-line p-6"
      onSubmit={(e) => {
        e.preventDefault();
        if (source.trim() && target.trim()) add.mutate();
      }}
    >
      <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
        <Plus className="size-5 text-accent-text" strokeWidth={1.6} /> Add an entry
      </h2>
      <div className="mt-5 flex flex-col gap-4">
        <Field label={langLabel(src, languages)} htmlFor="tm-src">
          <textarea id="tm-src" rows={2} className="field-input" value={source} onChange={(e) => setSource(e.target.value)} />
        </Field>
        <Field label={langLabel(tgt, languages)} htmlFor="tm-tgt">
          <textarea id="tm-tgt" rows={2} className="field-input" value={target} onChange={(e) => setTarget(e.target.value)} />
        </Field>
        <label className="flex items-center gap-2.5 text-[14px]">
          <input type="checkbox" className="size-4" checked={approved} onChange={(e) => setApproved(e.target.checked)} />
          Mark as approved
        </label>
        <Button type="submit" loading={add.isPending} disabled={!source.trim() || !target.trim()}>
          Add to memory
        </Button>
      </div>
    </form>
  );
}

function ImportFile({ src, tgt }: { src: string; tgt: string }) {
  const toast = useToast();
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const imp = useMutation({
    mutationFn: () => api.tmImport(file!, src, tgt),
    onSuccess: (r) => {
      toast.success(`Imported ${formatNumber(r.imported)} entries`);
      setFile(null);
      void qc.invalidateQueries({ queryKey: ["tm"] });
    },
    onError: (e) => toast.error("Import failed", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });
  return (
    <div className="rounded-[1.75rem] border border-line p-6">
      <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
        <FileUp className="size-5 text-accent-text" strokeWidth={1.6} /> Import a file
      </h2>
      <p className="mt-1 text-[14px] text-muted">TSV, CSV, JSONL or TMX. Imported into the language pair selected on the left.</p>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files[0];
          if (f) setFile(f);
        }}
        className={cn(
          "mt-4 flex flex-col items-center rounded-2xl border border-dashed px-4 py-6 text-center transition-colors",
          drag ? "border-accent bg-accent-soft" : "border-line-strong",
        )}
      >
        {file ? (
          <div className="flex w-full items-center gap-3 text-left">
            <div className="min-w-0 flex-1">
              <p className="truncate text-[14px] font-medium">{file.name}</p>
              <p className="text-[12px] text-subtle">{formatBytes(file.size)}</p>
            </div>
            <IconButton label="Remove file" className="size-8" onClick={() => setFile(null)}>
              <X className="size-4" strokeWidth={1.75} />
            </IconButton>
          </div>
        ) : (
          <>
            <p className="text-[14px] text-muted">Drop a file here, or</p>
            <Button variant="secondary" size="sm" className="mt-2" onClick={() => input.current?.click()}>
              Choose file
            </Button>
          </>
        )}
        <input
          ref={input}
          type="file"
          accept=".tsv,.csv,.jsonl,.json,.tmx,.txt"
          className="sr-only"
          tabIndex={-1}
          aria-label="Translation memory file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>
      <Button className="mt-4 w-full" disabled={!file} loading={imp.isPending} onClick={() => imp.mutate()}>
        Import
      </Button>
    </div>
  );
}
