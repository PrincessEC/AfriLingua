import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { CheckCheck, SkipForward, ThumbsDown, ThumbsUp } from "lucide-react";
import { api, errorMessage, isOffline, type Contribution } from "../../lib/api";
import { useContributor } from "../../lib/contributor";
import { langLabel, useLanguages } from "../../lib/languages";
import { useToast } from "../../lib/toast";
import { relativeTime } from "../../lib/utils";
import { Button, Pill, Skeleton } from "../../components/ui";
import { EmptyState, QueryError } from "../../components/states";

function AudioClip({ id }: { id: Contribution["id"] }) {
  // Fetch as a blob so an API key header can be attached when one is configured.
  const q = useQuery({
    queryKey: ["contribution-audio", String(id)],
    queryFn: async () => URL.createObjectURL(await api.contributionAudio(id)),
    staleTime: Infinity,
    gcTime: 60_000,
  });
  useEffect(() => {
    const url = q.data;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [q.data]);
  if (q.isPending) return <Skeleton className="h-12 w-full max-w-md rounded-full" />;
  if (q.isError) return <p className="text-[13px] text-danger">Could not load this recording.</p>;
  return <audio controls autoPlay src={q.data} className="w-full max-w-md" />;
}

export function ReviewTab({ lang, onReviewed }: { lang: string; onReviewed: () => void }) {
  const { languages } = useLanguages();
  const { name } = useContributor();
  const toast = useToast();
  const qc = useQueryClient();
  const reduce = useReducedMotion();
  const [skipped, setSkipped] = useState<Set<string>>(new Set());
  const [comment, setComment] = useState("");
  const [reviewedCount, setReviewedCount] = useState(0);

  const queue = useQuery({
    queryKey: ["review-queue", lang, name],
    queryFn: ({ signal }) => api.reviewQueue({ reviewer: name, lang }, signal),
  });

  const items = (queue.data ?? []).filter((c) => !skipped.has(String(c.id)));
  const current = items[0];

  const vote = useMutation({
    mutationFn: (v: "approve" | "reject") =>
      api.review(current!.id, { vote: v, reviewer: name, comment: comment.trim() || undefined }),
    onSuccess: (_res, v) => {
      setReviewedCount((n) => n + 1);
      onReviewed();
      setComment("");
      // Remove locally, then refetch when the queue runs low.
      qc.setQueryData<Contribution[]>(["review-queue", lang, name], (old) => old?.filter((c) => c.id !== current?.id));
      if (items.length <= 2) void queue.refetch();
      if (v === "approve") toast.success("Approved");
    },
    onError: (e) => toast.error("Vote not recorded", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });

  const skip = () => {
    if (!current) return;
    setSkipped((s) => new Set(s).add(String(current.id)));
    setComment("");
  };

  // Keyboard: A approve, R reject, S skip (ignored while typing).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || t.isContentEditable) return;
      if (e.ctrlKey || e.metaKey || e.altKey || !current || vote.isPending) return;
      const k = e.key.toLowerCase();
      if (k === "a") {
        e.preventDefault();
        vote.mutate("approve");
      } else if (k === "r") {
        e.preventDefault();
        vote.mutate("reject");
      } else if (k === "s") {
        e.preventDefault();
        skip();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (queue.isError) return <QueryError error={queue.error} onRetry={() => void queue.refetch()} />;
  if (queue.isPending) {
    return (
      <div className="shell">
        <div className="core space-y-4 p-8">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-9 w-3/4" />
          <Skeleton className="h-9 w-2/3" />
        </div>
      </div>
    );
  }
  if (!current) {
    return (
      <EmptyState
        icon={<CheckCheck className="size-5" strokeWidth={1.6} />}
        title={reviewedCount ? `All caught up. You reviewed ${reviewedCount}.` : "Nothing to review right now"}
        description={`There are no pending ${langLabel(lang, languages)} contributions from other people. Your own work never shows up here.`}
        action={
          <Button variant="secondary" onClick={() => void queue.refetch()}>
            Check again
          </Button>
        }
      />
    );
  }

  const isAudio = current.has_audio;
  const source = current.source_text ?? (isAudio ? current.text : undefined);
  const target = current.target_text;
  const tLang = current.tgt_lang ?? current.lang ?? lang;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <AnimatePresence mode="wait">
        <motion.div
          key={String(current.id)}
          initial={reduce ? false : { opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          exit={reduce ? undefined : { opacity: 0, x: -16, transition: { duration: 0.15 } }}
          transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
          className="shell"
        >
          <div className="core p-6 md:p-8">
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone="indigo">{isAudio ? "Recording" : current.kind}</Pill>
              {current.domain && <Pill>{current.domain}</Pill>}
              {current.dialect && <Pill tone="accent">{current.dialect}</Pill>}
              <span className="ml-auto text-[12px] text-subtle">
                {current.contributor ? `${current.contributor}, ` : ""}
                {relativeTime(current.created_at)}
              </span>
            </div>

            {isAudio ? (
              <div className="mt-6">
                <p className="text-[13px] text-subtle">Does this recording match the sentence?</p>
                <p className="mt-2 font-display text-[1.6rem] font-medium leading-snug tracking-[-0.02em]">{source}</p>
                <div className="mt-5">
                  <AudioClip id={current.id} />
                </div>
              </div>
            ) : (
              <div className="mt-6 space-y-5">
                <div>
                  <p className="text-[13px] text-subtle">{langLabel(current.src_lang ?? "english", languages)}</p>
                  <p className="mt-1 text-[1.2rem] leading-snug text-muted">{source}</p>
                </div>
                <div>
                  <p className="text-[13px] text-subtle">{langLabel(tLang, languages)}</p>
                  <p
                    lang={languages.find((l) => l.name === tLang)?.iso639_1 ?? undefined}
                    className="mt-1 font-display text-[1.7rem] font-medium leading-snug tracking-[-0.02em]"
                  >
                    {target}
                  </p>
                </div>
                {current.kind === "correction" && current.original_mt && (
                  <div className="rounded-2xl bg-sunken p-4">
                    <p className="text-[12px] text-subtle">Machine translation it replaces</p>
                    <p className="mt-1 text-[15px] text-muted line-through decoration-danger/50">{current.original_mt}</p>
                  </div>
                )}
              </div>
            )}

            <label htmlFor="review-comment" className="sr-only">
              Comment for the contributor
            </label>
            <input
              id="review-comment"
              className="field-input mt-7"
              placeholder="Optional comment, for example a spelling fix"
              value={comment}
              maxLength={240}
              onChange={(e) => setComment(e.target.value)}
            />

            <div className="mt-5 grid grid-cols-[1fr_1fr_auto] gap-2">
              <Button
                variant="secondary"
                size="lg"
                onClick={() => vote.mutate("reject")}
                disabled={vote.isPending}
                icon={<ThumbsDown className="size-4" strokeWidth={1.75} />}
              >
                Reject <span className="kbd ml-1 hidden sm:inline-flex">R</span>
              </Button>
              <Button
                size="lg"
                onClick={() => vote.mutate("approve")}
                loading={vote.isPending && vote.variables === "approve"}
                disabled={vote.isPending}
                icon={<ThumbsUp className="size-4" strokeWidth={1.75} />}
              >
                Approve <span className="kbd ml-1 hidden border-white/30 bg-white/15 text-accent-fg sm:inline-flex">A</span>
              </Button>
              <Button variant="ghost" size="lg" onClick={skip} aria-label="Skip" title="Skip (S)">
                <SkipForward className="size-4" strokeWidth={1.75} />
              </Button>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <aside className="flex flex-col gap-4 rounded-[1.75rem] border border-line p-6 text-[14px] leading-relaxed text-muted">
        <p className="font-semibold text-fg">How reviewing works</p>
        <p>
          Approve if a fluent speaker would say it this way. Small spelling slips are fine to approve with a comment. Reject if the
          meaning is wrong or the recording is unclear.
        </p>
        <p>Two approvals, and more approvals than rejections, add it to translation memory for everyone.</p>
        <div className="mt-auto flex items-center justify-between rounded-2xl bg-sunken p-4">
          <span>In your queue</span>
          <span className="font-display text-2xl font-semibold text-fg">{items.length}</span>
        </div>
      </aside>
    </div>
  );
}
