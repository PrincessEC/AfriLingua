import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useInView, useReducedMotion } from "motion/react";
import { Users } from "lucide-react";
import { AdireField } from "../../components/Brand";
import { cn } from "../../lib/utils";

interface DemoLine {
  en: string;
  out: { code: string; lang: string; text: string; origin: string; latency: string }[];
}

// Illustrative sample session. Origins and timings are examples, not measurements.
const LINES: DemoLine[] = [
  {
    en: "Good morning, everyone.",
    out: [
      { code: "YO", lang: "yo", text: "Ẹ káàárọ̀, gbogbo yín.", origin: "tm_exact", latency: "4 ms" },
      { code: "IG", lang: "ig", text: "Ụtụtụ ọma, unu niile.", origin: "tm_exact", latency: "5 ms" },
      { code: "HA", lang: "ha", text: "Barka da safiya, ku duka.", origin: "tm_exact", latency: "4 ms" },
    ],
  },
  {
    en: "Welcome to the conference.",
    out: [
      { code: "YO", lang: "yo", text: "Ẹ káàbọ̀ sí àpérò náà.", origin: "mt", latency: "0.4 s" },
      { code: "IG", lang: "ig", text: "Nnọọ na ogbako a.", origin: "mt", latency: "0.4 s" },
      { code: "HA", lang: "ha", text: "Barka da zuwa taron.", origin: "cache", latency: "1 ms" },
    ],
  },
  {
    en: "Thank you for coming.",
    out: [
      { code: "YO", lang: "yo", text: "A dúpẹ́ pé ẹ wá.", origin: "mt", latency: "0.3 s" },
      { code: "IG", lang: "ig", text: "Daalụ maka ịbịa.", origin: "mt", latency: "0.4 s" },
      { code: "HA", lang: "ha", text: "Na gode da zuwanku.", origin: "mt", latency: "0.3 s" },
    ],
  },
];

const WORD_MS = 210;
const HOLD_MS = 2800;
const STAGGER_MS = 280;

function Waveform({ active }: { active: boolean }) {
  const bars = 28;
  return (
    <div className="flex h-6 items-center gap-[3px]" aria-hidden>
      {Array.from({ length: bars }).map((_, i) => {
        const h = 0.25 + 0.75 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.45));
        return (
          <span
            key={i}
            className={cn(
              "w-[3px] origin-center rounded-full bg-ink-fg/70",
              active ? "animate-wave motion-reduce:animate-none" : "opacity-40",
            )}
            style={{ height: `${Math.round(h * 100)}%`, animationDelay: `${(i % 7) * 90}ms` }}
          />
        );
      })}
    </div>
  );
}

/**
 * Hero visual: a real, simplified rendering of the listener caption view,
 * replaying a sample session. Pauses off-screen; static under reduced motion.
 */
export function LiveCaptionDemo() {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.3 });
  const [lineIdx, setLineIdx] = useState(0);
  const [words, setWords] = useState(reduce ? 99 : 0);
  const [shown, setShown] = useState(reduce ? 3 : 0);

  const line = LINES[lineIdx]!;
  const enWords = line.en.split(" ");
  const isFinal = words >= enWords.length;

  useEffect(() => {
    if (reduce || !inView) return;
    let t: number;
    if (!isFinal) {
      t = window.setTimeout(() => setWords((w) => w + 1), words === 0 ? 500 : WORD_MS);
    } else if (shown < line.out.length) {
      t = window.setTimeout(() => setShown((s) => s + 1), shown === 0 ? 380 : STAGGER_MS);
    } else {
      t = window.setTimeout(() => {
        setLineIdx((i) => (i + 1) % LINES.length);
        setWords(0);
        setShown(0);
      }, HOLD_MS);
    }
    return () => window.clearTimeout(t);
  }, [reduce, inView, isFinal, words, shown, line.out.length]);

  return (
    <div ref={ref} className="relative">
      <div className="relative overflow-hidden rounded-2xl border border-ink-line bg-ink text-ink-fg shadow-[0_30px_80px_-40px_rgb(var(--shadow-tint)/0.6)]">
        <AdireField
          cols={7}
          rows={5}
          size={70}
          className="pointer-events-none absolute inset-0 h-full w-full text-ink-fg opacity-[0.06]"
          offset={2}
        />
        <div className="relative flex items-center justify-between gap-3 border-b border-white/10 px-5 py-3.5 sm:px-6">
          <div className="flex items-center gap-2.5">
            <span className="relative flex size-2.5">
              <span className="absolute inset-0 animate-ping rounded-full bg-[#3fcf80] opacity-60 motion-reduce:hidden" />
              <span className="relative size-2.5 rounded-full bg-[#3fcf80]" />
            </span>
            <span className="text-[15px] font-semibold">Live</span>
            <span className="text-[15px] text-ink-muted">Plenary hall</span>
          </div>
          <div className="flex items-center gap-3 text-[13px] text-ink-muted">
            <span className="hidden rounded-md border border-white/15 px-2 py-0.5 sm:inline">Sample session</span>
            <span className="inline-flex items-center gap-1.5">
              <Users className="size-3.5" strokeWidth={1.75} aria-hidden />
              <span>214 listening</span>
            </span>
          </div>
        </div>

        <div className="relative px-5 pb-6 pt-5 sm:px-6 sm:pb-7">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[12px] tracking-wide text-ink-muted">EN · speaker</span>
            <Waveform active={!reduce && inView && !isFinal} />
          </div>
          <p
            className="mt-3 min-h-[2.6em] font-display text-[1.65rem] font-medium leading-[1.2] tracking-[-0.02em] sm:text-[2rem]"
            aria-live="off"
          >
            {enWords.map((w, i) => (
              <span
                key={`${lineIdx}-${i}`}
                className={cn(
                  "transition-[opacity,color] duration-300",
                  i < words ? (isFinal ? "text-ink-fg" : "text-ink-fg/60") : "opacity-0",
                )}
              >
                {w}{" "}
              </span>
            ))}
            {!isFinal && <span className="ml-0.5 inline-block h-[0.9em] w-[2px] translate-y-[0.1em] animate-caret bg-[#3fcf80]" />}
          </p>

          <div className="mt-5 h-px bg-white/10" />

          <ul className="mt-2 divide-y divide-white/[0.07]">
            {line.out.map((o, i) => (
              <li key={o.code} className="grid min-h-[3.75rem] grid-cols-[2.25rem_1fr] items-center gap-3 py-2.5">
                <span className="font-mono text-[12px] font-medium tracking-wide text-[#4fd88b]">{o.code}</span>
                <AnimatePresence mode="wait" initial={false}>
                  {i < shown ? (
                    <motion.div
                      key={`${lineIdx}-${o.code}`}
                      initial={reduce ? false : { opacity: 0, y: 10, filter: "blur(4px)" }}
                      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                      exit={reduce ? undefined : { opacity: 0, y: -6, transition: { duration: 0.18 } }}
                      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                      className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
                    >
                      <span lang={o.lang} className="text-[1.15rem] font-medium leading-snug sm:text-[1.3rem]">
                        {o.text}
                      </span>
                      <span className="flex shrink-0 items-center gap-2 font-mono text-[12px] text-ink-muted">
                        <span className="rounded-md bg-white/10 px-2 py-0.5">{o.origin}</span>
                        {o.latency}
                      </span>
                    </motion.div>
                  ) : (
                    <motion.span
                      key={`wait-${lineIdx}-${o.code}`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0, transition: { duration: 0.1 } }}
                      className="block h-2 w-24 rounded-full bg-white/10"
                      aria-hidden
                    />
                  )}
                </AnimatePresence>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
