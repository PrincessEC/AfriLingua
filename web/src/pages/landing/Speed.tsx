import { motion, useReducedMotion } from "motion/react";
import { Reveal } from "../../components/Reveal";
import { BENCH, fmtMs } from "../../lib/benchmarks";
import { cn } from "../../lib/utils";

const container = "mx-auto max-w-[1240px] px-5 md:px-8";

const PATHS = [
  { name: "Saved translation", ms: BENCH.tmExactP50Ms },
  { name: "Similar sentence", ms: BENCH.tmFuzzyP50Ms },
  { name: "New sentence", ms: BENCH.mtP50Ms },
] as const;

// Log scale so a sub-millisecond hit and a full model call are both visible.
const LOG_MIN = Math.log10(0.05);
const LOG_MAX = Math.log10(2000);
const widthFor = (ms: number) => Math.max(2, ((Math.log10(Math.max(ms, 0.05)) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100);

const ACCURACY = [
  { title: "Measured", body: "Scores per language pair ship with every model release." },
  { title: "Corrected", body: "An approved fix is used from the very next request." },
  { title: "Gated", body: "A new model only ships if it beats the current one." },
];

export function Speed() {
  const reduce = useReducedMotion();
  return (
    <section id="speed" className="scroll-mt-20 border-b border-line" aria-labelledby="speed-title">
      <div className={cn(container, "py-28 md:py-36")}>
        <div className="grid gap-14 lg:grid-cols-[0.9fr_1.1fr] lg:gap-24">
          <Reveal>
            <h2 id="speed-title" className="text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.04em] md:text-[3.4rem]">
              Fast by default.
            </h2>
            <p className="mt-6 max-w-[30rem] text-[19px] leading-relaxed text-muted">
              Anything already translated comes straight from memory. Only new sentences reach the model.
            </p>
          </Reveal>

          <Reveal delay={0.06} className="min-w-0">
            <ol className="space-y-9">
              {PATHS.map((p, i) => (
                <li key={p.name}>
                  <div className="flex items-baseline justify-between gap-4">
                    <p className="text-[18px] font-medium text-fg">{p.name}</p>
                    <p className="font-mono text-[18px] tabular-nums text-fg">{fmtMs(p.ms)}</p>
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-sunken">
                    <motion.div
                      className={cn("h-full origin-left rounded-full", i < 2 ? "bg-accent" : "bg-fg")}
                      style={{ width: `${widthFor(p.ms)}%` }}
                      initial={reduce ? false : { scaleX: 0 }}
                      whileInView={{ scaleX: 1 }}
                      viewport={{ once: true, amount: 0.6 }}
                      transition={{ duration: 0.7, delay: 0.1 + i * 0.12, ease: [0.23, 1, 0.32, 1] }}
                    />
                  </div>
                </li>
              ))}
            </ol>
            <p className="mt-8 text-[15px] text-subtle">
              Median per sentence, {BENCH.machine}. Run <code className="font-mono text-fg">afrilingua bench</code> on yours.
            </p>
          </Reveal>
        </div>

        <div className="mt-28 grid gap-14 border-t border-line pt-20 lg:grid-cols-[0.9fr_1.1fr] lg:gap-24">
          <Reveal>
            <h2 className="text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.04em] md:text-[3.4rem]">
              Accurate, and getting better.
            </h2>
          </Reveal>
          <Reveal delay={0.06}>
            <dl className="grid gap-10 sm:grid-cols-3">
              {ACCURACY.map((a) => (
                <div key={a.title}>
                  <dt className="text-[19px] font-semibold tracking-tight text-fg">{a.title}</dt>
                  <dd className="mt-2 text-[17px] leading-relaxed text-muted">{a.body}</dd>
                </div>
              ))}
            </dl>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
