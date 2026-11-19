/**
 * Speed numbers shown on the site. One source of truth, so the landing page can
 * never quote a figure that was not measured.
 *
 * Source: `afrilingua bench` plus a sentence-level run, 2026-11-19, on an AMD Ryzen 9
 * 5900HS laptop (8 cores, 16 GB RAM), CPU only, int8, 8 threads, ON BATTERY POWER with
 * other work running. Plugged in and idle will be faster; re-measure and update.
 *
 * MT medians per target language (ms): igbo 730, hausa 773, yoruba 1096.
 * MT p95 per target language (ms): igbo 1191, hausa 1149, yoruba 1628.
 */
export interface BenchFigures {
  measured: boolean;
  machine: string;
  date: string;
  /** NLLB-200 600M, CTranslate2 int8, greedy, one new sentence (median across yo/ig/ha). */
  mtP50Ms: number;
  mtP95Ms: number;
  /** Exact translation-memory lookup on a 5,000-entry store. */
  tmExactP50Ms: number;
  /** Fuzzy (n-gram + rerank) translation-memory lookup on a 5,000-entry store. */
  tmFuzzyP50Ms: number;
  /** Repeat of a sentence already translated in this session. */
  cacheHitMs: number;
  /** Process memory with translation loaded for three languages. */
  rssGb: number;
}

export const BENCH: BenchFigures = {
  measured: true,
  machine: "Ryzen 9 laptop CPU on battery, no GPU",
  date: "19 Nov 2026",
  mtP50Ms: 773,
  mtP95Ms: 1191,
  tmExactP50Ms: 0.12,
  tmFuzzyP50Ms: 1.9,
  cacheHitMs: 0.05,
  rssGb: 2.0,
};

export function fmtMs(ms: number): string {
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms)} ms`;
}
