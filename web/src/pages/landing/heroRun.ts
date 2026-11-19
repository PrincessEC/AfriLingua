/**
 * A real run shown in the hero console: actual model output and measured latency
 * (first translation of each sentence, cache cleared), captured 2026-11-19 on a
 * Ryzen 9 laptop CPU on battery, int8, no GPU. Update together with lib/benchmarks.ts.
 */
export interface HeroRun {
  source: string;
  outputs: { lang: string; code: string; text: string; ms: number }[];
  caption: string;
  model: string;
}

export const HERO_RUN: HeroRun = {
  source: "Where is the nearest clinic?",
  outputs: [
    { lang: "yoruba", code: "yo", text: "Ibo ni ilé ìwòsàn tó sún mọ́ ẹ jù lọ wà?", ms: 1093 },
    { lang: "igbo", code: "ig", text: "Olee ebe ụlọ ọgwụ kacha nso dị?", ms: 493 },
    { lang: "hausa", code: "ha", text: "A ina ne asibitin da ya fi kusa?", ms: 765 },
  ],
  caption: "Real output · laptop CPU on battery · no GPU",
  model: "nllb-200-distilled-600M · int8",
};
