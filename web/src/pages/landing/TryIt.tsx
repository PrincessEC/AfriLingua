import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowLeftRight, CornerDownLeft, Languages } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { api, isOffline, errorMessage, type TranslationResult } from "../../lib/api";
import { useLanguages, langLabel } from "../../lib/languages";
import { cn } from "../../lib/utils";
import { Reveal } from "../../components/Reveal";
import { LanguageSelect } from "../../components/LanguageSelect";
import { Button, IconButton, LatencyBadge, OriginBadge } from "../../components/ui";
import { EngineOffline } from "../../components/states";

const SAMPLES = ["Good morning, everyone.", "Where is the nearest clinic?", "The meeting starts at ten o'clock."];

export function TryIt() {
  const { languages } = useLanguages();
  const reduce = useReducedMotion();
  const [text, setText] = useState(SAMPLES[0]!);
  const [src, setSrc] = useState("english");
  const [tgt, setTgt] = useState("yoruba");

  const m = useMutation({
    mutationFn: () => api.translate({ text, src_lang: src, tgt_lang: tgt, mode: "live", use_tm: true }),
  });
  const result: TranslationResult | undefined = m.data?.results[0];

  const submit = () => {
    if (text.trim()) m.mutate();
  };

  return (
    <section id="try" className="mx-auto max-w-[1240px] scroll-mt-24 px-5 py-8 md:px-8 md:py-12" aria-labelledby="try-title">
      <Reveal>
        <div className="shell rounded-[2rem] p-2">
          <div className="core rounded-[calc(2rem-0.5rem)]">
            <div className="flex flex-col gap-2 border-b border-line px-6 py-5 md:flex-row md:items-center md:justify-between md:px-8">
              <h2 id="try-title" className="flex items-center gap-3 text-xl font-semibold tracking-tight md:text-2xl">
                <Languages className="size-5 text-accent-text" strokeWidth={1.5} aria-hidden />
                Try it on your engine
              </h2>
              <p className="text-[13px] text-subtle">Runs against your local server. Nothing leaves this machine.</p>
            </div>

            <div className="grid lg:grid-cols-2">
              {/* Input */}
              <form
                className="flex flex-col gap-5 border-line p-6 md:p-8 lg:border-r"
                onSubmit={(e) => {
                  e.preventDefault();
                  submit();
                }}
              >
                <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-2">
                  <LanguageSelect label="From" value={src} onChange={setSrc} languages={languages} allowAuto />
                  <IconButton
                    label="Swap languages"
                    className="mb-0.5"
                    disabled={src === "auto"}
                    onClick={() => {
                      setSrc(tgt);
                      setTgt(src);
                    }}
                  >
                    <ArrowLeftRight className="size-4" strokeWidth={1.75} />
                  </IconButton>
                  <LanguageSelect label="To" value={tgt} onChange={setTgt} languages={languages} />
                </div>
                <div className="flex flex-col gap-2">
                  <label htmlFor="try-text" className="field-label">
                    Text to translate
                  </label>
                  <textarea
                    id="try-text"
                    rows={4}
                    maxLength={500}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        submit();
                      }
                    }}
                    className="field-input resize-none text-[17px] leading-relaxed"
                  />
                  <div className="flex flex-wrap gap-1.5">
                    {SAMPLES.map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => setText(s)}
                        className={cn(
                          "rounded-full px-3 py-1 text-[13px] ring-1 transition-colors",
                          text === s ? "bg-sunken text-fg ring-line-strong" : "text-muted ring-line hover:text-fg",
                        )}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="hidden items-center gap-1.5 text-[13px] text-subtle sm:inline-flex">
                    <span className="kbd">Ctrl</span>
                    <span className="kbd">
                      <CornerDownLeft className="size-3" strokeWidth={2} />
                    </span>
                    to translate
                  </span>
                  <Button type="submit" loading={m.isPending} disabled={!text.trim() || src === tgt}>
                    Translate
                  </Button>
                </div>
              </form>

              {/* Output */}
              <div className="flex min-h-[18rem] flex-col p-6 md:p-8" aria-live="polite">
                <p className="field-label">{langLabel(tgt, languages)}</p>
                <div className="mt-3 flex-1">
                  {m.isError ? (
                    isOffline(m.error) ? (
                      <EngineOffline compact description="The demo needs the local engine to translate." />
                    ) : (
                      <p className="rounded-2xl bg-danger-soft p-4 text-[15px] text-fg" role="alert">
                        {errorMessage(m.error)}
                      </p>
                    )
                  ) : m.isPending ? (
                    <div className="space-y-3" aria-label="Translating">
                      <div className="h-6 w-4/5 animate-pulse rounded-lg bg-sunken" />
                      <div className="h-6 w-3/5 animate-pulse rounded-lg bg-sunken" />
                    </div>
                  ) : result ? (
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={result.text}
                        initial={reduce ? false : { opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
                      >
                        <p className="font-display text-[1.8rem] font-medium leading-snug tracking-[-0.02em] text-fg">{result.text}</p>
                        <div className="mt-4 flex flex-wrap items-center gap-2">
                          <OriginBadge origin={result.origin} />
                          <LatencyBadge ms={result.latency_ms} />
                          {result.detected && src === "auto" && (
                            <span className="text-[13px] text-subtle">Detected {langLabel(result.src_lang, languages)}</span>
                          )}
                        </div>
                      </motion.div>
                    </AnimatePresence>
                  ) : (
                    <p className="text-[15px] leading-relaxed text-subtle">
                      Your translation appears here, with where it came from: an approved memory match, the cache, or the
                      translation model.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </Reveal>
    </section>
  );
}
