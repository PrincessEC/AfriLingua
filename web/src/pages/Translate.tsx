import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import { ArrowLeftRight, Check, Copy, CornerDownLeft, PencilLine, Square, Volume2, X } from "lucide-react";
import { api, errorMessage, isOffline, type Domain, type License, type TMMatch, type TranslationResult } from "../lib/api";
import { langLabel, useLanguages } from "../lib/languages";
import { playBlob, stopBlobPlayback } from "../lib/audio";
import { useContributor } from "../lib/contributor";
import { useToast } from "../lib/toast";
import { cn, copyText, formatMs } from "../lib/utils";
import { LanguageSelect } from "../components/LanguageSelect";
import { Button, Field, IconButton, LatencyBadge, OriginBadge, PageHeader, Pill, Select } from "../components/ui";
import { EngineOffline } from "../components/states";
import { MatchList } from "../components/TmMatches";
import { DOMAINS } from "../lib/domains";

const MAX_CHARS = 5000;

export default function Translate() {
  const { languages } = useLanguages();
  const toast = useToast();
  const reduce = useReducedMotion();
  const { profile, update, name } = useContributor();

  const [text, setText] = useState("");
  const [src, setSrc] = useState("auto");
  const [tgt, setTgt] = useState("yoruba");
  const [results, setResults] = useState<TranslationResult[] | null>(null);
  const [override, setOverride] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);
  const [suggestOpen, setSuggestOpen] = useState(false);
  const textRef = useRef<HTMLTextAreaElement>(null);

  const segments = useMemo(
    () =>
      text
        .split(/\n+/)
        .map((s) => s.trim())
        .filter(Boolean),
    [text],
  );

  const translate = useMutation({
    mutationFn: () =>
      api.translate(
        segments.length > 1
          ? { texts: segments, src_lang: src, tgt_lang: tgt, mode: "live", use_tm: true }
          : { text: segments[0] ?? "", src_lang: src, tgt_lang: tgt, mode: "live", use_tm: true },
      ),
    onSuccess: (r) => {
      setResults(r.results);
      setOverride(null);
      setSuggestOpen(false);
    },
  });

  const output = override ?? results?.map((r) => r.text).join("\n") ?? "";
  const first = results?.[0];
  const detected = src === "auto" && first ? first.src_lang : null;
  const totalLatency = results?.reduce((a, r) => Math.max(a, r.latency_ms ?? 0), 0);
  const allMatches = useMemo(() => {
    const seen = new Set<string>();
    const out: TMMatch[] = [];
    for (const r of results ?? [])
      for (const m of r.matches ?? []) {
        const k = String(m.id);
        if (!seen.has(k)) {
          seen.add(k);
          out.push(m);
        }
      }
    return out.sort((a, b) => b.score - a.score).slice(0, 8);
  }, [results]);

  const tgtSpec = languages.find((l) => l.name === tgt);
  const canTranslate = segments.length > 0 && text.length <= MAX_CHARS && src !== tgt && !translate.isPending;

  const run = () => {
    if (canTranslate) translate.mutate();
  };

  useEffect(() => {
    textRef.current?.focus();
    return () => stopBlobPlayback();
  }, []);

  const swap = () => {
    if (src === "auto") {
      if (!detected) return;
      setSrc(tgt);
      setTgt(detected);
    } else {
      setSrc(tgt);
      setTgt(src);
    }
    if (output) setText(output);
    setResults(null);
    setOverride(null);
  };

  const listen = async () => {
    if (speaking) {
      stopBlobPlayback();
      setSpeaking(false);
      return;
    }
    setSpeaking(true);
    try {
      const blob = await api.speech(output, tgt);
      await playBlob(blob);
    } catch (e) {
      toast.error("Could not play the voice", isOffline(e) ? "The engine is offline." : errorMessage(e));
    } finally {
      setSpeaking(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1240px] px-5 py-12 md:px-8 md:py-16">
      <PageHeader
        title="Translate"
        description="Type or paste text. Each line is translated on its own, so lists and scripts keep their shape."
      />

      <div className="shell rounded-2xl p-2">
        <div className="core grid overflow-hidden rounded-xl lg:grid-cols-2">
          {/* Source */}
          <div className="flex flex-col border-line lg:border-r">
            <div className="flex items-end gap-2 border-b border-line p-4 md:px-6">
              <LanguageSelect
                className="flex-1"
                label="Source language"
                value={src}
                onChange={(v) => {
                  setSrc(v);
                  if (v === tgt) setTgt(src === "auto" ? (v === "english" ? "yoruba" : "english") : src);
                }}
                languages={languages}
                allowAuto
              />
              <IconButton
                label="Swap languages"
                onClick={swap}
                disabled={src === "auto" && !detected}
                className="mb-0.5"
              >
                <ArrowLeftRight className="size-4" strokeWidth={1.75} />
              </IconButton>
            </div>
            <div className="relative flex-1">
              <label htmlFor="src-text" className="sr-only">
                Text to translate
              </label>
              <textarea
                id="src-text"
                ref={textRef}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    run();
                  }
                }}
                placeholder="Type something to translate"
                className="block min-h-[16rem] w-full resize-y bg-transparent px-5 py-5 text-[19px] leading-relaxed text-fg placeholder:text-subtle focus:outline-none md:px-7 lg:min-h-[22rem]"
              />
              {text && (
                <IconButton
                  label="Clear text"
                  className="absolute right-3 top-3 size-8"
                  onClick={() => {
                    setText("");
                    setResults(null);
                    textRef.current?.focus();
                  }}
                >
                  <X className="size-4" strokeWidth={1.75} />
                </IconButton>
              )}
            </div>
            <div className="flex items-center justify-between gap-3 border-t border-line px-5 py-3 md:px-6">
              <span
                className={cn("font-mono text-[13px]", text.length > MAX_CHARS ? "text-danger" : "text-subtle")}
                aria-live="polite"
              >
                {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
                {segments.length > 1 && <span className="ml-2">Â· {segments.length} lines</span>}
              </span>
              <div className="flex items-center gap-3">
                <span className="hidden items-center gap-1 text-[13px] text-subtle sm:inline-flex">
                  <span className="kbd">Ctrl</span>
                  <span className="kbd">
                    <CornerDownLeft className="size-3" strokeWidth={2} />
                  </span>
                </span>
                <Button onClick={run} loading={translate.isPending} disabled={!canTranslate}>
                  Translate
                </Button>
              </div>
            </div>
          </div>

          {/* Target */}
          <div className="flex flex-col bg-sunken/40">
            <div className="flex items-end gap-2 border-b border-line p-4 md:px-6">
              <LanguageSelect
                className="flex-1"
                label="Target language"
                value={tgt}
                onChange={(v) => {
                  setTgt(v);
                  setResults(null);
                }}
                languages={languages}
                exclude={src !== "auto" ? [src] : undefined}
              />
            </div>
            <div className="relative min-h-[16rem] flex-1 px-5 py-5 md:px-7 lg:min-h-[22rem]" aria-live="polite" aria-busy={translate.isPending}>
              {translate.isError ? (
                isOffline(translate.error) ? (
                  <EngineOffline compact />
                ) : (
                  <p role="alert" className="rounded-2xl bg-danger-soft p-4 text-[16px]">
                    {errorMessage(translate.error)}
                  </p>
                )
              ) : translate.isPending ? (
                <div className="space-y-3" aria-label="Translating">
                  <div className="h-6 w-4/5 animate-pulse rounded-lg bg-sunken" />
                  <div className="h-6 w-2/3 animate-pulse rounded-lg bg-sunken" />
                  <div className="h-6 w-1/2 animate-pulse rounded-lg bg-sunken" />
                </div>
              ) : output ? (
                <AnimatePresence mode="wait">
                  <motion.div
                    key={output}
                    initial={reduce ? false : { opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
                  >
                    <p lang={tgtSpec?.iso639_1 ?? undefined} className="whitespace-pre-wrap text-[20px] leading-relaxed text-fg">
                      {output}
                    </p>
                    {results && results.length > 1 && (
                      <ul className="mt-5 flex flex-wrap gap-1.5" aria-label="Origin per line">
                        {results.map((r, i) => (
                          <li key={i}>
                            <OriginBadge origin={r.origin} />
                          </li>
                        ))}
                      </ul>
                    )}
                  </motion.div>
                </AnimatePresence>
              ) : (
                <p className="text-[16px] text-subtle">The translation appears here.</p>
              )}
            </div>
            <div className="flex min-h-[3.75rem] flex-wrap items-center justify-between gap-2 border-t border-line px-5 py-3 md:px-6">
              <div className="flex flex-wrap items-center gap-2">
                {first && !override && results?.length === 1 && <OriginBadge origin={first.origin} />}
                {override && <Pill tone="accent">Edited by you</Pill>}
                {results && <LatencyBadge ms={totalLatency} />}
                {detected && <span className="text-[14px] text-subtle">Detected {langLabel(detected, languages)}</span>}
              </div>
              {output && (
                <div className="flex items-center gap-1">
                  <IconButton
                    label={copied ? "Copied" : "Copy translation"}
                    onClick={async () => {
                      if (await copyText(output)) {
                        setCopied(true);
                        window.setTimeout(() => setCopied(false), 1500);
                      }
                    }}
                  >
                    {copied ? <Check className="size-4 text-ok" strokeWidth={2} /> : <Copy className="size-4" strokeWidth={1.75} />}
                  </IconButton>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={listen}
                    disabled={tgtSpec ? !tgtSpec.supports_tts : false}
                    title={tgtSpec && !tgtSpec.supports_tts ? "No voice for this language yet" : undefined}
                    icon={speaking ? <Square className="size-3.5" strokeWidth={2} /> : <Volume2 className="size-4" strokeWidth={1.75} />}
                  >
                    {speaking ? "Stop" : "Listen"}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSuggestOpen((v) => !v)}
                    aria-expanded={suggestOpen}
                    icon={<PencilLine className="size-4" strokeWidth={1.75} />}
                  >
                    Suggest a better translation
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {suggestOpen && first && (
          <motion.div
            initial={reduce ? false : { opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? undefined : { opacity: 0, y: -8, transition: { duration: 0.15 } }}
            transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
          >
            <SuggestForm
              source={segments.join("\n")}
              machine={results?.map((r) => r.text).join("\n") ?? ""}
              srcLang={first.src_lang}
              tgtLang={tgt}
              license={profile.license}
              consent={profile.consent}
              contributor={name}
              onLicense={(l) => update({ license: l })}
              onDone={(t) => {
                setOverride(t);
                setSuggestOpen(false);
                if (!profile.consent) update({ consent: true, consentAt: new Date().toISOString() });
              }}
              onCancel={() => setSuggestOpen(false)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {results && (
        <section className="mt-12" aria-labelledby="tm-title">
          <div className="flex items-baseline justify-between gap-3">
            <h2 id="tm-title" className="text-xl font-semibold tracking-tight">
              Translation memory matches
            </h2>
            <span className="text-[14px] text-subtle">{allMatches.length ? `${allMatches.length} shown` : ""}</span>
          </div>
          {allMatches.length ? (
            <div className="mt-3">
              <MatchList matches={allMatches} lang={tgtSpec?.iso639_1 ?? undefined} onUse={(m) => setOverride(m.target)} />
            </div>
          ) : (
            <p className="mt-3 text-[16px] text-muted">
              No close matches in memory for this text yet. Approved suggestions are added here for everyone.
            </p>
          )}
        </section>
      )}
      {results && totalLatency !== undefined && <p className="sr-only">Translated in {formatMs(totalLatency)}</p>}
    </div>
  );
}

function SuggestForm({
  source,
  machine,
  srcLang,
  tgtLang,
  license,
  consent: priorConsent,
  contributor,
  onLicense,
  onDone,
  onCancel,
}: {
  source: string;
  machine: string;
  srcLang: string;
  tgtLang: string;
  license: License;
  consent: boolean;
  contributor: string;
  onLicense: (l: License) => void;
  onDone: (text: string) => void;
  onCancel: () => void;
}) {
  const toast = useToast();
  const [value, setValue] = useState(machine);
  const [domain, setDomain] = useState<Domain>("general");
  const [consent, setConsent] = useState(priorConsent);
  const [touched, setTouched] = useState(false);

  const submit = useMutation({
    mutationFn: () =>
      api.contribute({
        kind: "correction",
        src_lang: srcLang,
        tgt_lang: tgtLang,
        source_text: source,
        target_text: value.trim(),
        original_mt: machine,
        domain,
        contributor,
        license,
        consent: true,
      }),
    onSuccess: () => {
      toast.success("Thank you. Suggestion sent for review.", "Two approvals add it to translation memory for everyone.");
      onDone(value.trim());
    },
    onError: (e) => toast.error("Suggestion not saved", isOffline(e) ? "The engine is offline. Try again when it is running." : errorMessage(e)),
  });

  const unchanged = value.trim() === machine.trim();
  const consentError = touched && !consent ? "Please confirm you agree to share this contribution." : undefined;

  return (
    <form
      className="mt-6 rounded-2xl border border-line bg-raised p-6 md:p-8"
      onSubmit={(e) => {
        e.preventDefault();
        setTouched(true);
        if (!consent || unchanged || !value.trim()) return;
        submit.mutate();
      }}
    >
      <h2 className="text-lg font-semibold tracking-tight">Suggest a better translation</h2>
      <p className="mt-1 text-[16px] text-muted">Reviewers who speak {langLabel(tgtLang)} will check it before it is used.</p>
      <div className="mt-6 grid gap-5 md:grid-cols-[2fr_1fr]">
        <Field label="Your translation" htmlFor="suggest-text" help={unchanged ? "Edit the text to suggest a change." : undefined}>
          <textarea
            id="suggest-text"
            rows={3}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            className="field-input text-[18px]"
          />
        </Field>
        <div className="flex flex-col gap-5">
          <Field label="Domain" htmlFor="suggest-domain">
            <Select id="suggest-domain" value={domain} onChange={(e) => setDomain(e.target.value as Domain)}>
              {DOMAINS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Licence" htmlFor="suggest-license">
            <Select id="suggest-license" value={license} onChange={(e) => onLicense(e.target.value as License)}>
              <option value="CC-BY-4.0">CC-BY-4.0 (credit me)</option>
              <option value="CC0-1.0">CC0 (public domain)</option>
            </Select>
          </Field>
        </div>
      </div>
      <label className="mt-5 flex items-start gap-3 text-[16px] text-fg">
        <input type="checkbox" className="mt-1 size-4" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
        <span>
          I wrote this translation and agree to share it under {license === "CC0-1.0" ? "CC0" : "CC-BY-4.0"} to improve AfriLingua.
        </span>
      </label>
      {consentError && (
        <p className="mt-2 text-[14px] font-medium text-danger" role="alert">
          {consentError}
        </p>
      )}
      <div className="mt-6 flex flex-wrap gap-2">
        <Button type="submit" loading={submit.isPending} disabled={unchanged || !value.trim()}>
          Send suggestion
        </Button>
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
