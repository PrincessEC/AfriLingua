import { useState } from "react";
import { Link } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { motion, useReducedMotion } from "motion/react";
import {
  AudioLines,
  Bot,
  Church,
  GraduationCap,
  Landmark,
  Languages,
  Mic,
  Newspaper,
  Plus,
  Stethoscope,
  Tent,
  Volume2,
  ArrowRight,
} from "lucide-react";
import { api } from "../../lib/api";
import { GREETINGS, useLanguages } from "../../lib/languages";
import { cn, formatNumber } from "../../lib/utils";
import { AdireField, Rings } from "../../components/Brand";
import { Reveal } from "../../components/Reveal";
import { CtaLink, Pill } from "../../components/ui";
import { CommandChip } from "../../components/states";

const container = "mx-auto max-w-[1240px] px-5 md:px-8";

// ─── How it works ──────────────────────────────────────────────────────────

const STEPS = [
  { icon: Mic, title: "Speak", body: "The host's laptop captures the microphone and streams 16 kHz audio over the local network." },
  { icon: AudioLines, title: "Recognise", body: "Speech recognition turns each phrase into text, with live partial captions as words arrive." },
  {
    icon: Languages,
    title: "Translate",
    body: "Approved phrases come straight from translation memory. Everything else goes through an int8 translation model.",
  },
  { icon: Volume2, title: "Speak back", body: "Listeners read captions and can hear a synthesised voice in their language, one sentence at a time." },
];

const SEGMENTS = [
  { label: "Recognise", from: 0, to: 0.5, className: "bg-indigo" },
  { label: "Memory lookup", from: 0.5, to: 0.52, className: "bg-ok" },
  { label: "Translate", from: 0.52, to: 0.92, className: "bg-accent" },
  { label: "Deliver", from: 0.92, to: 1.0, className: "bg-ochre" },
  { label: "Speak back", from: 1.0, to: 1.6, className: "bg-fg/30" },
];
const SCALE = 2; // seconds

export function HowItWorks() {
  const reduce = useReducedMotion();
  return (
    <section className={cn(container, "py-24 md:py-32")} aria-labelledby="how-title">
      <Reveal>
        <h2 id="how-title" className="max-w-3xl text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
          From one voice to every language in the room, in about a second.
        </h2>
      </Reveal>

      <ol className="mt-14 grid gap-px overflow-hidden rounded-[1.75rem] bg-line ring-1 ring-line sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map(({ icon: Icon, title, body }, i) => (
          <Reveal as="li" key={title} delay={i * 0.08} className="relative bg-bg p-6 md:p-7">
            <div className="flex items-center justify-between">
              <span className="grid size-11 place-items-center rounded-full bg-sunken text-fg">
                <Icon className="size-5" strokeWidth={1.5} aria-hidden />
              </span>
              {i < STEPS.length - 1 && (
                <ArrowRight className="hidden size-4 text-subtle lg:block" strokeWidth={1.5} aria-hidden />
              )}
            </div>
            <h3 className="mt-6 text-xl font-semibold tracking-tight">{title}</h3>
            <p className="mt-2 text-[15px] leading-relaxed text-muted">{body}</p>
          </Reveal>
        ))}
      </ol>

      <Reveal className="mt-10">
        <div className="shell">
          <div className="core p-6 md:p-8">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
              <p className="font-display text-lg font-semibold tracking-tight">Latency budget per sentence</p>
              <p className="text-[13px] text-subtle">Design targets on a 4-core laptop CPU, not benchmarks.</p>
            </div>

            <div className="relative mt-10 pb-2">
              {/* markers */}
              {[
                { at: 1.0, label: "Caption on phones", value: "~1.0 s" },
                { at: 1.6, label: "Voice starts", value: "~1.6 s" },
              ].map((m) => (
                <div
                  key={m.label}
                  className="absolute -top-8 flex -translate-x-1/2 flex-col items-center"
                  style={{ left: `${(m.at / SCALE) * 100}%` }}
                >
                  <span className="whitespace-nowrap text-[12px] font-medium text-fg">
                    {m.label} <span className="font-mono text-subtle">{m.value}</span>
                  </span>
                  <span className="mt-1 h-9 w-px bg-line-strong" />
                </div>
              ))}
              <div className="relative h-4 overflow-hidden rounded-full bg-sunken">
                {SEGMENTS.map((s, i) => (
                  <motion.span
                    key={s.label}
                    className={cn("absolute inset-y-0 origin-left", s.className)}
                    style={{
                      left: `${(s.from / SCALE) * 100}%`,
                      width: `max(4px, ${((s.to - s.from) / SCALE) * 100}%)`,
                    }}
                    initial={reduce ? false : { scaleX: 0 }}
                    whileInView={{ scaleX: 1 }}
                    viewport={{ once: true, amount: 0.8 }}
                    transition={{ duration: 0.5, delay: 0.15 + i * 0.32, ease: [0.23, 1, 0.32, 1] }}
                  />
                ))}
              </div>
              <div className="mt-2 flex justify-between font-mono text-[11px] text-subtle" aria-hidden>
                {[0, 0.5, 1, 1.5, 2].map((t) => (
                  <span key={t}>{t} s</span>
                ))}
              </div>
            </div>

            <ul className="mt-6 flex flex-wrap gap-x-6 gap-y-2.5">
              {SEGMENTS.map((s) => (
                <li key={s.label} className="flex items-center gap-2 text-[13px] text-muted">
                  <span className={cn("size-2.5 rounded-full", s.className)} aria-hidden />
                  {s.label}
                  <span className="font-mono text-subtle">
                    {s.to - s.from < 0.05 ? "<10 ms" : `~${Math.round((s.to - s.from) * 1000)} ms`}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Reveal>
    </section>
  );
}

// ─── Use cases (bento) ─────────────────────────────────────────────────────

const CASES = [
  { icon: Stethoscope, title: "Healthcare", body: "Explain a diagnosis or consent form in the patient's first language." },
  { icon: Church, title: "Faith communities", body: "One sermon, captioned in every language in the congregation." },
  { icon: Landmark, title: "Courts and government", body: "Town halls, hearings and public notices that everyone can follow." },
  { icon: GraduationCap, title: "Education", body: "Lectures captioned live, and course notes translated as documents." },
  { icon: Newspaper, title: "Media", body: "Draft subtitles and bulletins across languages before air time." },
  { icon: Tent, title: "NGOs and field work", body: "Runs on a laptop in a clinic or village hall with no signal." },
];

export function UseCases() {
  return (
    <section className={cn(container, "py-24 md:py-28")} aria-labelledby="cases-title">
      <Reveal>
        <h2 id="cases-title" className="max-w-2xl text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
          Built for rooms where not everyone speaks English.
        </h2>
      </Reveal>

      <div className="mt-14 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 lg:grid-rows-[auto_auto_auto]">
        {/* Feature tile: conferences */}
        <Reveal className="sm:col-span-2 lg:row-span-2">
          <div className="relative flex h-full min-h-[22rem] flex-col justify-between overflow-hidden rounded-[1.75rem] bg-ink p-7 text-ink-fg md:p-9">
            <AdireField cols={6} rows={5} size={80} className="pointer-events-none absolute inset-0 h-full w-full text-ink-fg opacity-[0.08]" />
            <div className="relative">
              <Pill className="bg-white/10 text-ink-fg">Primary use</Pill>
            </div>
            <div className="relative">
              <h3 className="text-3xl font-semibold tracking-[-0.03em] md:text-4xl">Conferences and events</h3>
              <p className="mt-3 max-w-md text-[15px] leading-relaxed text-ink-muted md:text-base">
                Attendees scan a QR code and follow the speaker on their own phones, captions and audio in their language. No
                interpreter booths, no headsets to hand out.
              </p>
              <Link
                to="/live"
                className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-[#f2a07c] underline-offset-4 hover:underline"
              >
                Open the host console <ArrowRight className="size-4" strokeWidth={1.75} />
              </Link>
            </div>
          </div>
        </Reveal>

        {CASES.map(({ icon: Icon, title, body }, i) => (
          <Reveal key={title} delay={(i % 2) * 0.06}>
            <div
              className={cn(
                "flex h-full flex-col rounded-[1.75rem] p-6 ring-1 ring-line",
                i === 0 ? "bg-accent-soft" : i === 3 ? "bg-ochre-soft" : "bg-raised",
              )}
            >
              <Icon className="size-6 text-fg" strokeWidth={1.4} aria-hidden />
              <h3 className="mt-8 text-lg font-semibold tracking-tight">{title}</h3>
              <p className="mt-1.5 text-[14.5px] leading-relaxed text-muted">{body}</p>
            </div>
          </Reveal>
        ))}

        {/* Developers tile */}
        <Reveal className="sm:col-span-2">
          <div className="flex h-full flex-col justify-between gap-6 rounded-[1.75rem] bg-indigo-soft p-6 ring-1 ring-line md:flex-row md:items-end md:p-7">
            <div>
              <Bot className="size-6 text-indigo-text" strokeWidth={1.4} aria-hidden />
              <h3 className="mt-8 text-lg font-semibold tracking-tight">Developers and AI agents</h3>
              <p className="mt-1.5 max-w-sm text-[14.5px] leading-relaxed text-muted">
                Give Claude, Cursor or your own app African-language speech and translation through MCP or the OpenAI SDK.
              </p>
            </div>
            <a
              href="#developers"
              className="inline-flex shrink-0 items-center gap-2 font-mono text-[13px] text-indigo-text underline-offset-4 hover:underline"
            >
              afrilingua mcp <ArrowRight className="size-4" strokeWidth={1.75} />
            </a>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

// ─── Community ─────────────────────────────────────────────────────────────

const TONE_WORDS = [
  { word: "ọkọ", gloss: "husband" },
  { word: "ọkọ̀", gloss: "vehicle" },
  { word: "ọ̀kọ̀", gloss: "spear" },
];

export function Community() {
  const stats = useQuery({
    queryKey: ["contributions", "stats"],
    queryFn: ({ signal }) => api.contributionStats(signal),
    retry: false,
    staleTime: 60_000,
  });
  const s = stats.data;
  const items = [
    { label: "Contributions", value: s ? formatNumber(s.total) : null },
    { label: "Approved by reviewers", value: s ? formatNumber(s.approved) : null },
    { label: "Contributors", value: s ? formatNumber(s.contributors) : null },
    { label: "Hours of voice", value: s ? (s.hours_audio < 10 ? s.hours_audio.toFixed(1) : formatNumber(s.hours_audio)) : null },
  ];

  return (
    <section className="relative overflow-hidden border-y border-line bg-raised/40 py-24 md:py-32" aria-labelledby="community-title">
      <div className={cn(container, "grid items-center gap-14 lg:grid-cols-[1fr_1.1fr] lg:gap-20")}>
        <Reveal className="relative order-2 lg:order-1">
          <div className="relative mx-auto aspect-square max-w-[30rem]">
            <Rings className="absolute inset-0 size-full text-fg opacity-[0.12]" count={8} />
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 sm:gap-4">
              {TONE_WORDS.map((t, i) => (
                <div key={t.word} className={cn("flex items-baseline gap-4", i === 1 && "translate-x-6", i === 2 && "-translate-x-4")}>
                  <span
                    lang="yo"
                    className={cn(
                      "font-display text-[3.4rem] font-semibold leading-none tracking-[-0.03em] sm:text-[4.6rem]",
                      i === 1 ? "text-accent-text" : "text-fg",
                    )}
                  >
                    {t.word}
                  </span>
                  <span className="font-mono text-[13px] text-subtle">{t.gloss}</span>
                </div>
              ))}
            </div>
          </div>
          <p className="mt-2 text-center text-[13px] text-subtle">Three Yoruba words, told apart only by tone marks.</p>
        </Reveal>

        <div className="order-1 lg:order-2">
          <Reveal>
            <h2 id="community-title" className="text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
              The models are only as good as the people who teach them.
            </h2>
            <p className="mt-5 max-w-[56ch] text-[17px] leading-relaxed text-muted">
              Translate a sentence, record your voice with your dialect, or review someone else's work. Every approved
              contribution goes straight into translation memory and the next model release, under an open licence you choose.
            </p>
          </Reveal>

          <Reveal delay={0.1}>
            <dl className="mt-10 grid grid-cols-2 gap-x-8 gap-y-7 sm:grid-cols-4 lg:grid-cols-2 xl:grid-cols-4">
              {items.map((it) => (
                <div key={it.label}>
                  <dt className="text-[13px] text-subtle">{it.label}</dt>
                  <dd className="mt-1 font-display text-3xl font-semibold tracking-tight text-fg">
                    {it.value ?? <span className="text-subtle">-</span>}
                  </dd>
                </div>
              ))}
            </dl>
            {!s && (
              <p className="mt-4 text-[13px] text-subtle">
                {stats.isPending ? "Loading community numbers." : "Live numbers appear here when the engine is running."}
              </p>
            )}
          </Reveal>

          <Reveal delay={0.16} className="mt-10">
            <CtaLink to="/contribute">Contribute your language</CtaLink>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

// ─── Languages ─────────────────────────────────────────────────────────────

export function LanguagesSection() {
  const { languages, live } = useLanguages();
  return (
    <section className={cn(container, "py-24 md:py-28")} aria-labelledby="langs-title">
      <Reveal>
        <h2 id="langs-title" className="max-w-2xl text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
          Four languages today. Yours next.
        </h2>
        <p className="mt-4 max-w-[60ch] text-[17px] leading-relaxed text-muted">
          English, Yoruba, Igbo and Hausa ship built in. Any language the translation model covers can be added with a few lines
          of configuration, and speech support follows when community recordings arrive.
        </p>
      </Reveal>
      <ul className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {languages.map((l, i) => (
          <Reveal as="li" key={l.name} delay={(i % 4) * 0.06}>
            <div className="shell h-full">
              <div className="core flex h-full flex-col p-6">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-subtle">{l.nllb_code}</span>
                  {!l.builtin && <Pill tone="indigo">Added</Pill>}
                </div>
                <p lang={l.iso639_1 ?? undefined} className="mt-8 font-display text-[1.9rem] font-semibold leading-tight tracking-[-0.03em]">
                  {GREETINGS[l.name] ?? l.native_name ?? l.display_name}
                </p>
                <p className="mt-1 text-[15px] text-muted">
                  {l.display_name}
                  {l.native_name && l.native_name !== l.display_name && <span className="text-subtle"> / {l.native_name}</span>}
                </p>
                <div className="mt-6 flex flex-wrap gap-1.5">
                  <Pill tone="ok">Text</Pill>
                  <Pill tone={l.supports_asr ? "ok" : "neutral"}>{l.supports_asr ? "Speech in" : "No speech in yet"}</Pill>
                  <Pill tone={l.supports_tts ? "ok" : "neutral"}>{l.supports_tts ? "Voice out" : "No voice yet"}</Pill>
                </div>
              </div>
            </div>
          </Reveal>
        ))}
        <Reveal as="li" delay={0.1}>
          <Link
            to="/contribute"
            className="group flex h-full min-h-[14rem] flex-col justify-between rounded-[1.75rem] border border-dashed border-line-strong p-6 transition-colors hover:border-accent"
          >
            <Plus className="size-6 text-subtle transition-colors group-hover:text-accent-text" strokeWidth={1.4} aria-hidden />
            <div>
              <p className="font-display text-xl font-semibold tracking-tight">Bring a language</p>
              <p className="mt-1 text-[14.5px] text-muted">Start with a few hundred sentence pairs and voice clips.</p>
            </div>
          </Link>
        </Reveal>
      </ul>
      {!live && (
        <p className="mt-4 text-[13px] text-subtle">Showing the built-in list. Languages you add appear here when the engine is running.</p>
      )}
    </section>
  );
}

// ─── FAQ ───────────────────────────────────────────────────────────────────

const FAQS = [
  {
    q: "Does it really work without the internet?",
    a: "Yes. Models are downloaded once during setup. After that, speech recognition, translation and voice all run on your machine. Listeners connect to the host laptop over the local Wi-Fi network or a phone hotspot.",
  },
  {
    q: "What hardware do I need?",
    a: "A recent laptop with 4 CPU cores and 8 GB of RAM handles a live room. AfriLingua picks a runtime profile for your machine automatically and uses a GPU when one is available, but never requires it.",
  },
  {
    q: "How accurate is it?",
    a: "Good enough to follow a talk, not yet good enough for legal or clinical records without a human check. Accuracy improves fastest where the community has contributed: approved phrases are served word for word from translation memory.",
  },
  {
    q: "Who owns the data I contribute?",
    a: "You do. Contributions are licensed CC-BY-4.0 by default, or CC0 if you prefer. Your voice is never used to clone a voice unless you opt in separately, and nothing leaves your device until you choose to sync.",
  },
  {
    q: "How many listeners can join a room?",
    a: "Captions are small text messages, so a single laptop comfortably serves a large hall. Audio is heavier: plan for a few dozen audio listeners per room on modest hardware.",
  },
  {
    q: "Can I use it from my own app or AI agent?",
    a: "Yes. Run afrilingua mcp to expose translation, transcription and live rooms as MCP tools, or point any OpenAI SDK at the local /v1 endpoint for transcription and speech.",
  },
];

export function Faq() {
  const [open, setOpen] = useState<number | null>(0);
  return (
    <section className={cn(container, "py-24 md:py-28")} aria-labelledby="faq-title">
      <div className="mx-auto max-w-3xl">
        <Reveal>
          <h2 id="faq-title" className="text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
            Questions, answered.
          </h2>
        </Reveal>
        <div className="mt-10 divide-y divide-line border-y border-line">
          {FAQS.map((f, i) => {
            const isOpen = open === i;
            return (
              <div key={f.q}>
                <h3>
                  <button
                    type="button"
                    id={`faq-btn-${i}`}
                    aria-expanded={isOpen}
                    aria-controls={`faq-panel-${i}`}
                    onClick={() => setOpen(isOpen ? null : i)}
                    className="flex w-full items-center justify-between gap-6 py-5 text-left text-[17px] font-semibold tracking-tight text-fg"
                  >
                    {f.q}
                    <Plus
                      className={cn("size-5 shrink-0 text-subtle transition-transform duration-300 ease-out-strong", isOpen && "rotate-45")}
                      strokeWidth={1.5}
                      aria-hidden
                    />
                  </button>
                </h3>
                <div
                  id={`faq-panel-${i}`}
                  role="region"
                  aria-labelledby={`faq-btn-${i}`}
                  className={cn(
                    "grid transition-[grid-template-rows,opacity] duration-300 ease-out-strong",
                    isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
                  )}
                >
                  <div className="overflow-hidden">
                    <p className="max-w-[65ch] pb-6 text-[15.5px] leading-relaxed text-muted">{f.a}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ─── Final call ────────────────────────────────────────────────────────────

export function FinalCta() {
  return (
    <section className={cn(container, "pb-8")}>
      <Reveal>
        <div className="relative overflow-hidden rounded-[2rem] bg-ink px-7 py-14 text-ink-fg md:px-14 md:py-20">
          <AdireField cols={10} rows={4} size={80} className="pointer-events-none absolute inset-0 h-full w-full text-ink-fg opacity-[0.07]" offset={3} />
          <div className="relative max-w-2xl">
            <h2 className="text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
              Your next event can be heard in every language in the room.
            </h2>
            <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center">
              <CtaLink to="/live">Start interpreting</CtaLink>
              <CommandChip command="afrilingua serve" className="border-white/20 bg-white/5 text-ink-fg [&>span:last-child]:bg-white/10 [&>span:last-child]:text-ink-fg" />
            </div>
          </div>
        </div>
      </Reveal>
    </section>
  );
}
