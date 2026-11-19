import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Plus } from "lucide-react";
import { api } from "../../lib/api";
import { GREETINGS, useLanguages } from "../../lib/languages";
import { cn, formatNumber } from "../../lib/utils";
import { Reveal } from "../../components/Reveal";
import { CtaLink, Pill } from "../../components/ui";
import { CommandChip } from "../../components/states";
import { LiveCaptionDemo } from "./LiveCaptionDemo";

const container = "mx-auto max-w-[1240px] px-5 md:px-8";
const h2 = "text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.04em] md:text-[3.4rem]";
const lead = "mt-6 max-w-[30rem] text-[19px] leading-relaxed text-muted";

// ─── Products ──────────────────────────────────────────────────────────────

const PRODUCTS = [
  { title: "Text", body: "Messages, forms and support replies.", to: "/translate" },
  { title: "Documents", body: "DOCX and PDF, with formatting kept.", to: "/documents" },
  { title: "Speech", body: "Speech to text and text to speech.", to: "/#developers" },
];

export function Products() {
  return (
    <section className="border-b border-line" aria-labelledby="products-title">
      <div className={cn(container, "py-28 md:py-36")}>
        <Reveal>
          <h2 id="products-title" className={cn(h2, "max-w-3xl")}>
            One engine for everything you translate.
          </h2>
        </Reveal>

        <div className="mt-16 grid gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-3">
          {PRODUCTS.map((p, i) => (
            <Reveal key={p.title} delay={i * 0.05} className="bg-bg">
              <a href={p.to} className="group flex h-full flex-col p-8 transition-colors hover:bg-sunken">
                <h3 className="text-[22px] font-semibold tracking-tight">{p.title}</h3>
                <p className="mt-2 flex-1 text-[17px] leading-relaxed text-muted">{p.body}</p>
                <ArrowRight
                  className="mt-8 size-5 text-subtle transition-[color,transform] group-hover:translate-x-0.5 group-hover:text-fg"
                  strokeWidth={1.75}
                  aria-hidden
                />
              </a>
            </Reveal>
          ))}
        </div>

        <div className="mt-24 grid items-center gap-14 lg:grid-cols-[0.9fr_1.1fr] lg:gap-24">
          <Reveal>
            <h3 className="text-[2rem] font-semibold leading-[1.1] tracking-[-0.035em] md:text-[2.6rem]">
              Live interpretation for any room.
            </h3>
            <p className={lead}>The speaker talks. Everyone follows on their own phone, in their own language.</p>
            <div className="mt-10">
              <CtaLink to="/live" variant="secondary">
                Start a room
              </CtaLink>
            </div>
          </Reveal>
          <Reveal delay={0.06} className="min-w-0">
            <LiveCaptionDemo />
          </Reveal>
        </div>
      </div>
    </section>
  );
}

// ─── Languages ─────────────────────────────────────────────────────────────

export function LanguagesSection() {
  const { languages } = useLanguages();
  return (
    <section className="border-b border-line" aria-labelledby="langs-title">
      <div className={cn(container, "py-28 md:py-36")}>
        <Reveal>
          <h2 id="langs-title" className={h2}>
            Four languages to start.
          </h2>
          <p className={lead}>Add any of the 200 the model supports with a few lines of configuration.</p>
        </Reveal>
        <ul className="mt-16 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-5">
          {languages.map((l, i) => (
            <Reveal as="li" key={l.name} delay={(i % 4) * 0.04} className="flex flex-col bg-bg p-7">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[13px] text-subtle">{l.nllb_code}</span>
                {!l.builtin && <Pill tone="info">Added</Pill>}
              </div>
              <p lang={l.iso639_1 ?? undefined} className="mt-10 text-[2rem] font-semibold leading-tight tracking-[-0.03em]">
                {GREETINGS[l.name] ?? l.native_name ?? l.display_name}
              </p>
              <p className="mt-1 text-[17px] text-muted">{l.display_name}</p>
            </Reveal>
          ))}
          <Reveal as="li" delay={0.1} className="bg-bg">
            <a
              href="/contribute"
              className="group flex h-full min-h-[13rem] flex-col justify-between p-7 transition-colors hover:bg-sunken"
            >
              <Plus className="size-6 text-subtle transition-colors group-hover:text-fg" strokeWidth={1.5} aria-hidden />
              <p className="text-[19px] font-semibold tracking-tight">Add a language</p>
            </a>
          </Reveal>
        </ul>
      </div>
    </section>
  );
}

// ─── Community ─────────────────────────────────────────────────────────────

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
    { label: "Contributors", value: s ? formatNumber(s.contributors) : null },
  ];

  return (
    <section className="border-b border-line" aria-labelledby="community-title">
      <div className={cn(container, "grid items-end gap-14 py-28 md:py-36 lg:grid-cols-[1.1fr_0.9fr] lg:gap-24")}>
        <Reveal>
          <h2 id="community-title" className={h2}>
            Better with every speaker.
          </h2>
          <p className={lead}>Translate, record or review. Your contributions train the next release.</p>
          <div className="mt-10">
            <CtaLink to="/contribute">Contribute</CtaLink>
          </div>
        </Reveal>

        {s && (
          <Reveal delay={0.08}>
            <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line">
              {items.map((it) => (
                <div key={it.label} className="bg-bg p-7">
                  <dt className="text-[15px] text-subtle">{it.label}</dt>
                  <dd className="mt-2 text-[2.4rem] font-semibold tracking-tight text-fg">{it.value}</dd>
                </div>
              ))}
            </dl>
          </Reveal>
        )}
      </div>
    </section>
  );
}

// ─── FAQ ───────────────────────────────────────────────────────────────────

const FAQS = [
  { q: "Does it work without the internet?", a: "Yes. Models download once. After that everything runs on your machine." },
  { q: "What hardware do I need?", a: "A laptop with 4 cores and 8 GB of RAM. A GPU is used if you have one, but never required." },
  { q: "Can I use it commercially?", a: "The code is MIT. The default models (NLLB-200, MMS) are non-commercial, so check them first." },
  { q: "Who owns the data I contribute?", a: "You do. It is CC-BY-4.0 by default, and stays on your device until you share it." },
];

export function Faq() {
  return (
    <section className="border-b border-line" aria-labelledby="faq-title">
      <div className={cn(container, "grid gap-14 py-28 md:py-36 lg:grid-cols-[0.9fr_1.1fr] lg:gap-24")}>
        <Reveal>
          <h2 id="faq-title" className={h2}>
            Questions
          </h2>
        </Reveal>
        <dl className="divide-y divide-line border-y border-line">
          {FAQS.map((f) => (
            <div key={f.q} className="py-7">
              <dt className="text-[19px] font-semibold tracking-tight text-fg">{f.q}</dt>
              <dd className="mt-2 text-[17px] leading-relaxed text-muted">{f.a}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

// ─── Final call ────────────────────────────────────────────────────────────

export function FinalCta() {
  return (
    <section className={cn(container, "py-32 md:py-40")}>
      <Reveal className="mx-auto max-w-3xl text-center">
        <h2 className="text-[2.6rem] font-semibold leading-[1.05] tracking-[-0.045em] md:text-[3.8rem]">
          Start translating today.
        </h2>
        <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <CtaLink to="/#try">Try it now</CtaLink>
          <CommandChip command="afrilingua serve" />
        </div>
      </Reveal>
    </section>
  );
}
