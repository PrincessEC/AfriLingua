import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { Check, Copy } from "lucide-react";
import { GithubMark } from "../../components/Brand";
import { REPO_URL } from "../../components/AppShell";
import { CtaLink } from "../../components/ui";
import { fmtMs } from "../../lib/benchmarks";
import { cn, copyText } from "../../lib/utils";
import { HERO_RUN } from "./heroRun";

const INSTALL = 'pip install "afrilingua[all] @ git+https://github.com/fegig/AfriLingua.git@dev"';

const ease = [0.23, 1, 0.32, 1] as const;

export function Hero() {
  const reduce = useReducedMotion();
  const enter = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 12 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: 0.5, delay, ease },
        };

  return (
    <section className="border-b border-line">
      <div className="mx-auto grid max-w-[1240px] items-center gap-16 px-5 pb-24 pt-16 md:px-8 md:pt-24 lg:grid-cols-[1fr_1fr] lg:pb-32 lg:pt-28">
        <div className="min-w-0">
          <motion.h1
            {...enter(0)}
            className="text-[2.9rem] font-semibold leading-[1.02] tracking-[-0.045em] text-fg sm:text-[3.8rem] lg:text-[4.4rem]"
          >
            Fast, accurate translation for African languages.
          </motion.h1>

          <motion.p {...enter(0.08)} className="mt-7 max-w-[32rem] text-[19px] leading-relaxed text-muted md:text-[21px]">
            Open source. Runs offline on any laptop, no GPU or API bill. Yoruba, Igbo, Hausa and English to start.
          </motion.p>

          <motion.div {...enter(0.16)} className="mt-10 flex flex-wrap items-center gap-3">
            <CtaLink to="/#try">Try it now</CtaLink>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-13 items-center gap-2 rounded-lg px-6 text-[17px] font-medium text-fg ring-1 ring-inset ring-line-strong transition-colors hover:bg-sunken"
            >
              <GithubMark className="size-4" />
              GitHub
            </a>
          </motion.div>

          <motion.div {...enter(0.24)} className="mt-10 max-w-[32rem]">
            <InstallLine />
          </motion.div>
        </div>

        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12, ease }}
          className="min-w-0"
        >
          <HeroConsole />
        </motion.div>
      </div>
    </section>
  );
}

function InstallLine() {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-stretch overflow-hidden rounded-lg border border-line bg-raised">
      <span className="flex items-center border-r border-line px-3 font-mono text-[13px] text-subtle">$</span>
      <code className="no-scrollbar flex-1 overflow-x-auto whitespace-nowrap px-3 py-2.5 font-mono text-[14px] text-fg">{INSTALL}</code>
      <button
        type="button"
        onClick={async () => {
          if (await copyText(INSTALL)) {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1600);
          }
        }}
        className="grid w-10 shrink-0 place-items-center border-l border-line text-muted transition-colors hover:bg-sunken hover:text-fg"
        aria-label={copied ? "Copied" : "Copy install command"}
      >
        {copied ? <Check className="size-4 text-accent-text" strokeWidth={2} /> : <Copy className="size-4" strokeWidth={1.75} />}
      </button>
    </div>
  );
}

type ConsoleTab = "python" | "rest" | "mcp";

const CONSOLE_CODE: Record<ConsoleTab, { file: string; lines: { code: string; dim?: boolean }[] }> = {
  python: {
    file: "translate.py",
    lines: [
      { code: "import afrilingua" },
      { code: "" },
      { code: `text = "${HERO_RUN.source}"` },
      { code: `afrilingua.translate(text, to="yoruba")` },
      { code: `afrilingua.translate(text, to="igbo")` },
      { code: `afrilingua.translate(text, to="hausa")` },
    ],
  },
  rest: {
    file: "terminal",
    lines: [
      { code: "afrilingua serve", dim: true },
      { code: "" },
      { code: "curl localhost:8000/api/v1/translate \\" },
      { code: `  -H "Content-Type: application/json" \\` },
      { code: `  -d '{"text": "${HERO_RUN.source}", "tgt_lang": "yoruba"}'` },
    ],
  },
  mcp: {
    file: "any MCP client → afrilingua mcp",
    lines: [
      { code: "tool: translate_text", dim: true },
      { code: "" },
      { code: "{" },
      { code: `  "text": "${HERO_RUN.source}",` },
      { code: `  "target_language": "yoruba"` },
      { code: "}" },
    ],
  },
};

function HeroConsole() {
  const [tab, setTab] = useState<ConsoleTab>("python");
  const c = CONSOLE_CODE[tab];
  return (
    <figure className="overflow-hidden rounded-2xl border border-ink-line bg-ink text-ink-fg shadow-[0_40px_90px_-45px_rgb(var(--shadow-tint)/0.55)]">
      <div className="flex items-center justify-between gap-3 border-b border-ink-line px-2 py-2">
        <div role="tablist" aria-label="Interface" className="flex gap-0.5">
          {(["python", "rest", "mcp"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              type="button"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={cn(
                "rounded-md px-2.5 py-1 font-mono text-[13px] transition-colors",
                tab === t ? "bg-white/10 text-ink-fg" : "text-ink-muted hover:text-ink-fg",
              )}
            >
              {t === "python" ? "Python" : t === "rest" ? "REST" : "MCP"}
            </button>
          ))}
        </div>
        <span className="truncate pr-2 font-mono text-[12px] text-ink-muted">{c.file}</span>
      </div>

      <pre className="no-scrollbar min-h-[10.5rem] overflow-x-auto px-5 py-4 font-mono text-[14px] leading-[1.7]">
        {c.lines.map((l, i) => (
          <div key={i} className={l.dim ? "text-ink-muted" : undefined}>
            {l.code ? <Highlight code={l.code} /> : " "}
          </div>
        ))}
      </pre>

      <div className="border-t border-ink-line bg-white/[0.02] px-5 py-4">
        <ul className="space-y-2.5">
          {HERO_RUN.outputs.map((o) => (
            <li key={o.lang} className="grid grid-cols-[2.5rem_1fr_auto] items-baseline gap-3">
              <span className="font-mono text-[12px] font-medium uppercase text-[#ff8f45]">{o.code}</span>
              <span lang={o.code} className="min-w-0 text-[17px] leading-snug text-ink-fg">
                {o.text}
              </span>
              <span className="font-mono text-[12px] tabular-nums text-ink-muted">{fmtMs(o.ms)}</span>
            </li>
          ))}
        </ul>
        <figcaption className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-ink-line pt-3 font-mono text-[12px] text-ink-muted">
          <span>{HERO_RUN.caption}</span>
          <span>{HERO_RUN.model}</span>
        </figcaption>
      </div>
    </figure>
  );
}

/** Tiny highlighter for the hero console: strings green, keywords blue. */
function Highlight({ code }: { code: string }) {
  const parts = code.split(/("(?:[^"\\]|\\.)*"|'[^']*'|\bimport\b|\bcurl\b|\btool\b)/g);
  return (
    <>
      {parts.map((p, i) =>
        /^["']/.test(p) ? (
          <span key={i} className="text-[#ffb27a]">
            {p}
          </span>
        ) : /^(import|curl|tool)$/.test(p) ? (
          <span key={i} className="text-[#9ecbff]">
            {p}
          </span>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}
