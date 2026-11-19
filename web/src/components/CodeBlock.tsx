import { useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";
import { copyText, cn } from "../lib/utils";

/** Minimal, dependency-free highlighter: strings, keys, numbers, comments. */
function highlight(code: string, lang: "json" | "python" | "bash"): ReactNode[] {
  const out: ReactNode[] = [];
  const re =
    lang === "python"
      ? /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(from|import|as|with|open|def|return|print|None|True|False)\b|\b(\d+(?:\.\d+)?)\b/gm
      : lang === "bash"
        ? /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\s-{1,2}[a-zA-Z-]+)|\b(curl)\b/gm
        : /("(?:[^"\\]|\\.)*")(\s*:)?|\b(true|false|null)\b|\b(\d+(?:\.\d+)?)\b/gm;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(code))) {
    if (m.index > last) out.push(code.slice(last, m.index));
    const text = m[0];
    let cls = "";
    if (lang === "json") {
      if (m[1] && m[2]) {
        out.push(
          <span key={k++} className="text-[#a9b5f5]">
            {m[1]}
          </span>,
          m[2],
        );
        last = m.index + text.length;
        continue;
      }
      cls = m[1] ? "text-[#f2c27a]" : m[3] ? "text-[#f2a07c]" : "text-[#f2a07c]";
    } else if (lang === "python") {
      cls = m[1] ? "text-[#8f90a6] italic" : m[2] ? "text-[#f2c27a]" : m[3] ? "text-[#a9b5f5]" : "text-[#f2a07c]";
    } else {
      cls = m[1] ? "text-[#8f90a6] italic" : m[2] ? "text-[#f2c27a]" : m[3] ? "text-[#a9b5f5]" : "text-[#f2a07c]";
    }
    out.push(
      <span key={k++} className={cls}>
        {text}
      </span>,
    );
    last = m.index + text.length;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}

export function CodeBlock({
  code,
  lang,
  filename,
  className,
}: {
  code: string;
  lang: "json" | "python" | "bash";
  filename?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <div className={cn("overflow-hidden rounded-[1.25rem] bg-ink text-ink-fg ring-1 ring-white/5", className)}>
      <div className="flex items-center justify-between border-b border-white/10 py-2 pl-4 pr-2">
        <span className="font-mono text-xs text-ink-muted">{filename ?? lang}</span>
        <button
          type="button"
          onClick={async () => {
            if (await copyText(code)) {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1600);
            }
          }}
          className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium text-ink-muted transition-colors hover:bg-white/10 hover:text-ink-fg"
          aria-label={copied ? "Copied" : `Copy ${filename ?? "code"}`}
        >
          {copied ? <Check className="size-3.5 text-[#6fd3a5]" strokeWidth={2} /> : <Copy className="size-3.5" strokeWidth={1.75} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="scrollbar-thin overflow-x-auto p-4 text-[13px] leading-relaxed md:p-5">
        <code className="font-mono">{highlight(code, lang)}</code>
      </pre>
    </div>
  );
}
