import { useState } from "react";
import { Bot, Code, Terminal } from "lucide-react";
import { CodeBlock } from "../../components/CodeBlock";
import { Reveal } from "../../components/Reveal";
import { Tabs } from "../../components/ui";

type Tab = "mcp" | "python" | "curl";

const MCP = `{
  "mcpServers": {
    "afrilingua": {
      "command": "afrilingua",
      "args": ["mcp"]
    }
  }
}`;

const PY = `from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

with open("sermon.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="afrilingua",
        file=audio,
        language="yo",
    )

print(transcript.text)`;

const CURL = `curl http://127.0.0.1:8000/api/v1/translate \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Good morning, everyone.", "src_lang": "english", "tgt_lang": "igbo"}'`;

const SNIPPETS: Record<Tab, { code: string; lang: "json" | "python" | "bash"; filename: string; note: string }> = {
  mcp: {
    code: MCP,
    lang: "json",
    filename: "mcp config (Claude Desktop, Cursor, Gemini CLI)",
    note: "Agents get translate_text, transcribe_audio, synthesize_speech, search_translation_memory and create_live_room as tools.",
  },
  python: {
    code: PY,
    lang: "python",
    filename: "transcribe.py",
    note: "Existing Whisper code keeps working. Change base_url; the model name is ignored.",
  },
  curl: {
    code: CURL,
    lang: "bash",
    filename: "terminal",
    note: "Returns the translation with its origin (tm_exact, cache or mt) and latency.",
  },
};

const TOOLS = [
  "translate_text",
  "translate_batch",
  "detect_language",
  "transcribe_audio",
  "synthesize_speech",
  "translate_speech",
  "search_translation_memory",
  "create_live_room",
];

export function Developers() {
  const [tab, setTab] = useState<Tab>("mcp");
  const s = SNIPPETS[tab];
  return (
    <section id="developers" className="mx-auto max-w-[1240px] scroll-mt-24 px-5 py-24 md:px-8 md:py-32" aria-labelledby="dev-title">
      <div className="grid gap-12 lg:grid-cols-[0.85fr_1.15fr] lg:gap-16">
        <Reveal>
          <span className="inline-flex rounded-full bg-indigo-soft px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-text">
            For developers and AI agents
          </span>
          <h2 id="dev-title" className="mt-5 text-[2.1rem] font-semibold leading-[1.05] tracking-[-0.035em] md:text-5xl">
            One local server. Three ways in.
          </h2>
          <p className="mt-5 max-w-[50ch] text-[17px] leading-relaxed text-muted">
            The same engine that runs live rooms speaks MCP for agents, the OpenAI audio API for existing apps, and plain REST for
            everything else.
          </p>
          <ul className="mt-8 flex flex-wrap gap-1.5" aria-label="MCP tools">
            {TOOLS.map((t) => (
              <li key={t} className="rounded-full bg-sunken px-2.5 py-1 font-mono text-[12px] text-muted">
                {t}
              </li>
            ))}
          </ul>
        </Reveal>

        <Reveal delay={0.08} className="min-w-0">
          <Tabs
            label="Integration examples"
            layoutId="dev-tabs"
            value={tab}
            onChange={setTab}
            tabs={[
              { id: "mcp", label: "MCP", icon: <Bot className="size-4" strokeWidth={1.6} /> },
              { id: "python", label: "OpenAI SDK", icon: <Code className="size-4" strokeWidth={1.6} /> },
              { id: "curl", label: "curl", icon: <Terminal className="size-4" strokeWidth={1.6} /> },
            ]}
          />
          <div role="tabpanel" id={`dev-tabs-panel-${tab}`} aria-labelledby={`dev-tabs-tab-${tab}`} className="mt-4">
            <CodeBlock code={s.code} lang={s.lang} filename={s.filename} />
            <p className="mt-3 text-[14px] leading-relaxed text-subtle">{s.note}</p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
