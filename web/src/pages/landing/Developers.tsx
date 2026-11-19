import { useState } from "react";
import { Bot, Code, Package, Terminal } from "lucide-react";
import { CodeBlock } from "../../components/CodeBlock";
import { Reveal } from "../../components/Reveal";
import { Tabs } from "../../components/ui";

type Tab = "library" | "python" | "curl" | "mcp";

const MCP = `{
  "mcpServers": {
    "afrilingua": {
      "command": "afrilingua",
      "args": ["mcp"]
    }
  }
}`;

const LIB = `import afrilingua

# Text: source language is detected automatically
afrilingua.translate("Where is the clinic?", to="hausa")
afrilingua.translate(["Thank you", "Welcome"], to="igbo", source="english")

# Long documents use beam search for higher quality
afrilingua.translate(report_text, to="yoruba", mode="document")

# Speech
text  = afrilingua.transcribe(audio, lang="yoruba")   # 16 kHz mono float32
voice = afrilingua.speak("Ndewo", lang="igbo")        # waveform + sample rate`;

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
  library: {
    code: LIB,
    lang: "python",
    filename: "app.py",
    note: "Runs in-process. Models load on first use and stay warm; nothing leaves the machine.",
  },
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


export function Developers() {
  const [tab, setTab] = useState<Tab>("library");
  const s = SNIPPETS[tab];
  return (
    <section id="developers" className="mx-auto max-w-[1240px] scroll-mt-20 border-b border-line px-5 py-28 md:px-8 md:py-36" aria-labelledby="dev-title">
      <div className="grid gap-14 lg:grid-cols-[0.9fr_1.1fr] lg:gap-24">
        <Reveal>
          <h2 id="dev-title" className="text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.04em] md:text-[3.4rem]">
            Use it your way.
          </h2>
          <p className="mt-6 max-w-[30rem] text-[19px] leading-relaxed text-muted">
            A Python library, a local API, or tools for AI agents. Apps built for the OpenAI audio API work unchanged.
          </p>
        </Reveal>

        <Reveal delay={0.08} className="min-w-0">
          <Tabs
            label="Integration examples"
            layoutId="dev-tabs"
            value={tab}
            onChange={setTab}
            tabs={[
              { id: "library", label: "Python", icon: <Package className="size-4" strokeWidth={1.6} /> },
              { id: "python", label: "OpenAI SDK", icon: <Code className="size-4" strokeWidth={1.6} /> },
              { id: "curl", label: "REST", icon: <Terminal className="size-4" strokeWidth={1.6} /> },
              { id: "mcp", label: "MCP", icon: <Bot className="size-4" strokeWidth={1.6} /> },
            ]}
          />
          <div role="tabpanel" id={`dev-tabs-panel-${tab}`} aria-labelledby={`dev-tabs-tab-${tab}`} className="mt-4">
            <CodeBlock code={s.code} lang={s.lang} filename={s.filename} />
          </div>
        </Reveal>
      </div>
    </section>
  );
}
