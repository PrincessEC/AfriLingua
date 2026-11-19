import { CodeBlock } from "../../components/CodeBlock";
import { Reveal } from "../../components/Reveal";
import { cn } from "../../lib/utils";

const container = "mx-auto max-w-[1240px] px-5 md:px-8";

const CLI = `afrilingua corpora import phrases.tsv --format tsv \\
  --from english --to yoruba \\
  --source-name "Clinic phrasebook" --license CC-BY-4.0

afrilingua train splits --lang yoruba
afrilingua train mt`;

export function Train() {
  return (
    <section id="train" className="scroll-mt-20 border-b border-line" aria-labelledby="train-title">
      <div className={cn(container, "grid gap-14 py-28 md:py-36 lg:grid-cols-[0.9fr_1.1fr] lg:gap-24")}>
        <Reveal>
          <h2 id="train-title" className="text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.04em] md:text-[3.4rem]">
            Train it on your own translations.
          </h2>
          <p className="mt-6 max-w-[30rem] text-[19px] leading-relaxed text-muted">
            Upload any translated text you have. AfriLingua aligns it, cleans it and fine-tunes on it.
          </p>
          <p className="mt-6 max-w-[30rem] text-[16px] leading-relaxed text-subtle">
            TMX, CSV, TSV, JSONL, SRT, VTT, USFM or plain text.
          </p>
        </Reveal>
        <Reveal delay={0.06} className="min-w-0 self-center">
          <CodeBlock code={CLI} lang="bash" filename="terminal" />
          <p className="mt-5 text-[15px] leading-relaxed text-subtle">Only upload text you have the rights to share.</p>
        </Reveal>
      </div>
    </section>
  );
}
