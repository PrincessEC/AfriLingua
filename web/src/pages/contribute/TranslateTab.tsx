import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CornerDownLeft, RefreshCw, SkipForward } from "lucide-react";
import { api, errorMessage, isOffline, type Domain, type Prompt } from "../../lib/api";
import { useContributor } from "../../lib/contributor";
import { langLabel, useLanguages } from "../../lib/languages";
import { useToast } from "../../lib/toast";
import { Button, Field, Pill, Select, Skeleton } from "../../components/ui";
import { EmptyState, QueryError } from "../../components/states";
import { DOMAINS } from "../../lib/domains";

/** Common dialect suggestions; contributors can type anything. */
export const DIALECTS: Record<string, string[]> = {
  yoruba: ["Standard Yoruba", "Ọ̀yọ́", "Ìjẹ̀bú", "Ẹ̀gbá", "Èkìtì", "Ìjẹ̀ṣà", "Oǹdó", "Ifẹ̀"],
  igbo: ["Standard Igbo", "Owerri", "Onitsha", "Nsukka", "Enugu", "Ngwa", "Mbaise"],
  hausa: ["Standard Hausa (Kano)", "Sokoto", "Katsina", "Zaria", "Bauchi", "Niger Hausa"],
  english: ["Nigerian English", "Ghanaian English", "British English", "American English"],
};

export function DialectInput({ lang, value, onChange, id }: { lang: string; value: string; onChange: (v: string) => void; id: string }) {
  const listId = `${id}-list`;
  return (
    <>
      <input id={id} list={listId} className="field-input" value={value} maxLength={60} onChange={(e) => onChange(e.target.value)} />
      <datalist id={listId}>
        {(DIALECTS[lang] ?? []).map((d) => (
          <option key={d} value={d} />
        ))}
      </datalist>
    </>
  );
}

export function TranslateTab({ lang, onSubmitted }: { lang: string; onSubmitted: () => void }) {
  const { languages } = useLanguages();
  const { profile, update, name } = useContributor();
  const toast = useToast();
  const prompts = useQuery({
    queryKey: ["prompts", "translate", lang],
    queryFn: ({ signal }) => api.prompts({ lang, kind: "translate", n: 10 }, signal),
    staleTime: Infinity,
  });
  const [idx, setIdx] = useState(0);
  const [ownSource, setOwnSource] = useState("");
  const [ownMode, setOwnMode] = useState(false);
  const [translation, setTranslation] = useState("");
  const [domain, setDomain] = useState<Domain>("general");
  const [dialect, setDialect] = useState(profile.dialect);
  const [srcLang, setSrcLang] = useState("english");

  const list = prompts.data ?? [];
  const prompt: Prompt | undefined = list[idx];

  useEffect(() => {
    setIdx(0);
  }, [lang, prompts.dataUpdatedAt]);

  useEffect(() => {
    if (prompt?.domain && DOMAINS.some((d) => d.id === prompt.domain)) setDomain(prompt.domain as Domain);
    setTranslation("");
  }, [prompt?.id, prompt?.domain]);

  const source = ownMode || !prompt ? ownSource : prompt.text;
  const sourceLang = ownMode || !prompt ? srcLang : prompt.lang;

  const submit = useMutation({
    mutationFn: () =>
      api.contribute({
        kind: "translation",
        src_lang: sourceLang,
        tgt_lang: lang,
        source_text: source.trim(),
        target_text: translation.trim(),
        domain,
        dialect: dialect.trim() || undefined,
        region: profile.region || undefined,
        contributor: name,
        license: profile.license,
        consent: true,
      }),
    onSuccess: () => {
      toast.success("Saved. Thank you.", "It joins the review queue now.");
      if (dialect !== profile.dialect) update({ dialect });
      onSubmitted();
      setTranslation("");
      if (ownMode) setOwnSource("");
      else next();
    },
    onError: (e) => toast.error("Not saved", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });

  const next = () => {
    if (idx + 1 < list.length) setIdx(idx + 1);
    else void prompts.refetch().then(() => setIdx(0));
  };

  const progress = useMemo(() => (list.length ? `${idx + 1} of ${list.length}` : ""), [idx, list.length]);
  const canSubmit = source.trim() && translation.trim() && sourceLang !== lang;

  if (prompts.isError && !ownMode) {
    return (
      <div className="flex flex-col gap-4">
        <QueryError error={prompts.error} onRetry={() => void prompts.refetch()} />
        {!isOffline(prompts.error) && (
          <Button variant="secondary" className="self-start" onClick={() => setOwnMode(true)}>
            Bring your own sentence
          </Button>
        )}
      </div>
    );
  }

  return (
    <form
      className="grid gap-6 lg:grid-cols-[1.3fr_1fr]"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) submit.mutate();
      }}
    >
      <div className="shell">
        <div className="core p-6 md:p-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Pill tone="indigo">{langLabel(sourceLang, languages)}</Pill>
              <span className="text-[13px] text-subtle">to</span>
              <Pill tone="accent">{langLabel(lang, languages)}</Pill>
            </div>
            <div className="flex items-center gap-2 text-[13px] text-subtle">
              {!ownMode && progress}
              <button
                type="button"
                className="font-medium text-accent-text underline-offset-4 hover:underline"
                onClick={() => setOwnMode((v) => !v)}
              >
                {ownMode ? "Use suggested sentences" : "Bring your own sentence"}
              </button>
            </div>
          </div>

          <div className="mt-6">
            {ownMode ? (
              <div className="grid gap-4 sm:grid-cols-[10rem_1fr]">
                <Field label="Source language" htmlFor="own-src">
                  <Select id="own-src" value={srcLang} onChange={(e) => setSrcLang(e.target.value)}>
                    {languages
                      .filter((l) => l.name !== lang)
                      .map((l) => (
                        <option key={l.name} value={l.name}>
                          {l.display_name}
                        </option>
                      ))}
                  </Select>
                </Field>
                <Field label="Sentence" htmlFor="own-text">
                  <textarea id="own-text" rows={2} className="field-input text-[17px]" value={ownSource} onChange={(e) => setOwnSource(e.target.value)} />
                </Field>
              </div>
            ) : prompts.isPending ? (
              <div className="space-y-3">
                <Skeleton className="h-8 w-4/5" />
                <Skeleton className="h-8 w-1/2" />
              </div>
            ) : prompt ? (
              <p className="font-display text-[1.7rem] font-medium leading-snug tracking-[-0.02em] md:text-[2rem]">{prompt.text}</p>
            ) : (
              <EmptyState
                title="No sentences waiting"
                description="Every suggested sentence for this language has been translated. Bring your own, or check back later."
                action={
                  <Button variant="secondary" onClick={() => setOwnMode(true)}>
                    Bring your own sentence
                  </Button>
                }
              />
            )}
          </div>

          {(ownMode || prompt) && (
            <div className="mt-8">
              <Field label={`Your ${langLabel(lang, languages)} translation`} htmlFor="contrib-translation">
                <textarea
                  id="contrib-translation"
                  rows={3}
                  value={translation}
                  onChange={(e) => setTranslation(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canSubmit) {
                      e.preventDefault();
                      submit.mutate();
                    }
                  }}
                  className="field-input text-[18px] leading-relaxed"
                  placeholder="Write it as you would say it, with tone marks if you use them"
                />
              </Field>
              <div className="mt-5 flex flex-wrap items-center gap-2">
                <Button type="submit" loading={submit.isPending} disabled={!canSubmit}>
                  Submit
                </Button>
                {!ownMode && (
                  <Button variant="ghost" onClick={next} icon={<SkipForward className="size-4" strokeWidth={1.75} />}>
                    Skip
                  </Button>
                )}
                <span className="ml-auto hidden items-center gap-1 text-[12px] text-subtle sm:inline-flex">
                  <span className="kbd">Ctrl</span>
                  <span className="kbd">
                    <CornerDownLeft className="size-3" strokeWidth={2} />
                  </span>
                  to submit
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      <aside className="flex flex-col gap-5 rounded-[1.75rem] border border-line p-6">
        <Field label="Domain" htmlFor="contrib-domain" help="Helps the model learn the right words for each setting.">
          <Select id="contrib-domain" value={domain} onChange={(e) => setDomain(e.target.value as Domain)}>
            {DOMAINS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Your dialect" htmlFor="contrib-dialect" optional help="Remembered on this device.">
          <DialectInput id="contrib-dialect" lang={lang} value={dialect} onChange={setDialect} />
        </Field>
        <div className="mt-auto rounded-2xl bg-sunken p-4 text-[13.5px] leading-relaxed text-muted">
          <p className="font-semibold text-fg">Tips for good translations</p>
          <ul className="mt-2 list-disc space-y-1 pl-4">
            <li>Translate the meaning, not word for word.</li>
            <li>Use tone marks and sub-dots where you normally would.</li>
            <li>Skip anything you are unsure about. Skipping helps too.</li>
          </ul>
        </div>
        {!ownMode && list.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => void prompts.refetch()} icon={<RefreshCw className="size-4" strokeWidth={1.75} />}>
            New sentences
          </Button>
        )}
      </aside>
    </form>
  );
}
