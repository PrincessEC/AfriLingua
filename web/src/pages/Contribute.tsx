import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { BarChart3, Languages, Mic, ShieldCheck, UserRound, ListChecks } from "lucide-react";
import { useContributor } from "../lib/contributor";
import { useLanguages } from "../lib/languages";
import { useLocalStorage } from "../lib/hooks";
import { Button, Tabs } from "../components/ui";
import { EmptyState } from "../components/states";
import { ConsentModal } from "./contribute/ConsentModal";
import { TranslateTab } from "./contribute/TranslateTab";
import { RecordTab } from "./contribute/RecordTab";
import { ReviewTab } from "./contribute/ReviewTab";
import { StatsTab } from "./contribute/StatsTab";

type Tab = "translate" | "record" | "review" | "stats";
const TABS: Tab[] = ["translate", "record", "review", "stats"];

export default function Contribute() {
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab");
  const tab: Tab = TABS.includes(raw as Tab) ? (raw as Tab) : "translate";
  const setTab = (t: Tab) => {
    const next = new URLSearchParams(params);
    next.set("tab", t);
    setParams(next, { replace: true });
  };
  const { languages } = useLanguages();
  const { profile, name } = useContributor();
  const [lang, setLang] = useLocalStorage("afrilingua.contribLang", "yoruba");
  const [consentOpen, setConsentOpen] = useState(false);
  const [sessionCount, setSessionCount] = useState(0);
  const bump = () => setSessionCount((n) => n + 1);

  // First visit to a contributing tab: ask for consent once per visit.
  const asked = useRef(false);
  useEffect(() => {
    if (!profile.consent && tab !== "stats" && !asked.current) {
      asked.current = true;
      setConsentOpen(true);
    }
  }, [profile.consent, tab]);

  const contribLangs = languages.filter((l) => l.name !== "english");
  const effectiveLang = languages.some((l) => l.name === lang) ? lang : (contribLangs[0]?.name ?? "yoruba");

  return (
    <div className="mx-auto max-w-[1240px] px-5 py-12 md:px-8 md:py-16">
      <header className="mb-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-semibold tracking-[-0.03em] md:text-[2.6rem] md:leading-[1.05]">Teach the models your language</h1>
          <p className="mt-3 text-[17px] leading-relaxed text-muted">
            A few minutes of translating, recording or reviewing makes interpretation better for everyone who speaks it.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <p id="contrib-lang" className="field-label mb-2">
              I speak
            </p>
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-labelledby="contrib-lang">
              {languages.map((l) => (
                <button
                  key={l.name}
                  type="button"
                  role="radio"
                  aria-checked={effectiveLang === l.name}
                  onClick={() => setLang(l.name)}
                  className={`rounded-full px-4 py-2 text-[14.5px] font-medium ring-1 transition-colors ${
                    effectiveLang === l.name ? "bg-fg text-bg ring-fg" : "text-muted ring-line-strong hover:text-fg"
                  }`}
                >
                  {l.display_name}
                </button>
              ))}
            </div>
          </div>
          <Button
            variant="secondary"
            size="md"
            onClick={() => setConsentOpen(true)}
            icon={profile.consent ? <UserRound className="size-4" strokeWidth={1.75} /> : <ShieldCheck className="size-4" strokeWidth={1.75} />}
            title="Contributor settings"
          >
            <span className="max-w-[9rem] truncate">{profile.consent ? name : "Set up"}</span>
          </Button>
        </div>
      </header>

      <Tabs
        label="Contribution type"
        layoutId="contrib-tabs"
        value={tab}
        onChange={setTab}
        tabs={[
          { id: "translate", label: "Translate", icon: <Languages className="size-4" strokeWidth={1.6} /> },
          { id: "record", label: "Record", icon: <Mic className="size-4" strokeWidth={1.6} /> },
          { id: "review", label: "Review", icon: <ListChecks className="size-4" strokeWidth={1.6} /> },
          { id: "stats", label: "Leaderboard and stats", icon: <BarChart3 className="size-4" strokeWidth={1.6} /> },
        ]}
      />

      <div role="tabpanel" id={`contrib-tabs-panel-${tab}`} aria-labelledby={`contrib-tabs-tab-${tab}`} className="mt-8">
        {!profile.consent && (tab === "translate" || tab === "record") ? (
          <EmptyState
            icon={<ShieldCheck className="size-5" strokeWidth={1.6} />}
            title="One quick step before you contribute"
            description="Choose a licence and confirm your contributions are your own. It takes ten seconds and is stored only on this device."
            action={<Button onClick={() => setConsentOpen(true)}>Review and agree</Button>}
          />
        ) : (
          <>
            {tab === "translate" && <TranslateTab key={effectiveLang} lang={effectiveLang} onSubmitted={bump} />}
            {tab === "record" && <RecordTab key={effectiveLang} lang={effectiveLang} onSubmitted={bump} />}
            {tab === "review" && <ReviewTab key={effectiveLang} lang={effectiveLang} onReviewed={bump} />}
            {tab === "stats" && <StatsTab sessionCount={sessionCount} />}
          </>
        )}
      </div>

      <ConsentModal open={consentOpen} onClose={() => setConsentOpen(false)} onDecline={() => setConsentOpen(false)} />
    </div>
  );
}
